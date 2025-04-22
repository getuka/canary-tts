import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings
from vector_quantize_pytorch import ResidualFSQ

from .residual_vq import ResidualVQ
from .alias_free_torch import *
from .bs_roformer5 import TransformerBlock


class ISTFT(nn.Module):
    """
    カスタム ISTFT の実装である。
    torch.istft はウィンドウ処理時のカスタムパディング（center 以外）を許容しないため，
    CNN に類似した "same" パディングを実現するために実装したものである。

    引数:
        n_fft (int): フーリエ変換のサイズである。
        hop_length (int): 隣接する窓フレーム間の距離である。
        win_length (int): 窓フレームおよび STFT フィルターのサイズである。
        padding (str, optional): パディングの種類。 "center" または "same" を指定する。デフォルトは "same" である。
    """
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        複素スペクトログラムの逆短時間フーリエ変換 (ISTFT) を計算する。

        引数:
            spec (torch.Tensor): 入力複素スペクトログラム，形状は (B, N, T) で，B はバッチサイズ，N は周波数ビン数，T は時間フレーム数である。

        戻り値:
            torch.Tensor: 再構成された時系列信号，形状は (B, L) で，L は出力信号の長さである。
        """
        if self.padding == "center":
            # PyTorch 標準の実装にフォールバックする
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "入力は 3 次元テンソルである必要がある。"
        B, N, T = spec.shape

        # 逆 FFT の計算
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add の実行
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length)
        )[:, 0, 0, pad:-pad]

        # ウィンドウエンベロープの計算
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length)).squeeze()[pad:-pad]

        # 正規化
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class FourierHead(nn.Module):
    """
    逆フーリエ変換モジュールの基底クラスである。
    このクラスはサブクラスにおいて forward メソッドを実装することを要求する。
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x (torch.Tensor): 入力テンソル，形状は (B, L, H) である。

        戻り値:
            torch.Tensor: 再構成された時系列オーディオ信号，形状は (B, T) である。
        """
        raise NotImplementedError("サブクラスで forward メソッドを実装する必要がある。")


class ISTFTHead(FourierHead):
    """
    STFT 複素係数を予測するための ISTFT Head モジュールである。

    引数:
        dim (int): モデルの隠れ次元である。
        n_fft (int): フーリエ変換のサイズである。
        hop_length (int): 入力特徴の解像度に合わせた隣接フレーム間の距離である。
        padding (str, optional): パディングの種類。 "center" または "same" を指定する。デフォルトは "same" である。
    """
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ISTFTHead モジュールの順伝播を行う。

        引数:
            x (torch.Tensor): 入力テンソル，形状は (B, L, H) である。

        戻り値:
            torch.Tensor: 再構成された時系列オーディオ信号，形状は (B, T) である。
        """
        x_pred = self.out(x)
        # x_pred = x
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # 過度に大きな振幅を防ぐためのセーフガード
        # 実数部と虚数部を生成する
        x_cos = torch.cos(p)
        x_sin = torch.sin(p)
        # 直接複素値を生成する
        S = mag * (x_cos + 1j * x_sin)
        audio = self.istft(S)
        return audio.unsqueeze(1), x_pred


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """
    Swish 活性化関数である。

    引数:
        x (torch.Tensor): 入力テンソルである。

    戻り値:
        torch.Tensor: x * sigmoid(x) を計算した出力である。
    """
    return x * torch.sigmoid(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    """
    入力チャネル数に基づき Group Normalization モジュールを生成する。

    引数:
        in_channels (int): 入力チャネル数である。
        num_groups (int, optional): グループ数である。デフォルトは 32 である。

    戻り値:
        nn.Module: Group Normalization モジュールである。
    """
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """
    ResNet ブロックである。
    畳み込みと正規化，活性化関数を用い，残差接続を実現する。

    引数:
        in_channels (int): 入力チャネル数である。
        out_channels (int, optional): 出力チャネル数である。指定しない場合は in_channels と同じである。
        conv_shortcut (bool): ショートカットに畳み込みを使用するか否かを示すフラグである。
        dropout (float): ドロップアウト率である。
        temb_channels (int, optional): テンポラル埋め込みのチャネル数である。デフォルトは 512 である。
    """
    def __init__(self, *, in_channels: int, out_channels: int = None, conv_shortcut: bool = False, dropout: float, temb_channels: int = 512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。
            temb (torch.Tensor, optional): テンポラル埋め込みである。存在しない場合は None である。

        戻り値:
            torch.Tensor: ResNet ブロックを通過した出力である。
        """
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """
    注意機構ブロックである。
    入力に対して自己注意機構を適用し，特徴を強調する。

    引数:
        in_channels (int): 入力チャネル数である。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            torch.Tensor: 注意機構を適用した後の出力である。
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 注意重みの計算
        b, c, h = q.shape
        q = q.permute(0, 2, 1)  # 形状: (b, hw, c)
        w_ = torch.bmm(q, k)  # 形状: (b, hw, hw)  ; w[b,i,j] = sum_c q[b,i,c]*k[b,c,j]
        w_ = w_ * (c ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 値への注意の適用
        w_ = w_.permute(0, 2, 1)  # 形状: (b, hw, hw)
        h_ = torch.bmm(v, w_)  # 形状: (b, c, hw)
        h_ = self.proj_out(h_)
        return x + h_


def make_attn(in_channels: int, attn_type: str = "vanilla") -> nn.Module:
    """
    注意機構のタイプに基づき注意ブロックを生成するファクトリ関数である。

    引数:
        in_channels (int): 入力チャネル数である。
        attn_type (str, optional): 注意機構のタイプ。 "vanilla", "linear", "none" のいずれかを指定する。デフォルトは "vanilla" である。

    戻り値:
        nn.Module: 生成された注意ブロックである。
    """
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} は不明である。"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)


class Backbone(nn.Module):
    """
    ジェネレータのバックボーンの基底クラスである。
    全層で同一の時系列解像度を保持する。
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        引数:
            x (torch.Tensor): 入力テンソル，形状は (B, C, L) である。 B はバッチサイズ，C は出力特徴数，L はシーケンス長である。

        戻り値:
            torch.Tensor: 出力テンソル，形状は (B, L, H) である。 H はモデル次元である。
        """
        raise NotImplementedError("サブクラスで forward メソッドを実装する必要がある。")


class VocosBackbone(Backbone):
    """
    ConvNeXt ブロックを用いて構築された Vocos バックボーンモジュールである。
    Adaptive Layer Normalization による追加の条件付けをサポートする。

    引数:
        input_channels (int): 入力特徴チャネル数である。
        dim (int): モデルの隠れ次元である。
        intermediate_dim (int): ConvNeXtBlock で用いる中間次元である。
        num_layers (int): ConvNeXtBlock の層数である。
        layer_scale_init_value (float, optional): レイヤースケーリングの初期値である。デフォルトは 1 / num_layers である。
        adanorm_num_embeddings (int, optional): AdaLayerNorm 用の埋め込み数である。条件付けしない場合は None とする。デフォルトは None である。
    """
    def __init__(self, hidden_dim: int = 1024, depth: int = 12, heads: int = 16, pos_meb_dim: int = 64):
        super().__init__()

        # 入力特徴の埋め込み
        self.embed = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)

        self.temb_ch = 0
        block_in = hidden_dim
        dropout = 0.1

        # 事前ネットワーク（prior_net）の定義
        prior_net: list[nn.Module] = [
            ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout),
            ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout),
        ]
        self.prior_net = nn.Sequential(*prior_net)

        # Transformer ブロックの定義
        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
        transformer_blocks = [
            TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
            for _ in range(depth)
        ]
        self.transformers = nn.Sequential(*transformer_blocks)
        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # 事後ネットワーク（post_net）の定義
        post_net: list[nn.Module] = [
            ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout),
            ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout),
        ]
        self.post_net = nn.Sequential(*post_net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        VocosBackbone の順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソル，形状は (B, C, L) である。

        戻り値:
            torch.Tensor: 出力テンソル，形状は (B, L, H) である。
        """
        x = x.transpose(1, 2)
        x = self.embed(x)
        x = self.prior_net(x)
        x = x.transpose(1, 2)
        x = self.transformers(x)
        x = x.transpose(1, 2)
        x = self.post_net(x)
        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)
        return x


def init_weights(m: nn.Module) -> None:
    """
    畳み込み層の重みを初期化するための関数である。

    引数:
        m (nn.Module): 初期化対象のモジュールである。
    """
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class CodecDecoderVocos(nn.Module):
    """
    CodecDecoderVocos モジュールである。
    ResidualFSQ を用いて量子化を行い，VocosBackbone および ISTFTHead を用いてオーディオ信号を再構成する。

    引数:
        hidden_dim (int, optional): モデルの隠れ次元である。デフォルトは 1024 である。
        depth (int, optional): Transformer の層数である。デフォルトは 12 である。
        heads (int, optional): Transformer のヘッド数である。デフォルトは 16 である。
        pos_meb_dim (int, optional): ロータリ位置埋め込みの次元である。デフォルトは 64 である。
        hop_length (int, optional): hop_length である。デフォルトは 320 である。
        vq_num_quantizers (int, optional): 量子化器の数である。デフォルトは 1 である。
        vq_dim (int, optional): 量子化器の次元である。デフォルトは 2048 である。
        vq_commit_weight (float, optional): 量子化コミットメントの重みである。デフォルトは 0.25 である。
        vq_weight_init (bool, optional): 量子化器の重み初期化の有無である。デフォルトは False である。
        vq_full_commit_loss (bool, optional): 完全コミット損失を用いるか否か。デフォルトは False である。
        codebook_size (int, optional): コードブックのサイズである。デフォルトは 16384 である。
        codebook_dim (int, optional): コードブックの次元である。デフォルトは 16 である。
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 16,
        pos_meb_dim: int = 64,
        hop_length: int = 320,
        vq_num_quantizers: int = 1,
        vq_dim: int = 2048,
        vq_commit_weight: float = 0.25,
        vq_weight_init: bool = False,
        vq_full_commit_loss: bool = False,
        codebook_size: int = 16384,
        codebook_dim: int = 16
        ):
        super().__init__()
        self.hop_length = hop_length

        self.quantizer = ResidualFSQ(
            dim=vq_dim,
            levels=[4, 4, 4, 4, 4, 4, 4, 4],
            num_quantizers=1
        )
        # 以下は ResidualVQ を利用する場合の実装例である
        # self.quantizer = ResidualVQ(
        #     num_quantizers=vq_num_quantizers,
        #     dim=vq_dim,
        #     codebook_size=codebook_size,
        #     codebook_dim=codebook_dim,
        #     threshold_ema_dead_code=2,
        #     commitment=vq_commit_weight,
        #     weight_init=vq_weight_init,
        #     full_commit_loss=vq_full_commit_loss,
        # )

        self.backbone = VocosBackbone(hidden_dim=hidden_dim, depth=depth, heads=heads, pos_meb_dim=pos_meb_dim)
        self.head = ISTFTHead(dim=hidden_dim, n_fft=self.hop_length * 4, hop_length=self.hop_length, padding="same")

        self.reset_parameters()

    def forward(self, x: torch.Tensor, vq: bool = True):
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。
            vq (bool, optional): 量子化を使用するか否かのフラグである。デフォルトは True である。

        戻り値:
            量子化を使用する場合: (x, q, None)
            量子化を使用しない場合: (x, _)  を返す。
        """
        if vq is True:
            # 入力の次元を変更してから量子化器に入力する
            x = x.permute(0, 2, 1)
            x, q = self.quantizer(x)
            x = x.permute(0, 2, 1)
            q = q.permute(0, 2, 1)
            return x, q, None
        x = self.backbone(x)
        x, _ = self.head(x)
        return x, _

    def vq2emb(self, vq: torch.Tensor) -> torch.Tensor:
        """
        量子化コードを埋め込みに変換する。

        引数:
            vq (torch.Tensor): 量子化コードである。

        戻り値:
            torch.Tensor: 量子化器によって変換された埋め込みである。
        """
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self) -> torch.Tensor:
        """
        量子化器の埋め込みを取得する。

        戻り値:
            torch.Tensor: 量子化器の埋め込みである。
        """
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq: torch.Tensor) -> torch.Tensor:
        """
        量子化器を用いた推論を行う。

        引数:
            vq (torch.Tensor): 入力量子化コードである。

        戻り値:
            torch.Tensor: 推論結果である。
        """
        x = vq[None, :, :]
        x = self.model(x)
        return x

    def inference_0(self, x: torch.Tensor):
        """
        量子化器を用いて推論を行う（詳細な損失等も返す）。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            推論結果および損失（None を返す場合もある）。
        """
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x: torch.Tensor):
        """
        推論を行う。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            推論結果と None を返す。
        """
        x = self.model(x)
        return x, None

    def remove_weight_norm(self) -> None:
        """
        全層から Weight Normalization を除去する。
        """
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # このモジュールには weight norm が適用されていない
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self) -> None:
        """
        全層に Weight Normalization を適用する。
        """
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def reset_parameters(self) -> None:
        """
        モデル全体のパラメータを初期化する。
        """
        self.apply(init_weights)


class CodecDecoderVocos_transpose(nn.Module):
    """
    CodecDecoderVocos_transpose モジュールである。
    ResidualVQ を用いて量子化を行い，VocosBackbone に加え逆変換（ConvTranspose1d）を用いてオーディオ信号を再構成する。

    引数は CodecDecoderVocos とほぼ同一である。
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 16,
        pos_meb_dim: int = 64,
        hop_length: int = 320,
        vq_num_quantizers: int = 1,
        vq_dim: int = 1024,
        vq_commit_weight: float = 0.25,
        vq_weight_init: bool = False,
        vq_full_commit_loss: bool = False,
        codebook_size: int = 16384,
        codebook_dim: int = 16
        ):
        super().__init__()
        self.hop_length = hop_length

        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
        )

        self.backbone = VocosBackbone(hidden_dim=hidden_dim, depth=depth, heads=heads, pos_meb_dim=pos_meb_dim)

        self.inverse_mel_conv = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1  # 入力長と一致させるための調整
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            )
        )

        self.head = ISTFTHead(dim=hidden_dim, n_fft=self.hop_length * 4, hop_length=self.hop_length, padding="same")

        self.reset_parameters()

    def forward(self, x: torch.Tensor, vq: bool = True):
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。
            vq (bool, optional): 量子化を使用するか否かのフラグである。デフォルトは True である。

        戻り値:
            量子化を使用する場合: (x, q, commit_loss)
            量子化を使用しない場合: (x, _) を返す。
        """
        if vq is True:
            x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x = self.backbone(x)
        x, _ = self.head(x)
        return x, _

    def vq2emb(self, vq: torch.Tensor) -> torch.Tensor:
        """
        量子化コードを埋め込みに変換する。

        引数:
            vq (torch.Tensor): 量子化コードである。

        戻り値:
            torch.Tensor: 量子化器によって変換された埋め込みである。
        """
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self) -> torch.Tensor:
        """
        量子化器の埋め込みを取得する。

        戻り値:
            torch.Tensor: 量子化器の埋め込みである。
        """
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq: torch.Tensor) -> torch.Tensor:
        """
        量子化器を用いた推論を行う。

        引数:
            vq (torch.Tensor): 入力量子化コードである。

        戻り値:
            torch.Tensor: 推論結果である。
        """
        x = vq[None, :, :]
        x = self.model(x)
        return x

    def inference_0(self, x: torch.Tensor):
        """
        量子化器を用いた推論を行う（詳細な損失等も返す）。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            推論結果および損失（None を返す場合もある）。
        """
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x: torch.Tensor):
        """
        推論を行う。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            推論結果と None を返す。
        """
        x = self.model(x)
        return x, None

    def remove_weight_norm(self) -> None:
        """
        全層から Weight Normalization を除去する。
        """
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self) -> None:
        """
        全層に Weight Normalization を適用する。
        """
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def reset_parameters(self) -> None:
        """
        モデル全体のパラメータを初期化する。
        """
        self.apply(init_weights)


def main():
    """
    main 関数である。デバイスの設定，モデルの初期化，ダミー入力による前方伝播のテストを行う。
    """
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # モデルの初期化
    model = CodecDecoderVocos_transpose().to(device)
    print("Model initialized.")

    # テスト入力の作成: (batch_size, in_channels, sequence_length)
    batch_size = 2
    in_channels = 1024
    sequence_length = 50  # サンプルの長さ（必要に応じて調整可能）
    dummy_input = torch.randn(batch_size, in_channels, sequence_length).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # モデルを評価モードに設定する
    model.eval()

    # 量子化を使用しない場合の前方伝播
    with torch.no_grad():
        output_no_vq = model(dummy_input, vq=False)
        print("\nForward pass without VQ:")
        print(f"Output shape: {output_no_vq.shape}")


if __name__ == "__main__":
    main()