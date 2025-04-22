from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import utils
from torchtune.modules import RotaryPositionalEmbeddings

from .residual_vq import ResidualVQ
from .module import WNConv1d, DecoderBlock, ResLSTM
from .alias_free_torch import *
from . import activations
from . import blocks as blocks
from .bs_roformer5 import TransformerBlock


def init_weights(m):
    """畳み込み層の重みを初期化するための関数である。"""
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class CodecDecoder(nn.Module):
    """
    CodecDecoder モジュールである。
    ResidualVQ を用いて量子化を行い，その後に複数の畳み込みブロックを通して最終出力を生成する構造である。

    引数:
        in_channels (int, optional): 入力チャネル数である。デフォルトは 1024 である。
        upsample_initial_channel (int, optional): アップサンプル前の初期チャネル数である。デフォルトは 1536 である。
        ngf (int, optional): 基本フィルタ数である。デフォルトは 48 である。
        use_rnn (bool, optional): RNN を使用するか否かのフラグである。デフォルトは True である。
        rnn_bidirectional (bool, optional): RNN を双方向にするか否かのフラグである。デフォルトは False である。
        rnn_num_layers (int, optional): RNN の層数である。デフォルトは 2 である。
        up_ratios (tuple, optional): 各層でのアップサンプリング倍率のタプルである。デフォルトは (5, 4, 4, 4, 2) である。
        dilations (tuple, optional): 各層での膨張率のタプルである。デフォルトは (1, 3, 9) である。
        vq_num_quantizers (int, optional): 量子化器の数である。デフォルトは 1 である。
        vq_dim (int, optional): 量子化器の次元である。デフォルトは 2048 である。
        vq_commit_weight (float, optional): 量子化コミットメントの重みである。デフォルトは 0.25 である。
        vq_weight_init (bool, optional): 量子化器の重み初期化の有無である。デフォルトは False である。
        vq_full_commit_loss (bool, optional): 完全コミット損失を使用するか否かのフラグである。デフォルトは False である。
        codebook_size (int, optional): コードブックのサイズである。デフォルトは 16384 である。
        codebook_dim (int, optional): コードブックの次元である。デフォルトは 32 である。
    """
    def __init__(
        self,
        in_channels: int = 1024,
        upsample_initial_channel: int = 1536,
        ngf: int = 48,
        use_rnn: bool = True,
        rnn_bidirectional: bool = False,
        rnn_num_layers: int = 2,
        up_ratios: tuple = (5, 4, 4, 4, 2),
        dilations: tuple = (1, 3, 9),
        vq_num_quantizers: int = 1,
        vq_dim: int = 2048,
        vq_commit_weight: float = 0.25,
        vq_weight_init: bool = False,
        vq_full_commit_loss: bool = False,
        codebook_size: int = 16384,
        codebook_dim: int = 32
        ):
        super().__init__()

        # アップサンプリング後のホップ長は各倍率の積である
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        # 量子化器の初期化
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
        channels = upsample_initial_channel
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]

        if use_rnn:
            layers += [ResLSTM(channels, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional)]

        # DecoderBlock をアップサンプリング倍率に応じて積み重ねる
        for i, stride in enumerate(up_ratios):
            input_dim = channels // (2 ** i)
            output_dim = channels // (2 ** (i + 1))
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations)]

        layers += [
            Activation1d(activation=activations.SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, vq: bool = True) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。
            vq (bool, optional): 量子化を使用するか否かのフラグである。デフォルトは True である。

        戻り値:
            量子化を使用する場合は (x, q, commit_loss) を返し，
            量子化を使用しない場合は self.model(x) を返す。
        """
        if vq is True:
            x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x = self.model(x)
        return x

    def vq2emb(self, vq: torch.Tensor) -> torch.Tensor:
        """
        量子化コードを埋め込みに変換する。

        引数:
            vq (torch.Tensor): 量子化コードである。

        戻り値:
            torch.Tensor: 変換された埋め込みである。
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

    def inference_0(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        量子化器を用いて推論を行い，損失等も返す。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            Tuple[torch.Tensor, Any]:
                - 推論結果
                - None
        """
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        推論を行う。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            Tuple[torch.Tensor, Any]:
                - 推論結果
                - None
        """
        x = self.model(x)
        return x, None

    def remove_weight_norm(self):
        """
        全層から Weight Normalization を除去する。
        """
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """
        全層に Weight Normalization を適用する。
        """
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """
        モデル全体のパラメータを初期化する。
        """
        self.apply(init_weights)


class CodecDecoder_oobleck_Transformer(nn.Module):
    """
    CodecDecoder_oobleck_Transformer モジュールである。
    ResidualVQ を用いて量子化を行い，Transformer ブロックと拡張残差デコーダを組み合わせて
    オーディオ信号を再構成する構造である。

    引数:
        ngf (int, optional): 基本フィルタ数である。デフォルトは 32 である。
        up_ratios (tuple, optional): 各層でのアップサンプリング倍率のタプルである。デフォルトは (5, 4, 4, 4, 2) である。
        dilations (tuple, optional): 各層での膨張率のタプルである。デフォルトは (1, 3, 9) である。
        vq_num_quantizers (int, optional): 量子化器の数である。デフォルトは 1 である。
        vq_dim (int, optional): 量子化器の次元である。デフォルトは 1024 である。
        vq_commit_weight (float, optional): 量子化コミットメントの重みである。デフォルトは 0.25 である。
        vq_weight_init (bool, optional): 量子化器の重み初期化の有無である。デフォルトは False である。
        vq_full_commit_loss (bool, optional): 完全コミット損失を使用するか否かのフラグである。デフォルトは False である。
        codebook_size (int, optional): コードブックのサイズである。デフォルトは 16384 である。
        codebook_dim (int, optional): コードブックの次元である。デフォルトは 16 である。
        hidden_dim (int, optional): Transformer の隠れ次元である。デフォルトは 1024 である。
        depth (int, optional): Transformer ブロックの層数である。デフォルトは 12 である。
        heads (int, optional): Transformer の注意ヘッド数である。デフォルトは 16 である。
        pos_meb_dim (int, optional): ロータリ位置埋め込みの次元である。デフォルトは 64 である。
    """
    def __init__(
        self,
        ngf: int = 32,
        up_ratios: tuple = (5, 4, 4, 4, 2),
        dilations: tuple = (1, 3, 9),
        vq_num_quantizers: int = 1,
        vq_dim: int = 1024,
        vq_commit_weight: float = 0.25,
        vq_weight_init: bool = False,
        vq_full_commit_loss: bool = False,
        codebook_size: int = 16384,
        codebook_dim: int = 16,
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 16,
        pos_meb_dim: int = 64
        ):
        super().__init__()

        # アップサンプリング後のホップ長は各倍率の積である
        self.hop_length = np.prod(up_ratios)
        self.capacity = ngf
        self.up_ratios = up_ratios
        self.hidden_dim = hidden_dim

        # 量子化器の初期化
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

        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
        transformer_blocks = [
            TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
            for _ in range(depth)
        ]
        self.transformers = nn.Sequential(*transformer_blocks)
        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.conv_blocks = blocks.DilatedResidualDecoder(
            capacity=self.capacity,
            dilated_unit=self.dilated_unit,
            upsampling_unit=self.upsampling_unit,
            ratios=up_ratios,
            dilations=dilations,
            pre_network_conv=self.pre_conv,
            post_network_conv=self.post_conv,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor, vq: bool = True) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。
            vq (bool, optional): 量子化を使用するか否かのフラグである。デフォルトは True である。

        戻り値:
            量子化を使用する場合は (x, q, commit_loss) を返し，
            量子化を使用しない場合は変換後のテンソルを返す。
        """
        if vq is True:
            x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x = self.transformers(x)
        x = self.final_layer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        return x

    def vq2emb(self, vq: torch.Tensor) -> torch.Tensor:
        """
        量子化コードを埋め込みに変換する。

        引数:
            vq (torch.Tensor): 量子化コードである。

        戻り値:
            torch.Tensor: 変換された埋め込みである。
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

    def inference_0(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        量子化器を用いた推論を行い，損失等も返す。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            Tuple[torch.Tensor, Any]:
                - 推論結果
                - None
        """
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        推論を行う。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            Tuple[torch.Tensor, Any]:
                - 推論結果
                - None
        """
        x = self.model(x)
        return x, None

    def remove_weight_norm(self):
        """
        全層から Weight Normalization を除去する。
        """
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """
        全層に Weight Normalization を適用する。
        """
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """
        モデル全体のパラメータを初期化する。
        """
        self.apply(init_weights)

    def pre_conv(self, out_channels: int) -> nn.Conv1d:
        """
        前処理用の畳み込み層を生成する。

        引数:
            out_channels (int): 出力チャネル数である。

        戻り値:
            nn.Conv1d: 生成された畳み込み層である。
        """
        return nn.Conv1d(in_channels=self.hidden_dim, out_channels=out_channels, kernel_size=1)

    def post_conv(self, in_channels: int) -> nn.Conv1d:
        """
        後処理用の畳み込み層を生成する。最終出力チャネルは 1 である。

        引数:
            in_channels (int): 入力チャネル数である。

        戻り値:
            nn.Conv1d: 生成された畳み込み層である。
        """
        return nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1)

    def dilated_unit(self, hidden_dim: int, dilation: int) -> nn.Module:
        """
        膨張畳み込みユニットを生成する。

        引数:
            hidden_dim (int): 入力チャネル数である。
            dilation (int): 膨張率である。

        戻り値:
            nn.Module: 生成された膨張畳み込みユニットである。
        """
        return blocks.DilatedConvolutionalUnit(
            hidden_dim=hidden_dim,
            dilation=dilation,
            kernel_size=3,
            activation=nn.ReLU,
            normalization=utils.weight_norm
        )

    def upsampling_unit(self, input_dim: int, output_dim: int, stride: int) -> nn.Module:
        """
        アップサンプリングユニットを生成する。

        引数:
            input_dim (int): 入力チャネル数である。
            output_dim (int): 出力チャネル数である。
            stride (int): ストライドであり，アップサンプリング倍率を決定する。

        戻り値:
            nn.Module: 生成されたアップサンプリングユニットである。
        """
        return blocks.UpsamplingUnit(
            input_dim=input_dim,
            output_dim=output_dim,
            stride=stride,
            activation=nn.ReLU,
            normalization=utils.weight_norm
        )


def main():
    """
    main 関数である。デバイスの設定，モデルの初期化，ダミー入力による前方伝播のテストを行う。
    """
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # モデルの初期化
    model = CodecDecoder_oobleck_Transformer().to(device)
    print("Model initialized.")

    # テスト入力の作成: (batch_size, sequence_length, in_channels)
    batch_size = 2
    in_channels = 1024
    sequence_length = 100  # サンプル長（必要に応じて調整可能）
    dummy_input = torch.randn(batch_size, sequence_length, in_channels).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # モデルを評価モードに設定する
    model.eval()
    output_no_vq = model(dummy_input, vq=False)
    c = 1


if __name__ == "__main__":
    main()
