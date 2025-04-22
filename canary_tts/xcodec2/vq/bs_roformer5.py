import torch
import torch.nn as nn
from einops import rearrange
from torchtune.modules import RotaryPositionalEmbeddings


class RMSNorm(nn.Module):
    """
    RMSNorm クラスである。
    このクラスは Llama モデル等で用いられる RMS 正規化を実装している。
    詳細は https://github.com/meta-llama/llama/blob/main/llama/model.py を参照する。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初期化メソッドである。

        引数:
            dim (int): 入力テンソルの最終次元の大きさである。
            eps (float, optional): 数値安定性のための極小値である。デフォルトは 1e-6 である。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            torch.Tensor: 正規化後の出力テンソルである。
        """
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.weight
        return output


class MLP(nn.Module):
    """
    多層パーセプトロン (MLP) を実装するクラスである。
    2 つの線形変換層と SiLU 活性化関数を用いている。
    """

    def __init__(self, dim: int) -> None:
        """
        初期化メソッドである。

        引数:
            dim (int): 入力および出力の次元数である。
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            torch.Tensor: MLP を通過した出力テンソルである。
        """
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """
    注意機構を実装するクラスである。

    このクラスは Flash Attention を利用しており，
    入力テンソルは形状 (batch, time, hidden_dim) を持つものとする。
    """

    def __init__(self, dim: int, n_heads: int, rotary_embed: RotaryPositionalEmbeddings):
        """
        初期化メソッドである。

        引数:
            dim (int): 入力の次元数である。
            n_heads (int): 注意ヘッドの数である。dim は n_heads で割り切れる必要がある。
            rotary_embed (RotaryPositionalEmbeddings): ロータリ位置埋め込みモジュールである。
        """
        super().__init__()
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.dim = dim
        self.rotary_embed = rotary_embed

        # Flash Attention が利用可能か確認する
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "Flash Attention を利用するためには torch.nn.functional.scaled_dot_product_attention が必要である。"

        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルであり，形状は (b, t, h * d) である。
                ここで b はバッチサイズ，t は時間ステップ数，h はヘッド数，d は各ヘッドの次元である。

        戻り値:
            torch.Tensor: 注意機構を適用した後の出力テンソルであり，形状は (b, t, h * d) である。
        """
        B, T, C = x.size()
        # 線形変換後，3 つのテンソル (q, k, v) に分割する．
        q, k, v = rearrange(self.c_attn(x), 'b t (r h d) -> r b h t d', r=3, h=self.n_heads)
        # q, k, v の形状は (b, h, t, d) となる

        q = self.rotary_embed(q)
        k = self.rotary_embed(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)

        y = rearrange(y, 'b h t d -> b t (h d)')
        y = self.c_proj(y)
        # 出力形状は (b, t, h * d) である

        return y


class TransformerBlock(nn.Module):
    """
    Transformer ブロックを実装するクラスである。
    このブロックは RMSNorm による正規化，注意機構，および MLP を備える。
    """

    def __init__(self, dim: int, n_heads: int, rotary_embed: RotaryPositionalEmbeddings):
        """
        初期化メソッドである。

        引数:
            dim (int): 入力および出力の次元数である。
            n_heads (int): 注意ヘッドの数である。
            rotary_embed (RotaryPositionalEmbeddings): ロータリ位置埋め込みモジュールである。
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = Attention(dim=dim, n_heads=n_heads, rotary_embed=rotary_embed)
        self.mlp = MLP(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            torch.Tensor: Transformer ブロックを通過した出力テンソルである。
        """
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


if __name__ == '__main__':
    # ロータリ位置埋め込みモジュールを初期化する（次元は 128 である）
    rotary_embed_128 = RotaryPositionalEmbeddings(dim=128)
    # Transformer ブロックを初期化する（次元は 1024，ヘッド数は 8 である）
    transformer_block = TransformerBlock(
        dim=1024,
        n_heads=8,
        rotary_embed=rotary_embed_128
    )
    # サンプル入力として，バッチサイズ 2，時系列長 128，次元 1024 の乱数テンソルを生成する
    x = torch.randn(2, 128, 1024)
    y = transformer_block(x)
    print(y.shape)