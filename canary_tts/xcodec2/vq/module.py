# 標準ライブラリ
from typing import Optional, Tuple

# サードパーティライブラリ
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm

# 自作モジュール
from . import activations
from .alias_free_torch import Activation1d


def WNConv1d(*args, **kwargs) -> nn.Conv1d:
    """重み正規化を適用した１次元畳み込み層を返す関数である．"""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs) -> nn.ConvTranspose1d:
    """重み正規化を適用した１次元逆畳み込み層を返す関数である．"""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    """Residual Unitである．"""

    def __init__(self, dim: int = 16, dilation: int = 1) -> None:
        """
        コンストラクタである．

        Args:
            dim (int): 入力チャネル数である．デフォルトは16である．
            dilation (int): 拡張率である．デフォルトは1である．
        """
        super().__init__()
        pad = ((7 - 1) * dilation) // 2  # パディング値の計算
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            x (torch.Tensor): 入力テンソルである．

        Returns:
            torch.Tensor: 出力テンソルである．
        """
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """エンコーダブロックである．"""

    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        dilations: Tuple[int, ...] = (1, 3, 9)
        ) -> None:
        """
        コンストラクタである．

        Args:
            dim (int): 入力チャネル数である．デフォルトは16である．
            stride (int): ストライドである．デフォルトは1である．
            dilations (tuple[int, ...]): 拡張率のタプルである．デフォルトは(1, 3, 9)である．
        """
        super().__init__()
        # 残差ユニットのリストを生成する．
        runits = [ResidualUnit(dim // 2, dilation=d) for d in dilations]
        self.block = nn.Sequential(
            *runits,
            Activation1d(activation=activations.SnakeBeta(
                dim // 2, alpha_logscale=True)),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            x (torch.Tensor): 入力テンソルである．

        Returns:
            torch.Tensor: 出力テンソルである．
        """
        return self.block(x)


class DecoderBlock(nn.Module):
    """デコーダブロックである．"""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        dilations: Tuple[int, ...] = (1, 3, 9)
    ) -> None:
        """
        コンストラクタである．

        Args:
            input_dim (int): 入力チャネル数である．デフォルトは16である．
            output_dim (int): 出力チャネル数である．デフォルトは8である．
            stride (int): ストライドである．デフォルトは1である．
            dilations (tuple[int, ...]): 拡張率のタプルである．デフォルトは(1, 3, 9)である．
        """
        super().__init__()
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(input_dim, alpha_logscale=True)),
            WNConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=stride // 2 + stride % 2, output_padding=stride % 2)
        )
        # 残差ユニットを追加する．
        self.block.extend([ResidualUnit(output_dim, dilation=d) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            x (torch.Tensor): 入力テンソルである．

        Returns:
            torch.Tensor: 出力テンソルである．
        """
        return self.block(x)


class ResLSTM(nn.Module):
    """残差接続付きLSTMである．"""

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        skip: bool = True
    ) -> None:
        """
        コンストラクタである．

        Args:
            dimension (int): 入力次元である．
            num_layers (int): LSTMの層数である．デフォルトは2である．
            bidirectional (bool): 双方向LSTMを用いるか否かである．デフォルトはFalseである．
            skip (bool): 残差接続を行うか否かである．デフォルトはTrueである．
        """
        super().__init__()
        self.skip: bool = skip
        self.lstm: nn.LSTM = nn.LSTM(
            dimension,
            dimension if not bidirectional else dimension // 2,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            x (torch.Tensor): 入力テンソル，形状は(Batch, 特徴, 時刻)である．

        Returns:
            torch.Tensor: 出力テンソル，形状は(Batch, 特徴, 時刻)である．
        """
        x = rearrange(x, "b f t -> b t f")
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = rearrange(y, "b t f -> b f t")
        return y


class ConvNeXtBlock(nn.Module):
    """
    １次元音声信号に適用するために変形したConvNeXt Blockである．

    Args:
        dim (int): 入力チャネル数である．
        intermediate_dim (int): 中間層の次元数である．
        layer_scale_init_value (float): 層スケールの初期値である．0より大きい場合にスケーリングを適用する．
        adanorm_num_embeddings (Optional[int]): AdaLayerNorm用の埋め込み数である．Noneの場合は条件なしのLayerNormを用いる．
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dwconv: nn.Conv1d = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm: bool = adanorm_num_embeddings is not None

        if adanorm_num_embeddings:
            self.norm: nn.Module = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.pwconv1: nn.Linear = nn.Linear(dim, intermediate_dim)
        self.act: nn.GELU = nn.GELU()
        self.pwconv2: nn.Linear = nn.Linear(intermediate_dim, dim)
        self.gamma: Optional[nn.Parameter] = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            x (torch.Tensor): 入力テンソルである．
            cond_embedding_id (Optional[torch.Tensor]): 条件付き埋め込みIDである．

        Returns:
            torch.Tensor: 出力テンソルである．
        """
        residual: torch.Tensor = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (Batch, チャネル, 時刻)から(Batch, 時刻, チャネル)に変換する．
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (Batch, 時刻, チャネル)から(Batch, チャネル, 時刻)に変換する．
        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    埋め込みを用いた適応型Layer Normalizationである．

    Args:
        num_embeddings (int): 埋め込み数である．
        embedding_dim (int): 埋め込みの次元数である．
        eps (float): 正規化時のイプシロン値である．デフォルトは1e-6である．
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps
        self.dim: int = embedding_dim
        self.scale: nn.Embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift: nn.Embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            x (torch.Tensor): 入力テンソルである．
            cond_embedding_id (torch.Tensor): 条件付き埋め込みIDである．

        Returns:
            torch.Tensor: 正規化後の出力テンソルである．
        """
        scale: torch.Tensor = self.scale(cond_embedding_id)
        shift: torch.Tensor = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x


class ResBlock1(nn.Module):
    """
    HiFi-GAN V1から変形した，アップサンプリング層を持たない拡張付き１次元畳み込みのResBlockである．

    Args:
        dim (int): 入力チャネル数である．
        kernel_size (int): 畳み込みカーネルのサイズである．デフォルトは3である．
        dilation (Tuple[int, int, int]): 拡張率のタプルである．デフォルトは(1, 3, 5)である．
        lrelu_slope (float): LeakyReLU活性化関数の負の傾きである．デフォルトは0.1である．
        layer_scale_init_value (Optional[float]): 層スケールの初期値である．Noneの場合はスケーリングを行わない．
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.lrelu_slope: float = lrelu_slope
        self.convs1: nn.ModuleList = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=dilation[0], padding=self.get_padding(kernel_size, dilation[0]))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=dilation[1], padding=self.get_padding(kernel_size, dilation[1]))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=dilation[2], padding=self.get_padding(kernel_size, dilation[2]))),
            ]
        )
        self.convs2: nn.ModuleList = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
            ]
        )
        self.gamma: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            x (torch.Tensor): 入力テンソルである．

        Returns:
            torch.Tensor: 出力テンソルである．
        """
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        """
        重み正規化を削除する．
        """
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        """
        畳み込みのためのパディング値を計算する．

        Args:
            kernel_size (int): カーネルサイズである．
            dilation (int): 拡張率である．デフォルトは1である．

        Returns:
            int: 計算されたパディング値である．
        """
        return int((kernel_size * dilation - dilation) / 2)


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    入力テンソルの各要素の対数を，極小値でクリップして計算する関数である．

    Args:
        x (torch.Tensor): 入力テンソルである．
        clip_val (float): クリップする最小値である．デフォルトは1e-7である．

    Returns:
        torch.Tensor: クリップ後の対数値である．
    """
    return torch.log(torch.clip(x, min=clip_val))


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    対称対数関数である．

    Args:
        x (torch.Tensor): 入力テンソルである．

    Returns:
        torch.Tensor: 対称対数変換後のテンソルである．
    """
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    対称指数関数である．

    Args:
        x (torch.Tensor): 入力テンソルである．

    Returns:
        torch.Tensor: 対称指数変換後のテンソルである．
    """
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


class SemanticEncoder(nn.Module):
    """セマンティックエンコーダである．"""

    def __init__(
        self,
        input_channels: int,
        code_dim: int,
        encode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        """
        コンストラクタである．

        Args:
            input_channels (int): 入力チャネル数である．
            code_dim (int): コードの次元数である．
            encode_channels (int): エンコードチャネル数である．
            kernel_size (int): 畳み込みカーネルのサイズである．デフォルトは3である．
            bias (bool): バイアスを用いるか否かである．デフォルトはTrueである．
        """
        super().__init__()
        # 初期畳み込み： input_channels を encode_channels に写像する．
        self.initial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        # 残差ブロック
        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias
            )
        )
        # 最終畳み込み： encode_channels を code_dim に写像する．
        self.final_conv = nn.Conv1d(
            in_channels=encode_channels,
            out_channels=code_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            x (torch.Tensor): 入力テンソル，形状は (Batch, 入力チャネル数, 長さ) である．

        Returns:
            torch.Tensor: エンコード後のテンソル，形状は (Batch, コード次元, 長さ) である．
        """
        x = self.initial_conv(x)              # (Batch, エンコードチャネル数, 長さ)
        x = self.residual_blocks(x) + x       # 残差接続
        x = self.final_conv(x)                # (Batch, コード次元, 長さ)
        return x


class SemanticDecoder(nn.Module):
    """セマンティックデコーダである．"""

    def __init__(
        self,
        code_dim: int,
        output_channels: int,
        decode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        """
        コンストラクタである．

        Args:
            code_dim (int): コードの次元数である．
            output_channels (int): 出力チャネル数である．
            decode_channels (int): デコードチャネル数である．
            kernel_size (int): 畳み込みカーネルのサイズである．デフォルトは3である．
            bias (bool): バイアスを用いるか否かである．デフォルトはTrueである．
        """
        super().__init__()
        # 初期畳み込み： code_dim を decode_channels に写像する．
        self.initial_conv = nn.Conv1d(
            in_channels=code_dim,
            out_channels=decode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        # 残差ブロック
        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                decode_channels,
                decode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                decode_channels,
                decode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias
            )
        )
        # 最終畳み込み： decode_channels を output_channels に写像する．
        self.final_conv = nn.Conv1d(
            in_channels=decode_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理である．

        Args:
            z (torch.Tensor): 入力テンソル，形状は (Batch, コード次元, 長さ) である．

        Returns:
            torch.Tensor: デコード後のテンソル，形状は (Batch, 出力チャネル数, 長さ) である．
        """
        x = self.initial_conv(z)         # (Batch, デコードチャネル数, 長さ)
        x = self.residual_blocks(x) + x  # 残差接続
        x = self.final_conv(x)           # (Batch, 出力チャネル数, 長さ)
        return x