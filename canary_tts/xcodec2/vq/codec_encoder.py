from typing import Tuple

import torch
from torch import nn
import numpy as np
from torch.nn import utils
from torchtune.modules import RotaryPositionalEmbeddings

from .module import WNConv1d, EncoderBlock, ResLSTM
from .alias_free_torch import *
from .bs_roformer5 import TransformerBlock
from . import activations
from . import blocks as blocks


def init_weights(m: nn.Module):
    """重みの初期化を行う関数である。

    畳み込み層の場合は，切断正規分布に従って重みを初期化し，バイアスを0に設定する。
    """
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class CodecEncoder(nn.Module):
    """CodecEncoderモジュールである。

    畳み込み層とEncoderBlock，および必要に応じてRNNを組み合わせ，
    最終的な出力を生成するエンコーダである。
    """

    def __init__(
        self,
        ngf: int = 48,
        use_rnn: bool =True,
        rnn_bidirectional: bool =False,
        rnn_num_layers: int = 2,
        up_ratios: tuple = (2, 2, 4, 4, 5),
        dilations: tuple = (1, 3, 9),
        out_channels: int =1024
        ):
        """コンストラクタである。

        Args:
            ngf (int): 基本チャネル数。
            use_rnn (bool): RNNを使用するか否か。
            rnn_bidirectional (bool): 双方向RNNを使用するか否か。
            rnn_num_layers (int): RNNの層数。
            up_ratios (tuple): アップサンプリング比率。
            dilations (tuple): ダイレーション値。
            out_channels (int): 出力チャネル数。
        """
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        # 最初の畳み込み層を作成する
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # ストライドによりダウンサンプリングする際にチャネル数を倍増するEncoderBlockを作成する
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, dilations=dilations)]

        # RNNを使用する場合
        if use_rnn:
            self.block += [ResLSTM(d_model, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional)]

        # 最後の畳み込み層を作成する
        self.block += [
            Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        # ブロックをnn.Sequentialにまとめる
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        out = self.block(x)
        return out

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """推論時の順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        return self.block(x)

    def remove_weight_norm(self):
        """全層からウェイトノルムを削除する。"""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # このモジュールはウェイトノルムを持たない
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """全層にウェイトノルムを適用する。"""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """全パラメータの初期化を行う。"""
        self.apply(init_weights)


class Transpose(nn.Module):
    """テンソルの次元を入れ替えるモジュールである。"""

    def __init__(self, dim1: int, dim2: int):
        """コンストラクタである。

        Args:
            dim1 (int): 入れ替える次元1。
            dim2 (int): 入れ替える次元2。
        """
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を行い，指定された次元を入れ替える。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 次元が入れ替えられたテンソル。
        """
        return x.transpose(self.dim1, self.dim2)


class CodecEncoder_Transformer(nn.Module):
    """Transformerを用いたCodecEncoderモジュールである。"""

    def __init__(
        self,
        ngf: int = 48,
        up_ratios: list = [2, 2, 4, 4, 5],
        dilations: tuple = (1, 3, 9),
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 12,
        pos_meb_dim: int = 64
        ):
        """コンストラクタである。

        Args:
            ngf (int): 基本チャネル数。
            up_ratios (list): アップサンプリング比率。
            dilations (tuple): ダイレーション値。
            hidden_dim (int): 隠れ層の次元数。
            depth (int): Transformer層の深さ。
            heads (int): アテンションヘッド数。
            pos_meb_dim (int): 位置エンベディングの次元数。
        """
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        d_model = ngf
        self.conv_blocks = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.conv_blocks += [EncoderBlock(d_model, stride=stride, dilations=dilations)]

        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        # 以下のTransformer関連のコードはコメントアウトしている
        # time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
        # transformer_blocks = [
        #     TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
        #     for _ in range(depth)
        # ]
        # self.transformers = nn.Sequential(*transformer_blocks)
        # self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.conv_final_block = [
            Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, hidden_dim, kernel_size=3, padding=1),
        ]
        self.conv_final_block = nn.Sequential(*self.conv_final_block)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        x = self.conv_blocks(x)
        x = self.conv_final_block(x)
        x = x.permute(0, 2, 1)
        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """推論時の順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        return self.block(x)

    def remove_weight_norm(self):
        """全層からウェイトノルムを削除する。"""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # このモジュールはウェイトノルムを持たない
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """全層にウェイトノルムを適用する。"""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """全パラメータの初期化を行う。"""
        self.apply(init_weights)


class Codec_oobleck_Transformer(nn.Module):
    """Transformerを用いたCodec_oobleckモジュールである。"""

    def __init__(
        self,
        ngf: int = 32,
        up_ratios: tuple = (2, 2, 4, 4, 5),
        dilations: tuple = (1, 3, 9),
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 16,
        pos_meb_dim: int = 64
        ):
        """コンストラクタである。

        Args:
            ngf (int): 基本チャネル数。
            up_ratios (tuple): アップサンプリング比率。
            dilations (tuple): ダイレーション値。
            hidden_dim (int): 隠れ層の次元数。
            depth (int): Transformer層の深さ。
            heads (int): アテンションヘッド数。
            pos_meb_dim (int): 位置エンベディングの次元数。
        """
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.hidden_dim = hidden_dim

        # DilatedResidualEncoderを用いて畳み込みブロックを作成する
        self.conv_blocks = blocks.DilatedResidualEncoder(
            capacity=ngf,
            dilated_unit=self.dilated_unit,
            downsampling_unit=self.downsampling_unit,
            ratios=up_ratios,
            dilations=dilations,
            pre_network_conv=self.pre_conv,
            post_network_conv=self.post_conv,
        )

        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)

        transformer_blocks = [
            TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
            for _ in range(depth)
        ]

        self.transformers = nn.Sequential(*transformer_blocks)
        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        x = self.transformers(x)
        x = self.final_layer_norm(x)
        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """推論時の順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        return self.block(x)

    def remove_weight_norm(self):
        """全層からウェイトノルムを削除する。"""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # このモジュールはウェイトノルムを持たない
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """全層にウェイトノルムを適用する。"""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """全パラメータの初期化を行う。"""
        self.apply(init_weights)

    def dilated_unit(self, hidden_dim: int, dilation: int) -> nn.Module:
        """拡張畳み込みユニットを生成する。

        Args:
            hidden_dim (int): 入力チャネル数。
            dilation (int): ダイレーション値。

        Returns:
            nn.Module: 拡張畳み込みユニット。
        """
        return blocks.DilatedConvolutionalUnit(
            hidden_dim,
            dilation,
            kernel_size=3,
            activation=nn.ReLU,
            normalization=utils.weight_norm
        )

    def downsampling_unit(self, input_dim: int, output_dim: int, stride: int) -> nn.Module:
        """ダウンサンプリングユニットを生成する。

        Args:
            input_dim (int): 入力チャネル数。
            output_dim (int): 出力チャネル数。
            stride (int): ストライド値。

        Returns:
            nn.Module: ダウンサンプリングユニット。
        """
        return blocks.DownsamplingUnit(
            input_dim,
            output_dim,
            stride,
            nn.ReLU,
            normalization=utils.weight_norm
        )

    def pre_conv(self, out_channels: int) -> nn.Conv1d:
        """前処理用の畳み込み層を生成する。

        Args:
            out_channels (int): 出力チャネル数。

        Returns:
            nn.Conv1d: 畳み込み層。
        """
        return nn.Conv1d(1, out_channels, 1)

    def post_conv(self, in_channels: int) -> nn.Conv1d:
        """後処理用の畳み込み層を生成する。

        Args:
            in_channels (int): 入力チャネル数。

        Returns:
            nn.Conv1d: 畳み込み層。
        """
        return nn.Conv1d(in_channels, self.hidden_dim, 1)


class CodecEncoder_only_Transformer(nn.Module):
    """Transformerのみを用いたCodecEncoderモジュールである。"""

    def __init__(
        self,
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 16,
        pos_meb_dim: int = 64
        ):
        """コンストラクタである。

        Args:
            hidden_dim (int): 隠れ層の次元数。
            depth (int): Transformer層の深さ。
            heads (int): アテンションヘッド数。
            pos_meb_dim (int): 位置エンベディングの次元数。
        """
        super().__init__()
        # 以下，入力を線形変換する場合の例（現在はコメントアウト）
        # self.embed = nn.Linear(input_dim, hidden_dim)

        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
        transformer_blocks = [TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed) for _ in range(depth)]
        self.transformers = nn.Sequential(*transformer_blocks)
        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        # x = self.embed(x)
        x = self.transformers(x)
        x = self.final_layer_norm(x)
        return x


def get_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    """モデルのサイズを計算する関数である。

    モデルの全パラメータ数およびサイズ（MB）を返す。

    Args:
        model (torch.nn.Module): 対象のモデル。

    Returns:
        Tuple[int, float]:
            - int: 全パラメータ数。
            - float: モデルサイズ（MB）。
    """
    # 全パラメータ数を計算する
    total_params = sum(p.numel() for p in model.parameters())

    # 各パラメータは32ビット（4バイト）と仮定してモデルサイズを計算する
    model_size_bytes = total_params  # 各パラメータ4バイト
    model_size_mb = model_size_bytes / (1024 ** 2)

    return total_params, model_size_mb


if __name__ == '__main__':
    # Codec_oobleck_Transformerのインスタンス生成
    model = Codec_oobleck_Transformer()
    # 入力テンソルの例
    x = torch.randn(1, 1, 16000)
    output = model(x)
    print("出力形状:", output.shape)
