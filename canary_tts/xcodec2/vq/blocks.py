from typing import Callable, Sequence, Type, Union
import numpy as np
import torch
import torch.nn as nn

# モジュール生成用の型エイリアスである
ModuleFactory = Union[Type[nn.Module], Callable[[], nn.Module]]


class FeedForwardModule(nn.Module):
    """
    単純なフィードフォワードモジュールである。
    本クラスは nn.Module を継承し，self.net に設定されたネットワークを順伝播させる。
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            torch.Tensor: ネットワークの出力である。
        """
        return self.net(x)


class Residual(nn.Module):
    """
    残差接続を実現するモジュールである。
    入力に対して，module の出力と入力の和を返す。
    """

    def __init__(self, module: nn.Module) -> None:
        """
        初期化メソッドである。

        引数:
            module (nn.Module): 残差接続を適用する対象のモジュールである。
        """
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を実行する。

        引数:
            x (torch.Tensor): 入力テンソルである。

        戻り値:
            torch.Tensor: module(x) と x の和である。
        """
        return self.module(x) + x


class DilatedConvolutionalUnit(FeedForwardModule):
    """
    拡張畳み込みユニットである。
    入力テンソルに対して，活性化関数と正規化層，および拡張畳み込みを適用する。
    """

    def __init__(
            self,
            hidden_dim: int,
            dilation: int,
            kernel_size: int,
            activation: ModuleFactory,
            normalization: Callable[[nn.Module], nn.Module] = lambda x: x) -> None:
        """
        初期化メソッドである。

        引数:
            hidden_dim (int): 入力および出力チャネル数である。
            dilation (int): 拡張率である。
            kernel_size (int): カーネルサイズである。
            activation (ModuleFactory): 活性化関数のモジュールまたは生成関数である。
            normalization (Callable[[nn.Module], nn.Module], optional): 正規化層を生成する関数である。デフォルトは恒等写像である。
        """
        super().__init__()
        self.net = nn.Sequential(
            activation(),
            normalization(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=((kernel_size - 1) * dilation) // 2,
                )),
            activation(),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1
            ),
        )


class UpsamplingUnit(FeedForwardModule):
    """
    アップサンプリングユニットである。
    転置畳み込みを用いて，入力テンソルの時系列長を拡大する。
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            stride: int,
            activation: ModuleFactory,
            normalization: Callable[[nn.Module], nn.Module] = lambda x: x) -> None:
        """
        初期化メソッドである。

        引数:
            input_dim (int): 入力チャネル数である。
            output_dim (int): 出力チャネル数である。
            stride (int): ストライドであり，アップサンプリングの倍率を決定する。
            activation (ModuleFactory): 活性化関数のモジュールまたは生成関数である。
            normalization (Callable[[nn.Module], nn.Module], optional): 正規化層を生成する関数である。デフォルトは恒等写像である。
        """
        super().__init__()
        self.net = nn.Sequential(
            activation(),
            normalization(
                nn.ConvTranspose1d(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=1 if stride % 2 != 0 else 0
                ))
        )


class DownsamplingUnit(FeedForwardModule):
    """
    ダウンサンプリングユニットである。
    畳み込みを用いて，入力テンソルの時系列長を縮小する。
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            stride: int,
            activation: ModuleFactory,
            normalization: Callable[[nn.Module], nn.Module] = lambda x: x) -> None:
        """
        初期化メソッドである。

        引数:
            input_dim (int): 入力チャネル数である。
            output_dim (int): 出力チャネル数である。
            stride (int): ストライドであり，ダウンサンプリングの倍率を決定する。
            activation (ModuleFactory): 活性化関数のモジュールまたは生成関数である。
            normalization (Callable[[nn.Module], nn.Module], optional): 正規化層を生成する関数である。デフォルトは恒等写像である。
        """
        super().__init__()
        self.net = nn.Sequential(
            activation(),
            normalization(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                ))
        )


class DilatedResidualEncoder(FeedForwardModule):
    """
    拡張残差エンコーダである。
    複数の拡張畳み込みユニットとダウンサンプリングユニットを組み合わせたネットワークで，入力の特徴抽出を行う。
    """

    def __init__(
            self,
            capacity: int,
            dilated_unit: Type[DilatedConvolutionalUnit],
            downsampling_unit: Type[DownsamplingUnit],
            ratios: Sequence[int],
            dilations: Union[Sequence[int], Sequence[Sequence[int]]],
            pre_network_conv: Type[nn.Conv1d],
            post_network_conv: Type[nn.Conv1d],
            normalization: Callable[[nn.Module], nn.Module] = lambda x: x) -> None:
        """
        初期化メソッドである。

        引数:
            capacity (int): 基本チャネル数の倍率である。
            dilated_unit (Type[DilatedConvolutionalUnit]): 拡張畳み込みユニットのクラスである。
            downsampling_unit (Type[DownsamplingUnit]): ダウンサンプリングユニットのクラスである。
            ratios (Sequence[int]): 各段におけるダウンサンプリング倍率のシーケンスである。
            dilations (Union[Sequence[int], Sequence[Sequence[int]]]): 各段における拡張率またはそのリストである。
            pre_network_conv (Type[nn.Conv1d]): ネットワーク前処理の畳み込み層のクラスである。
            post_network_conv (Type[nn.Conv1d]): ネットワーク後処理の畳み込み層のクラスである。
            normalization (Callable[[nn.Module], nn.Module], optional): 正規化層を生成する関数である。デフォルトは恒等写像である。
        """
        super().__init__()
        channels = capacity * 2 ** np.arange(len(ratios) + 1)
        dilations_list = self.normalize_dilations(dilations, ratios)

        net = [normalization(pre_network_conv(out_channels=channels[0]))]

        for ratio, dilations, input_dim, output_dim in zip(ratios, dilations_list, channels[:-1], channels[1:]):
            for dilation in dilations:
                net.append(Residual(dilated_unit(input_dim, dilation)))
            net.append(downsampling_unit(input_dim, output_dim, ratio))

        net.append(post_network_conv(in_channels=output_dim))
        self.net = nn.Sequential(*net)

    @staticmethod
    def normalize_dilations(dilations: Union[Sequence[int], Sequence[Sequence[int]]],
                            ratios: Sequence[int]):
        """
        dilations の形式を正規化する。

        引数:
            dilations (Union[Sequence[int], Sequence[Sequence[int]]]): 拡張率のシーケンスまたはリストである。
            ratios (Sequence[int]): ダウンサンプリング倍率のシーケンスである。

        戻り値:
            Sequence[Sequence[int]]: 各段における拡張率のリストである。
        """
        if isinstance(dilations[0], int):
            dilations = [dilations for _ in ratios]
        return dilations


class DilatedResidualDecoder(FeedForwardModule):
    """
    拡張残差デコーダである。
    エンコーダの逆処理を行い，アップサンプリングと拡張畳み込みユニットを組み合わせたネットワークで再構成を行う。
    """

    def __init__(
            self,
            capacity: int,
            dilated_unit: Type[DilatedConvolutionalUnit],
            upsampling_unit: Type[UpsamplingUnit],
            ratios: Sequence[int],
            dilations: Union[Sequence[int], Sequence[Sequence[int]]],
            pre_network_conv: Type[nn.Conv1d],
            post_network_conv: Type[nn.Conv1d],
            normalization: Callable[[nn.Module], nn.Module] = lambda x: x) -> None:
        """
        初期化メソッドである。

        引数:
            capacity (int): 基本チャネル数の倍率である。
            dilated_unit (Type[DilatedConvolutionalUnit]): 拡張畳み込みユニットのクラスである。
            upsampling_unit (Type[UpsamplingUnit]): アップサンプリングユニットのクラスである。
            ratios (Sequence[int]): 各段におけるアップサンプリング倍率のシーケンスである。
            dilations (Union[Sequence[int], Sequence[Sequence[int]]]): 各段における拡張率またはそのリストである。
            pre_network_conv (Type[nn.Conv1d]): ネットワーク前処理の畳み込み層のクラスである。
            post_network_conv (Type[nn.Conv1d]): ネットワーク後処理の畳み込み層のクラスである。
            normalization (Callable[[nn.Module], nn.Module], optional): 正規化層を生成する関数である。デフォルトは恒等写像である。
        """
        super().__init__()
        channels = capacity * 2 ** np.arange(len(ratios) + 1)
        channels = channels[::-1]
        dilations_list = self.normalize_dilations(dilations, ratios)
        dilations_list = dilations_list[::-1]

        net = [pre_network_conv(out_channels=channels[0])]

        for ratio, dilations, input_dim, output_dim in zip(ratios, dilations_list, channels[:-1], channels[1:]):
            net.append(upsampling_unit(input_dim, output_dim, ratio))
            for dilation in dilations:
                net.append(Residual(dilated_unit(output_dim, dilation)))
        net.append(normalization(post_network_conv(in_channels=output_dim)))
        self.net = nn.Sequential(*net)

    @staticmethod
    def normalize_dilations(dilations: Union[Sequence[int], Sequence[Sequence[int]]], ratios: Sequence[int]):
        """
        dilations の形式を正規化する。

        引数:
            dilations (Union[Sequence[int], Sequence[Sequence[int]]]): 拡張率のシーケンスまたはリストである。
            ratios (Sequence[int]): アップサンプリング倍率のシーケンスである。

        戻り値:
            Sequence[Sequence[int]]: 各段における拡張率のリストである。
        """
        if isinstance(dilations[0], int):
            dilations = [dilations for _ in ratios]
        return dilations