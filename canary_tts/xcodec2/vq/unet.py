from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """畳み込みブロックを実装するクラスである。

    Attributes:
        bn1 (nn.BatchNorm2d): １番目のバッチ正規化層である。
        bn2 (nn.BatchNorm2d): ２番目のバッチ正規化層である。
        conv1 (nn.Conv2d): １番目の畳み込み層である。
        conv2 (nn.Conv2d): ２番目の畳み込み層である。
        shortcut (nn.Conv2d): ショートカット用の畳み込み層である（必要な場合）。
        is_shortcut (bool): ショートカットが必要か否かのフラグである。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)) -> None:
        super(ConvBlock, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播処理を行う関数である。

        Args:
            x (torch.Tensor): 入力テンソルである。

        Returns:
            torch.Tensor: 出力テンソルである。
        """
        h = self.conv1(F.leaky_relu_(self.bn1(x)))
        h = self.conv2(F.leaky_relu_(self.bn2(h)))
        if self.is_shortcut:
            return self.shortcut(x) + h
        else:
            return x + h


class EncoderBlock(nn.Module):
    """エンコーダーブロックを実装するクラスである。

    Attributes:
        pool_size (int): プーリングのサイズである。
        conv_block (ConvBlock): 畳み込みブロックである。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)) -> None:
        super(EncoderBlock, self).__init__()
        self.pool_size = 2
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播処理を行う関数である。

        Args:
            x (torch.Tensor): 入力テンソルである。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 出力テンソルと潜在特徴テンソルである。
        """
        latent = self.conv_block(x)
        output = F.avg_pool2d(latent, kernel_size=self.pool_size)
        return output, latent


class DecoderBlock(nn.Module):
    """デコーダーブロックを実装するクラスである。

    Attributes:
        upsample (nn.ConvTranspose2d): アップサンプリング用の転置畳み込み層である。
        conv_block (ConvBlock): 畳み込みブロックである。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)) -> None:
        super(DecoderBlock, self).__init__()
        stride = 2  # ストライドの設定である
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=stride,
            stride=stride,
            padding=(0, 0),
            bias=False,
        )
        self.conv_block = ConvBlock(in_channels * 2, out_channels, kernel_size)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """順伝播処理を行う関数である。

        Args:
            x (torch.Tensor): 入力テンソルである。
            latent (torch.Tensor): エンコーダーからの潜在特徴テンソルである。

        Returns:
            torch.Tensor: 出力テンソルである。
        """
        x = self.upsample(x)
        x = torch.cat((x, latent), dim=1)
        output = self.conv_block(x)
        return output


class UNet(nn.Module):
    """UNetモデルを実装するクラスである。

    Attributes:
        downsample_ratio (int): ダウンサンプリングの比率である。
        encoder_block1 (EncoderBlock): １番目のエンコーダーブロックである。
        encoder_block2 (EncoderBlock): ２番目のエンコーダーブロックである。
        encoder_block3 (EncoderBlock): ３番目のエンコーダーブロックである。
        encoder_block4 (EncoderBlock): ４番目のエンコーダーブロックである。
        middle (EncoderBlock): ミドルブロックである。
        decoder_block1 (DecoderBlock): １番目のデコーダーブロックである。
        decoder_block2 (DecoderBlock): ２番目のデコーダーブロックである。
        decoder_block3 (DecoderBlock): ３番目のデコーダーブロックである。
        decoder_block4 (DecoderBlock): ４番目のデコーダーブロックである。
        fc (nn.Linear): 全結合層である。
    """

    def __init__(self, freq_dim: int = 1281, out_channel: int = 1024) -> None:
        super(UNet, self).__init__()
        self.downsample_ratio: int = 16
        in_channels = 1  # 音声チャネル数である

        self.encoder_block1 = EncoderBlock(in_channels, 16)
        self.encoder_block2 = EncoderBlock(16, 64)
        self.encoder_block3 = EncoderBlock(64, 256)
        self.encoder_block4 = EncoderBlock(256, 1024)
        self.middle = EncoderBlock(1024, 1024)
        self.decoder_block1 = DecoderBlock(1024, 256)
        self.decoder_block2 = DecoderBlock(256, 64)
        self.decoder_block3 = DecoderBlock(64, 16)
        self.decoder_block4 = DecoderBlock(16, 16)

        self.fc = nn.Linear(freq_dim * 16, out_channel)

    def forward(self, x_ori: torch.Tensor) -> torch.Tensor:
        """順伝播処理を行う関数である。

        Args:
            x_ori (torch.Tensor): 入力複素数テンソルであり，形状は (B, C, T, F) である。

        Returns:
            torch.Tensor: 出力複素数テンソルである。
        """
        x = self.process_image(x_ori)
        x1, latent1 = self.encoder_block1(x)
        x2, latent2 = self.encoder_block2(x1)
        x3, latent3 = self.encoder_block3(x2)
        x4, latent4 = self.encoder_block4(x3)
        _, h = self.middle(x4)
        x5 = self.decoder_block1(h, latent4)
        x6 = self.decoder_block2(x5, latent3)
        x7 = self.decoder_block3(x6, latent2)
        x8 = self.decoder_block4(x7, latent1)
        x = self.unprocess_image(x8, x_ori.shape[2])
        x = x.permute(0, 2, 1, 3).contiguous()  # 形状を [B, T, C, F] に変換する
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        return x

    def process_image(self, x: torch.Tensor) -> torch.Tensor:
        """周波数スペクトログラムをダウンサンプリング可能な形状に整形する関数である。

        Args:
            x (torch.Tensor): 入力テンソルであり，形状は (B, C, T, F) である。

        Returns:
            torch.Tensor: 前処理後のテンソルである。
        """
        B, C, T, freq = x.shape
        pad_len = int(np.ceil(T / self.downsample_ratio)) * self.downsample_ratio - T
        # 下方向にパディングを行う
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # 周波数軸の末尾１チャネルを削除する
        output = x[:, :, :, 0:freq - 1]
        return output

    def unprocess_image(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """周波数スペクトログラムを元の形状に復元する関数である。

        Args:
            x (torch.Tensor): 前処理後のテンソルである。
            time_steps (int): 元の時間ステップ数である。

        Returns:
            torch.Tensor: 復元後のテンソルである。
        """
        # 周波数軸の右側に１チャネル分のパディングを行う
        x = F.pad(x, pad=(0, 1))
        output = x[:, :, 0:time_steps, :]
        return output


def test_unet() -> None:
    """UNetモデルの動作確認を行う関数である。

    テストとして，ランダムな複素数テンソルを入力し，出力の形状および複素数テンソルであることを確認する。
    """
    # 入力パラメータの設定である
    batch_size = 6
    channels = 1  # 音声チャネル数である
    time_steps = 256  # 時間ステップ数である
    freq_bins = 1024  # 周波数ビン数である

    # ランダムなテンソルを生成する（実部と虚部）
    real_part = torch.randn(batch_size, channels, time_steps, freq_bins)
    imag_part = torch.randn(batch_size, channels, time_steps, freq_bins)
    # 複素数テンソルの生成は環境に依存するため，実部のみを使用する
    complex_sp = real_part  # torch.complex(real_part, imag_part)

    # UNetモデルのインスタンスを生成する
    model = UNet()

    # 順伝播処理を実施する
    output = model(complex_sp)

    # 入出力の形状を表示する
    print("入力形状:", complex_sp.shape)
    print("出力形状:", output.shape)

    # 出力が複素数テンソルであることを確認する
    assert torch.is_complex(output), "出力が複素数テンソルではない"
    # 出力形状が入力形状と一致することを確認する
    assert output.shape == complex_sp.shape, "出力形状が入力形状と一致しない"

    print("テスト成功 モデルは正常に動作する")


if __name__ == "__main__":
    test_unet()
