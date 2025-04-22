# MIT ライセンスに基づき https://github.com/EdwardDixon/snake から実装を移植した。
# LICENSE は incl_licenses ディレクトリ内にある。

import torch
from torch import nn, sin, pow  # 数学関数 sin, pow をインポートする
from torch.nn import Parameter


class Snake(nn.Module):
    """
    サイン関数に基づく周期的活性化関数の実装である。

    【形状】
        入力： (B, C, T)
        出力： (B, C, T) 入力と同じ形状

    【パラメータ】
        alpha : 学習可能なパラメータ

    【参考文献】
        本活性化関数は Liu Ziyin, Tilman Hartwig, Masahito Ueda による論文に基づく:
        https://arxiv.org/abs/2006.08195

    【使用例】
        >>> a1 = Snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        初期化メソッドである。

        引数:
            in_features (int): 入力の特徴数である。
            alpha (float): 学習可能なパラメータ。デフォルトは 1.0 であり，値が大きいほど高周波となる。
            alpha_trainable (bool): alpha が学習可能か否かを示すフラグである。
            alpha_logscale (bool): alpha を対数スケールで扱うか否かを示すフラグである。
        """
        super(Snake, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            # 対数スケールの場合，alpha はゼロで初期化される
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            # 線形スケールの場合，alpha は1で初期化される
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        # ゼロ除算防止用の極小値
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        """
        順伝播を行うメソッドである。
        入力各要素に対して以下の式を適用する:
            Snake(x) = x + (1/alpha) * sin^2(x * alpha)
        """
        # alpha の次元を入力 x に合わせる [B, C, T] に変換する
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        # 式に基づく計算を行う
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
        return x


class SnakeBeta(nn.Module):
    """
    周期的成分の振幅を制御するために別々のパラメータを用いる修正済みの Snake 関数である。

    【形状】
        入力： (B, C, T)
        出力： (B, C, T) 入力と同じ形状

    【パラメータ】
        alpha : 周波数を制御する学習可能なパラメータである。
        beta  : 振幅を制御する学習可能なパラメータである。

    【参考文献】
        本活性化関数は Liu Ziyin, Tilman Hartwig, Masahito Ueda による論文に基づく:
        https://arxiv.org/abs/2006.08195

    【使用例】
        >>> a1 = SnakeBeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        初期化メソッドである。

        引数:
            in_features (int): 入力の特徴数である。
            alpha (float): 周波数を制御する学習可能なパラメータ。デフォルトは 1.0 である。
            alpha_trainable (bool): alpha および beta が学習可能か否かを示すフラグである。
            alpha_logscale (bool): alpha および beta を対数スケールで扱うか否かを示すフラグである。
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            # 対数スケールの場合，alpha と beta はゼロで初期化される
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.bias = Parameter(torch.zeros(in_features) * alpha)
        else:
            # 線形スケールの場合，alpha と beta は1で初期化される
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.bias = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.bias.requires_grad = alpha_trainable
        # ゼロ除算防止用の極小値
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        """
        順伝播を行うメソッドである。
        入力各要素に対して以下の式を適用する:
            SnakeBeta(x) = x + (1/beta) * sin^2(x * alpha)
        ここで，beta は self.bias に対応する。
        """
        # alpha と beta の次元を入力 x に合わせる [B, C, T] に変換する
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.bias.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        # 式に基づく計算を行う
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
        return x
