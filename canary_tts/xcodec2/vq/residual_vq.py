from typing import List, Union, Any, Tuple
import torch
from torch import nn
from .factorized_vector_quantize import FactorizedVectorQuantize as VQ


class ResidualVQ(nn.Module):
    """残差型ベクトル量子化を実装するクラスである。

    Attributes:
        layers (nn.ModuleList): 各量子化レイヤを保持するモジュールリストである。
        num_quantizers (int): 量子化器の数である。
    """

    def __init__(
        self,
        *,
        num_quantizers: int,
        codebook_size: Union[int, List[int]],
        **kwargs: Any,
    ) -> None:
        """初期化処理を行う関数である。

        Args:
            num_quantizers (int): 量子化器の数である。
            codebook_size (Union[int, List[int]]): コードブックのサイズである。整数の場合は各量子化器に同一のサイズが適用される。
            **kwargs (Any): その他のキーワード引数である。
        """
        super().__init__()

        # codebook_sizeが整数の場合、各量子化器用のリストに変換する
        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * num_quantizers

        # 各量子化器をレイヤとして保持する
        self.layers: nn.ModuleList = nn.ModuleList([VQ(codebook_size=size, **kwargs) for size in codebook_size])
        self.num_quantizers: int = num_quantizers

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """入力テンソルに対して量子化処理を行い、量子化後の出力と各レイヤのインデックスおよび損失を返す関数である。

        Args:
            x (torch.Tensor): 入力テンソルである。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                量子化後の出力テンソル、各レイヤから得られたインデックスのテンソル、各レイヤの損失のテンソルである。
        """
        quantized_out: torch.Tensor = 0.
        residual: torch.Tensor = x

        all_losses: List[torch.Tensor] = []
        all_indices: List[torch.Tensor] = []

        # 各量子化レイヤに対して処理を実施する
        for idx, layer in enumerate(self.layers):
            # レイヤに残差を入力して量子化処理を行う
            quantized, indices, loss = layer(residual)

            # 残差から量子化結果を差し引く
            residual = residual - quantized

            # 量子化出力を累積する
            quantized_out = quantized_out + quantized

            # 損失を平均化する
            loss = loss.mean()

            # インデックスと損失をリストに追加する
            all_indices.append(indices)
            all_losses.append(loss)

        # 各レイヤの損失およびインデックスをテンソルに変換する
        all_losses, all_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, all_indices, all_losses

    def vq2emb(self, vq: torch.Tensor, proj: bool = True) -> torch.Tensor:
        """量子化されたインデックスを埋め込み表現に変換する関数である。

        Args:
            vq (torch.Tensor): 量子化されたインデックスを含むテンソルであり、形状は[B, T, num_quantizers]である。
            proj (bool): 射影処理を行うか否かのフラグである。

        Returns:
            torch.Tensor: 埋め込み表現に変換されたテンソルである。
        """
        quantized_out: torch.Tensor = 0.
        # 各量子化レイヤに対して処理を実施する
        for idx, layer in enumerate(self.layers):
            # 対応するインデックスから埋め込み表現を取得する
            quantized = layer.vq2emb(vq[:, :, idx], proj=proj)
            quantized_out = quantized_out + quantized
        return quantized_out

    def get_emb(self) -> List[torch.Tensor]:
        """各量子化レイヤの埋め込み行列を取得する関数である。

        Returns:
            List[torch.Tensor]: 各レイヤの埋め込み行列のリストである。
        """
        embs = []
        # 各量子化レイヤから埋め込み行列を取得する
        for idx, layer in enumerate(self.layers):
            embs.append(layer.get_emb())
        return embs
