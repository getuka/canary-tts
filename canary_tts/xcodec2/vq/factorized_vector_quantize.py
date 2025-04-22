from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange


class FactorizedVectorQuantize(nn.Module):
    """固定されたコードブックを用いて入力テンソルを量子化し，対応するコードブックベクトルを返すモジュールである．"""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float,
        **kwargs: Any
        ) -> None:
        """
        コンストラクタ

        Args:
            dim (int): 入力特徴数
            codebook_size (int): コードブックのサイズ
            codebook_dim (int): コードブック内の各エントリの次元
            commitment (float): コミットメント損失の係数
            **kwargs: その他のキーワード引数
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment

        # 入力とコードブックの次元が異なる場合，線形変換を用いる
        if dim != self.codebook_dim:
            self.in_proj = weight_norm(nn.Linear(dim, self.codebook_dim))
            self.out_proj = weight_norm(nn.Linear(self.codebook_dim, dim))
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        self._codebook = nn.Embedding(codebook_size, self.codebook_dim)

    @property
    def codebook(self) -> nn.Embedding:
        """コードブックの埋め込み層を返す．"""
        return self._codebook

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        入力テンソルを固定されたコードブックで量子化し，
        対応するコードブックベクトル，コードブックインデックス，
        及びコミットメント損失を返す．

        Args:
            z (torch.Tensor): 入力テンソル，形状は [B, D, T]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - 量子化された連続表現，形状は [B, D, T]
                - コードブックのインデックス（離散表現），形状は [B, T]
                - コミットメント損失

        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        # 線形層のため，次元を入れ替える
        z = rearrange(z, "b d t -> b t d")

        # 低次元空間へ投影
        z_e = self.in_proj(z)  # z_e の形状は (B x T x D)
        z_e = rearrange(z_e, "b t d -> b d t")
        z_q, indices = self.decode_latents(z_e)

        # 学習時はコミットメント損失とコードブック損失を計算する
        if self.training:
            commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2]) * self.commitment
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
            commit_loss = commitment_loss + codebook_loss
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)

        # 直線的な勾配推定器を用いる（順伝播はそのまま，逆伝播では勾配を流す）
        z_q = z_e + (z_q - z_e).detach()

        # 出力の次元を元に戻す
        z_q = rearrange(z_q, "b d t -> b t d")
        z_q = self.out_proj(z_q)
        z_q = rearrange(z_q, "b t d -> b d t")

        return z_q, indices, commit_loss

    def vq2emb(self, vq: torch.Tensor, proj: bool = True) -> torch.Tensor:
        """
        量子化されたインデックスから埋め込みベクトルを取得する．

        Args:
            vq (torch.Tensor): 量子化されたインデックス
            proj (bool, optional): 出力に線形変換を適用するか否か．デフォルトは True

        Returns:
            torch.Tensor: 埋め込みベクトル
        """
        emb = self.embed_code(vq)
        if proj:
            emb = self.out_proj(emb)
        return emb

    def get_emb(self) -> torch.Tensor:
        """
        コードブックの重みを返す．

        Returns:
            torch.Tensor: コードブックの重み
        """
        return self.codebook.weight

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """
        インデックスに対応する埋め込みベクトルを取得する．

        Args:
            embed_id (torch.Tensor): 埋め込みインデックス

        Returns:
            torch.Tensor: 埋め込みベクトル
        """
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """
        インデックスから埋め込みベクトルを取得し，次元を入れ替えて返す．

        Args:
            embed_id (torch.Tensor): 埋め込みインデックス

        Returns:
            torch.Tensor: 次元が入れ替わった埋め込みベクトル
        """
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        潜在表現をコードブックの埋め込みに変換する．
        L2正規化を用いて各ベクトル間のユークリッド距離を計算し，
        最も近いコードブックのエントリを選択する．

        Args:
            latents (torch.Tensor): 潜在表現，形状は [B, D, T]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 変換後の埋め込みベクトル，形状は [B, D, T]
                - コードブックのインデックス，形状は [B, T]
        """
        # (B x T) x D に変換する
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook の形状は (N x D)

        # エンコーディングとコードブックを L2 正規化する
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # コードブックとのユークリッド距離を計算する
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        # 最大値（距離が最小）のインデックスを取得する
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices
