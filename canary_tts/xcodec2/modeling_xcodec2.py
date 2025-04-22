import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoFeatureExtractor, Wav2Vec2BertModel

from .configuration_bigcodec import BigCodecConfig
from .vq.codec_encoder import CodecEncoder_Transformer
from .vq.codec_decoder_vocos import CodecDecoderVocos
from .vq.module import SemanticEncoder


class XCodec2Model(PreTrainedModel):
    """XCodec2Modelクラスである。

    このクラスは音声のエンコード・デコードを行うモデルである。
    """

    config_class = BigCodecConfig

    def __init__(self, config: BigCodecConfig):
        """初期化メソッドである。

        Args:
            config (BigCodecConfig): モデル設定を保持するコンフィグである。
        """
        super().__init__(config)

        # 1) セマンティックモデル
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0",
            output_hidden_states=True
        )
        self.semantic_model.eval()

        self.SemanticEncoder_module = SemanticEncoder(
            config.semantic_hidden_size,
            config.semantic_hidden_size,
            config.semantic_hidden_size
        )

        # 2) コーデックエンコーダ
        self.CodecEnc = CodecEncoder_Transformer()

        # 3) コーデックデコーダ
        self.generator = CodecDecoderVocos()

        # 4) 2つの全結合層
        self.fc_prior = nn.Linear(2048, 2048)
        self.fc_post_a = nn.Linear(2048, 1024)

        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.feature_extractor = feature_extractor

    def forward(self, input_waveform, sample_rate=16000):
        """forward メソッドである。

        このメソッドは必ずしも forward と呼ばれる必要はなく，別のメソッドに分割してもよいが，
        pipeline と互換性を持たせるためには forward 内に核心となるロジックを記述する必要がある。

        Args:
            input_waveform (Tensor): [batch_size, waveform_length] の形状を持つ入力波形である。
            sample_rate (int, optional): サンプリングレート。デフォルトは 16000 である。

        Returns:
            Tensor: 再構成された音声波形である。
        """
        # 1) 特徴抽出
        # 必要に応じてパディング処理を実施可能である
        input_features = self.feature_extractor(
            input_waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)  # [batch, frames, feat_dim]

        # 2) セマンティック層
        semantic_output = self.semantic_model(input_features)
        semantic_hidden_16 = semantic_output.hidden_states[16]  # 16層目を使用
        semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
        semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

        # 3) コーデックエンコーダ
        wav = input_waveform.unsqueeze(1).to(self.device)  # shape: [batch, 1, time]
        vq_emb = self.CodecEnc(wav)  # [batch, time//down, 1024] （例示）
        vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

        # セマンティックベクトルの時間フレーム数を揃える（例示のための単純な処理）
        if vq_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(vq_emb.shape[-1], semantic_encoded.shape[-1])
            vq_emb = vq_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]

        # 4) 結合
        concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 1024 + 1024, frames]

        # 5) fc_prior（全結合層）
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        # 6) デコーダの量子化部分
        _, vq_code, _ = self.generator(concat_emb, vq=True)
        vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
        vq_post_emb = vq_post_emb.transpose(1, 2)

        # 7) fc_post_a（全結合層）
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)

        # 8) 最終的に波形へデコードする
        recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]

        # recon_audio: [batch, time]
        return recon_audio

    def encode_code(self, input_waveform, sample_rate=16000):
        """入力された音声をコード表現にエンコードする。

        Args:
            input_waveform (Tensor): [batch_size, waveform_length] の形状を持つ入力音声である。
            sample_rate (int, optional): サンプリングレート。デフォルトは 16000 である。

        Returns:
            Tensor: エンコードされたコードである。
        """
        with torch.no_grad():
            # 1) 特徴抽出
            input_features = self.feature_extractor(
                [x.cpu().numpy() for x in input_waveform],
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)  # [batch, frames, feat_dim]

            # 2) セマンティック層
            semantic_output = self.semantic_model(input_features)
            semantic_hidden_16 = semantic_output.hidden_states[16]  # 16層目を使用
            semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
            semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

            # 3) コーデックエンコーダ
            wav = input_waveform.unsqueeze(1).to(self.device)  # shape: [batch, 1, time]
            vq_emb = self.CodecEnc(wav)  # [batch, time//down, 1024] （例示）
            vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

            # セマンティックベクトルの時間フレーム数を揃える（例示のための単純な処理）
            if vq_emb.shape[-1] != semantic_encoded.shape[-1]:
                min_len = min(vq_emb.shape[-1], semantic_encoded.shape[-1])
                vq_emb = vq_emb[:, :, :min_len]
                semantic_encoded = semantic_encoded[:, :, :min_len]

            # 例: semantic_encoded.shape torch.Size([1, 1024, 848])
            # 例: vq_emb.shape torch.Size([32, 1024, 848])

            # 4) 結合
            concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 2048, frames]

            # 5) fc_prior（全結合層）
            concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

            # 6) デコーダの量子化部分を実行しコードを取得
            _, vq_code, _ = self.generator(concat_emb, vq=True)

            # vq_code: [batch, frames]
            return vq_code

    def decode_code(self, vq_code):
        """エンコードされたコードを音声にデコードする。

        Args:
            vq_code (Tensor): エンコードされたコードである。[batch, frames] の形状を持つ。

        Returns:
            Tensor: デコードされた音声波形である。[batch, waveform_length] の形状である。
        """
        with torch.no_grad():
            # 量子化後の埋め込みを取得
            vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
            vq_post_emb = vq_post_emb.transpose(1, 2)  # [batch, 1024, frames]

            # 7) fc_post_a（全結合層）
            vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)  # [batch, 1024, frames]

            # 8) 最終的に波形へデコードする
            recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]  # [batch, time]
            return recon_audio
