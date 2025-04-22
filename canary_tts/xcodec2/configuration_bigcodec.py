from transformers import PretrainedConfig


class BigCodecConfig(PretrainedConfig):
    """BigCodecConfigクラスである。"""

    model_type = "bigcodec"

    def __init__(
        self,
        # 以下は例示のパラメータである
        semantic_hidden_size=1024,
        codec_encoder_hidden_size=1024,
        codec_decoder_hidden_size=1024,
        use_vocos=True,
        **kwargs
    ):
        """初期化メソッドである。

        Args:
            semantic_hidden_size (int): セマンティック層の隠れ状態のサイズである。
            codec_encoder_hidden_size (int): コーデックエンコーダの隠れ状態のサイズである。
            codec_decoder_hidden_size (int): コーデックデコーダの隠れ状態のサイズである。
            use_vocos (bool): vocosの使用有無を示すフラグである。
            **kwargs: その他のキーワード引数である。
        """
        super().__init__(**kwargs)
        self.semantic_hidden_size = semantic_hidden_size
        self.codec_encoder_hidden_size = codec_encoder_hidden_size
        self.codec_decoder_hidden_size = codec_decoder_hidden_size
        self.use_vocos = use_vocos
