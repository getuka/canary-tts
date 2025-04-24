from setuptools import setup, find_packages

setup(
    name='canary_tts',               # プロジェクト名
    version='0.1.0',                 # バージョン
    packages=find_packages(include=[
        'canary_tts', 'canary_tts.*'
    ]),
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'torchao',
        'torchtune',
        'transformers==4.44.2',
        'einops',
        'vector-quantize-pytorch',
        'protobuf',
        'accelerate',
        'ffmpeg',
        'soundfile',
        'vector-quantize-pytorch',
        "RubyInserter @ git+https://github.com/getuka/RubyInserter.git"
    ],
)
