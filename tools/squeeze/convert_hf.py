"""llama.cpp's convert_hf_to_gguf.py, but the model config is read from config.json exactly as published. Newer
transformers (5.15) drops Gemma 4's text_config.global_head_dim when it parses the config, and the converter then
fails on it; reading the file directly keeps every key the model's authors wrote.

usage: PYTHONPATH=<engine>/gguf-py python convert_hf.py <engine dir> <converter args...>
"""
import json, sys
from pathlib import Path

engine = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(engine))
sys.argv[0] = str(engine / "convert_hf_to_gguf.py")

from conversion import base


def load_config_json(dir_model: Path, is_mistral_format: bool):
    with open(Path(dir_model) / ("params.json" if is_mistral_format else "config.json"), encoding="utf-8") as f:
        return json.load(f)


base.ModelBase.load_hparams = staticmethod(load_config_json)

import convert_hf_to_gguf
convert_hf_to_gguf.main()
