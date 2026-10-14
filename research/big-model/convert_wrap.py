# Run convert_hf_to_gguf with settings that newer transformers drops (e.g. Gemma 4 global_head_dim) restored
# from the model folder's own config.json.
import json, runpy, sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))   # usage: convert_wrap.py <llama.cpp source dir> <convert_hf_to_gguf args>
import conversion.base as base
orig = base.ModelBase.load_hparams
def load_hparams(dir_model, is_mistral_format):
    c = orig(dir_model, is_mistral_format)
    raw = json.loads((Path(dir_model) / "config.json").read_text())
    for k, v in raw.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            for kk, vv in v.items():
                c[k].setdefault(kk, vv)
        else:
            c.setdefault(k, v)
    return c
base.ModelBase.load_hparams = staticmethod(load_hparams)
sys.argv = ["convert_hf_to_gguf.py"] + sys.argv[1:]
runpy.run_path(sys.path[0] + "/convert_hf_to_gguf.py", run_name="__main__")
