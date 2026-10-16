"""Split the calibration text by whole documents: every HOLD-th conversation or article goes to a held-out section at
the END of the file, never used to fit anything, only to check that matching generalizes to text it has not seen.
Writes calib.txt (fit documents, then held-out documents), calib_fit.txt (fit only, for the importance matrix) and
calib_split.json with the row (token position) where the held-out section starts, measured with the model's own
tokenizer through a running llama-server (same tokenizer settings as llama-imatrix: special tokens on, BOS added).

usage: split_calib.py <server url> <dir with calib_chat.txt and calib_plain.txt> [HOLD]
"""
import json, sys, urllib.request
from pathlib import Path

url, d = sys.argv[1], Path(sys.argv[2])
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def load(name, chat):
    return [f"<|turn>user\n{x['prompt']}<turn|>\n<|turn>model\n{x['reply']}<turn|>\n" if chat else x["reply"] + "\n\n"
            for x in json.loads((d / f"{name}.txt.json").read_text())]


chat, plain = load("calib_chat", True), load("calib_plain", False)
docs = chat + plain
fit = [x for i, x in enumerate(docs) if i % HOLD != HOLD - 1]
hold = [x for i, x in enumerate(docs) if i % HOLD == HOLD - 1]


def n_tokens(text):
    body = {"content": text, "add_special": True, "parse_special": True}
    req = urllib.request.Request(url + "/tokenize", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(req, timeout=600))["tokens"])


fit_text, hold_text = "".join(fit), "".join(hold)
(d / "calib.txt").write_text(fit_text + hold_text)
(d / "calib_fit.txt").write_text(fit_text)
info = {"documents": len(docs), "chat": len(chat), "articles": len(plain), "fit_docs": len(fit), "hold_docs": len(hold),
        "every": HOLD, "hold_from_row": n_tokens(fit_text), "tokens": n_tokens(fit_text + hold_text)}
(d / "calib_split.json").write_text(json.dumps(info, indent=1))
print(info)
