"""Collect test thoughts for the dictionary tests by having him READ a text (no generating needed).

Before every word-piece of the text, save the hidden state he would use to pick the next piece.
Also saves the dictionary of the same brain, so thoughts and dictionary always match.

usage: read_collect.py TEXT_FILE BRAIN_GGUF OUT_NAME
  -> .run/results/vocab-trim/<OUT_NAME>/H.npz (one array "read") and E.npy

Sanity check printed at the end: how often his top pick equals the real next piece of the text.
Around half is normal for reading real text; near zero would mean the saved thoughts are wrong.
"""
import json, subprocess, sys, time, urllib.request
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
sys.path.insert(0, f"{RUN}/lcpp-src/gguf-py")
from gguf import GGUFReader
from gguf.quants import dequantize

TEXT, BRAIN, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = Path(f"{RUN}/results/vocab-trim/{NAME}")
OUT.mkdir(parents=True, exist_ok=True)
SERVER = f"{RUN}/lcpp-src/build-mac/bin/llama-server"
CHUNK = 1500                                                 # word-pieces per read, cut at paragraph breaks
PORT = 8089


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


# CPU only, 4 threads: the board's setup
srv = subprocess.Popen([SERVER, "-m", BRAIN, "-ngl", "0", "-t", "4", "-c", "4096", "-ub", "2048", "-b", "2048",
                        "--embeddings", "--pooling", "none", "--port", str(PORT), "--no-webui"],
                       stdout=open(OUT / "server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception:
            time.sleep(1)

    chunks, cur = [], ""
    for para in open(TEXT).read().split("\n\n"):
        if cur and len(post("/tokenize", {"content": cur + "\n\n" + para})["tokens"]) > CHUNK:
            chunks.append(cur); cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    chunks.append(cur)

    rows, nxt = [], []
    t0 = time.time()
    for i, c in enumerate(chunks):
        toks = post("/tokenize", {"content": c, "add_special": True})["tokens"]
        out = post("/embedding", {"content": c})
        e = np.array(out[0]["embedding"] if isinstance(out, list) else out["embedding"], dtype=np.float32)
        rows.append(e[:-1])                                  # the thought before each next piece
        nxt.extend(toks[1:len(e)])
        print(f"chunk {i + 1}/{len(chunks)}: {len(e)} pieces [{time.time() - t0:.0f} s]", flush=True)
finally:
    srv.terminate(); srv.wait()

H = np.concatenate(rows)
np.savez(OUT / "H.npz", read=H)
r = GGUFReader(BRAIN)
t = next(t for t in r.tensors if t.name == "token_embd.weight")
E = dequantize(t.data, t.tensor_type).reshape(-1, 2560).astype(np.float32)
np.save(OUT / "E.npy", E)

nxt = np.array(nxt[:len(H)])
top = np.concatenate([(H[i:i + 2048] @ E.T).argmax(1) for i in range(0, len(H), 2048)])
print(f"{len(H)} thoughts saved; his top pick = the real next piece {np.mean(top == nxt):.1%} of the time")
