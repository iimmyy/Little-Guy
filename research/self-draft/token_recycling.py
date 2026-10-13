"""Token Recycling (Luo et al., ACL 2025) on Little Guy: free guesses from his own leftovers, measured before building.

Every time the brain scores a position it already knows its top-k next words. Keep them in a small table:
after word w, the brain's most recent top-k candidates. To guess ahead from the current word, walk that table
(a chain, or a small tree). The full brain then checks all guesses in one pass and keeps the longest correct run,
so nothing he says changes (greedy here; with his dice the hit rate is lower).

Simulation (teacher-forced, exact for greedy): his full brain writes greedy replies with the persona on; we read
prompt + reply once to get the top-k at every position; the table starts filled from the prompt (persona + question)
and is updated with every checked position as the reply goes on.

Cost model from the board (2026-10-13, brain C + dictionary preview, 4 big cores): checking n words at once costs
1.00 / 1.48 / 2.58 / 4.09 x one word for n = 1 / 4 / 8 / 16 (2 and 3 padded up to 4).

usage: token_recycling.py [n_prompts]
"""
import json, sys, time, subprocess, urllib.request, os
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/behavior"))
from common import SERVER, LOGS, load_persona
from scoring import QUESTIONS

BRAIN = str(ROOT / ".run/weights/experiments/clean-Q2_K-C.gguf")
DICT = ROOT / ".run/results/vocab-trim/ufs-writeup/E.npy"
PORT, TOPK = 8097, 8
COST = {1: 1.00, 2: 1.48, 3: 1.48, 4: 1.48, 8: 2.58, 16: 4.09}
PROMPTS = list(QUESTIONS.values()) + [
    "Tell me a short story about a robot who learns to bake bread.", "Explain how a bike gear works, simply.",
    "Give me three ideas for a rainy Sunday.", "What's a good way to fall asleep faster?",
    "Can you help me plan a cheap weekend trip?", "Why do cats knead blankets?",
    "I'm nervous about a job interview tomorrow.", "What should I know before getting a puppy?",
    "hello im immy", "what do you think of this pen here?", "how was your day?", "can you remind me what we talked about?"]
# draft shapes: list of (parent index, which candidate rank); index 0 is the current word (already known)
SHAPES = {
    "chain 1":      [(0, 0)],
    "chain 2":      [(0, 0), (1, 0)],
    "chain 3":      [(0, 0), (1, 0), (2, 0)],
    "tree 3 (2+1)": [(0, 0), (0, 1), (1, 0)],
    "tree 7":       [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (4, 0)],
    "tree 15":      [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0),
                     (5, 0), (5, 1), (6, 0), (8, 0), (11, 0)],
}


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def serve(embeddings):
    args = [str(SERVER), "-m", BRAIN, "-ngl", "0", "-t", "8", "-c", "4096", "--port", str(PORT), "--no-webui", "--reasoning", "off"]
    if embeddings:
        args += ["--embeddings", "--pooling", "none", "-ub", "4096", "-b", "4096"]
    p = subprocess.Popen(args, stdout=open(LOGS / "token_recycling.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); return p
        except Exception:
            time.sleep(1)
    raise SystemExit("server failed")


def simulate(toks, top, n_prompt, shape):
    """walk the reply; returns (words produced, checks made) for one draft shape"""
    table = {}
    for i in range(n_prompt - 1):           # warm the table from the prompt: after toks[i], he expected top[i]
        table[int(toks[i])] = top[i]
    p, words, checks = n_prompt - 1, 0, 0  # p = index of the current (last known) word
    end = len(toks) - 1
    while p < end:
        nodes = [int(toks[p])]
        parents = [-1]
        for par, rank in shape:
            cand = table.get(nodes[par])
            nodes.append(int(cand[rank]) if cand is not None and rank < len(cand) else -1)
            parents.append(par)
        # longest accepted path: node i is accepted if its parent is accepted and it equals the real next word
        depth = {0: 0}
        best = 0
        for i in range(1, len(nodes)):
            par = parents[i]
            if par in depth and nodes[i] >= 0 and p + depth[par] + 1 <= end and nodes[i] == toks[p + depth[par] + 1]:
                depth[i] = depth[par] + 1
                best = max(best, depth[i])
        n_new = best + 1                      # accepted guesses + the word the check itself produces
        for q in range(p, min(p + n_new, end)):
            table[int(toks[q])] = top[q]      # the check scored these positions, so recycle their top-k
        p += n_new
        words += n_new
        checks += 1
    return words, checks


srv = serve(False)
texts = []
try:
    for q in PROMPTS[:int(sys.argv[1]) if len(sys.argv) > 1 else len(PROMPTS)]:
        prompt = post("/apply-template", {"messages": [{"role": "system", "content": load_persona()}, {"role": "user", "content": q}],
                                          "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        reply = post("/completion", {"prompt": prompt, "n_predict": 160, "temperature": 0, "cache_prompt": False})["content"]
        n_prompt = len(post("/tokenize", {"content": prompt, "add_special": True})["tokens"])
        toks = np.array(post("/tokenize", {"content": prompt + reply, "add_special": True})["tokens"])
        texts.append((prompt + reply, n_prompt, toks))
finally:
    srv.terminate(); srv.wait()

E = np.load(DICT)
srv = serve(True)
data = []
try:
    for text, n_prompt, toks in texts:
        r = post("/embedding", {"content": text})
        h = np.array(r[0]["embedding"] if isinstance(r, list) else r["embedding"], dtype=np.float32)
        top = np.argsort(-(h @ E.T), axis=1)[:, :TOPK]     # top-k next words at every position
        sane = np.mean(top[n_prompt - 1:-1, 0] == toks[n_prompt:])
        data.append((toks, top, n_prompt, sane))
finally:
    srv.terminate(); srv.wait()

print(f"{len(data)} greedy replies, {sum(len(t) - n for t, _, n, _ in data)} reply words; "
      f"sanity (top word = greedy reply) {np.mean([d[3] for d in data]):.1%}\n")
print(f"{'draft shape':14s} {'guesses':>7s} {'words/check':>11s} {'check cost':>10s} {'speedup':>8s}")
for name, shape in SHAPES.items():
    w = c = 0
    for toks, top, n_prompt, _ in data:
        a, b = simulate(toks, top, n_prompt, shape)
        w += a; c += b
    n_verify = len(shape) + 1
    cost = COST.get(n_verify) or COST[min(k for k in COST if k >= n_verify)]
    print(f"{name:14s} {len(shape):7d} {w / c:11.2f} {cost:9.2f}x {w / c / cost:7.2f}x", flush=True)
