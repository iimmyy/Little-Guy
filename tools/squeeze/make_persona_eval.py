"""Persona eval for the squeeze pipeline: gold replies from a reference brain (e.g. the 4-bit QAT) to
persona_prompts.txt with the persona as system message, greedy, then one long chat transcript in the model's own
chat template (rendered by llama-server's /apply-template) for llama-perplexity --kl-divergence with LG_PPL_SPECIAL=1.
The transcript holds the private persona: keep it out of git.

usage: make_persona_eval.py <server url> <out.txt> [max_tokens]
"""
import json, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/behavior"))
from common import load_persona

url, out = sys.argv[1], Path(sys.argv[2])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 120
prompts = [l.strip() for l in (Path(__file__).parent / "persona_prompts.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]


def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


persona = load_persona()
msgs = [{"role": "system", "content": persona}]
for q in prompts:   # each reply is generated fresh (single turn), then the transcript strings them together
    r = post("/v1/chat/completions", {"messages": [{"role": "system", "content": persona}, {"role": "user", "content": q}],
                                      "max_tokens": max_tokens, "temperature": 0, "seed": 1})
    msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": r["choices"][0]["message"]["content"].strip()}]
text = post("/apply-template", {"messages": msgs})["prompt"]
out.write_text(text)
print(f"{len(prompts)} exchanges, {len(text)} chars -> {out}")
