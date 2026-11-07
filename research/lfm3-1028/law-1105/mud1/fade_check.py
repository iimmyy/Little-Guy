#!/usr/bin/env python3
"""Engine against Python on a faded file. usage: fade_check.py <hf> <steps.jsonl> <port> <part conv|attn> <gain> [pages=10]"""
import json, random, sys, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
hf, stepsf, port, part, g = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval().requires_grad_(False)
for L, b in enumerate(m.model.layers):
    conv = m.config.layer_types[L] == "conv"
    if conv == (part == "conv"):
        b.post_operator_norm.register_forward_hook(lambda mod, a, o: o * g)
J = [json.loads(l) for l in open(stepsf)]
J = [j for j in J if j.get("split") == "held"]; J.sort(key=lambda j: int(j["step"])); random.Random(7).shuffle(J)
same = n = 0; D = []
for j in J:
    h = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
    ids = tok(h + j["pre"], add_special_tokens=False).input_ids
    if len(ids) > 2500:
        continue
    with torch.inference_mode():
        lp = torch.log_softmax(m(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1)
    top = int(lp.argmax())
    body = json.dumps(dict(prompt=ids, n_predict=1, n_probs=5, temperature=0, cache_prompt=False)).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    tp = r["completion_probabilities"][0]["top_logprobs"]
    et, el = tp[0]["id"], tp[0]["logprob"]
    same += et == top; D.append(abs(el - float(lp[et]))); n += 1
    print(n, tok.decode([top])[:12].replace("\n", "|"), tok.decode([et])[:12].replace("\n", "|"), round(D[-1], 3), flush=True)
    if n == int(sys.argv[6] if len(sys.argv) > 6 else 10):
        break
D.sort()
print(f"FADE_CHECK same top {same} of {n}, median difference {D[len(D)//2]:.3f} nats, largest {D[-1]:.3f}")
sys.exit(0 if same >= n - 1 and D[len(D)//2] <= 0.3 else 1)
