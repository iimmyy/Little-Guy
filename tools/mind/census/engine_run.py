#!/usr/bin/env python3
"""Answers from a stock llama-server (no hooks, no patches), in gen_patient.py's row format so math_table.py reads
them unchanged, plus the engine's own timing per answer (ChatGPT's test, 2026-10-28: untouched vs baked model through
the same unmodified engine, same prompts, same sampling; length, accuracy, seconds).

The prompt is the model's own chat template text (same as gen_patient.py), sent raw to /completion, so both models
see the same tokens. Thinking words are counted with the model's tokenizer.

usage: engine_run.py <server url> <problems.jsonl> <out.jsonl> <hf folder for template + tokenizer>
                     [maxnew=6000 sample=T,top_p,top_k seed=0 (leave out for greedy) chat=1 par=1]
chat=1 sends the question to /v1/chat/completions instead (the server writes the template itself, and its own
--reasoning-budget only acts on this path); the thinking comes back as reasoning_content. think=0 (chat path only)
turns thinking off in the template (the fast-reply mode). tail=0 sends the bare question (no answer-format line).
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


url, probs, out, hf = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
MAXNEW, SEED = int(opt.get("maxnew", 6000)), int(opt.get("seed", 0))
if hf == "none":                    # chat=1 only: no tokenizer on this machine; think_tokens and reply_tokens are then whitespace words, gen_tokens stays the server's exact count
    class _Words:
        def __call__(self, text, **kw):
            return type("Enc", (), {"input_ids": text.split()})()
    tok = _Words()
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
TAIL = "" if opt.get("tail") == "0" else "\n\nEnd with a line: Answer: <your answer>"   # tail=0: the bare question
END = "</think>"
if "sample" in opt:
    t, tp, tk = opt["sample"].split(",")
    samp = {"temperature": float(t), "top_p": float(tp), "top_k": int(tk), "seed": SEED}
else:
    samp = {"temperature": 0.0, "top_k": 1}
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
items = [json.loads(l) for l in open(probs)]
todo = [(n, it) for n, it in enumerate(items) if it["id"] not in done]


def ask(job):
    n, it = job
    t0 = time.time()
    if opt.get("chat") == "1":
        body = json.dumps({"messages": [{"role": "user", "content": it["q"] + TAIL}], "max_tokens": MAXNEW,
                           "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": opt.get("think", "1") == "1"}, **samp}).encode()
        res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body,
                                                                      {"Content-Type": "application/json"})))
        msg = res["choices"][0]["message"]
        think, reply = msg.get("reasoning_content") or "", msg.get("content") or ""
        res["stop_type"] = "eos" if res["choices"][0].get("finish_reason") == "stop" else "limit"
    else:
        prompt = tok.apply_chat_template([{"role": "user", "content": it["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        body = json.dumps({"prompt": prompt, "n_predict": MAXNEW, "cache_prompt": False, **samp}).encode()
        res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/completion", body,
                                                                      {"Content-Type": "application/json"})))
        text = res["content"]
        think, _, reply = text.partition(END) if END in text else ("", "", text)
    return n, it, think, reply, res, time.time() - t0


# par=N sends N questions at once (server needs --parallel N); answers are written as they come back, tokens counted here
with open(out, "a") as f, ThreadPoolExecutor(int(opt.get("par", 1))) as pool:
    for n, it, think, reply, res, wall in pool.map(ask, todo):
        tm = res.get("timings", {})
        f.write(json.dumps({**it, "thinking": think.strip(), "reply": reply.strip(),
                            "think_tokens": len(tok(think, add_special_tokens=False).input_ids),
                            "reply_tokens": len(tok(reply, add_special_tokens=False).input_ids),
                            "finish": "stop" if res.get("stop_type") in ("eos", "word") or res.get("stopped_eos") else "length",
                            "seconds": wall, "gen_tokens": tm.get("predicted_n"), "tok_per_s": tm.get("predicted_per_second"),
                            "prompt_ms": tm.get("prompt_ms")}) + "\n")
        f.flush()
        print(f"  {n + 1}/{len(items)} {it['id']}: thinking {len(tok(think).input_ids)}, {wall:.1f} s", flush=True)
print("ENGINE_DONE")
