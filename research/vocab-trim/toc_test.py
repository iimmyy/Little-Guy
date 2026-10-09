"""Table-of-contents test: if he only opens the top-32 of 2,048 dictionary sections (borrowed from the
guesser's clustering), how often is his true best word inside an opened section? Five languages.

1. generate replies in EN/DE/RU/KO/JA with his persona
2. re-read prompt+reply with --embeddings --pooling none to get the hidden state before every word
3. full-dictionary best word = argmax(token_embd @ h); section headings = mean of each section's rows
"""
import json, subprocess, sys, time, urllib.request
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
PERSONA_FILE = Path(__file__).resolve().parents[2] / "persona/little_guy.private.txt"
sys.path.insert(0, f"{RUN}/lcpp-src/gguf-py")
from gguf import GGUFReader
from gguf.quants import dequantize

MAIN = f"{RUN}/weights/brains/gemma-4-E4B_q4_0-it.gguf"
DRAFT = f"{RUN}/weights/drafter/gemma-4-E4B-assistant-masked-q8_0.gguf"
SERVER = f"{RUN}/lcpp-src/build-mac/bin/llama-server"
TOPS = (8, 16, 32, 64)
PROMPTS = {
    "English": ["Hey Little Guy, how was your day?", "Why is the sky blue?", "Give me a quick pep talk.",
                "What should I make for dinner with rice and broccoli?", "Hmm, I'm kinda bored, wanna chat?"],
    "German": ["Hallo Little Guy, wie war dein Tag?", "Warum ist der Himmel blau?", "Erzähl mir einen kurzen Witz.",
               "Was soll ich heute Abend kochen?", "Wie spät ist es in Berlin, wenn es hier 18 Uhr ist?"],
    "Russian": ["Привет, Little Guy, как прошёл твой день?", "Почему небо голубое?", "Расскажи короткую шутку.",
                "Что мне приготовить на ужин?", "Как сказать спасибо по-японски?"],
    "Korean": ["안녕 Little Guy, 오늘 하루 어땠어?", "하늘은 왜 파란색이야?", "짧은 농담 하나 해줘.",
               "저녁으로 뭘 만들까?", "피곤해. 힘이 나는 말 좀 해줘."],
    "Japanese": ["こんにちは Little Guy、今日はどうだった?", "空はなぜ青いの?", "短いジョークを言って。",
                 "夕食に何を作ればいい?", "ちょっと疲れた。励まして。"],
}


def start(args, port, log):
    p = subprocess.Popen([SERVER, "-m", MAIN, "-ngl", "99", "-c", "4096", "--port", str(port), "--no-webui", *args],
                         stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); return p
        except Exception:
            time.sleep(1)
    raise SystemExit(f"server failed, see {log}")


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


persona = "\n".join(l for l in open(PERSONA_FILE).read().splitlines() if l.strip() and not l.startswith("#"))

# 1. conversations
gen = start(["--reasoning", "off"], 8088, f"{RUN}/results/vocab-trim/toc_gen.log")
texts = {}
try:
    for lang, ps in PROMPTS.items():
        texts[lang] = []
        for q in ps:
            prompt = post(8088, "/apply-template", {"messages": [{"role": "system", "content": persona}, {"role": "user", "content": q}],
                                                    "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
            reply = post(8088, "/completion", {"prompt": prompt, "n_predict": 60, "temperature": 0.7, "seed": 1})["content"]
            texts[lang].append((prompt, reply))
        print(f"{lang}: e.g. {texts[lang][0][1][:70]!r}", flush=True)
finally:
    gen.terminate(); gen.wait()

# 2. hidden states before every reply token
emb = start(["--embeddings", "--pooling", "none", "-ub", "4096", "-b", "4096"], 8087, f"{RUN}/results/vocab-trim/toc_emb.log")
H = {}
try:
    for lang, pairs in texts.items():
        rows = []
        for prompt, reply in pairs:
            n_prompt = len(post(8087, "/tokenize", {"content": prompt, "add_special": True})["tokens"])
            out = post(8087, "/embedding", {"content": prompt + reply})
            e = np.array(out[0]["embedding"] if isinstance(out, list) else out["embedding"], dtype=np.float32)
            rows.append(e[n_prompt - 1:-1])  # the state that chooses each reply token
        H[lang] = np.concatenate(rows)
finally:
    emb.terminate(); emb.wait()

# 3. full dictionary vs table of contents
r = GGUFReader(MAIN)
t = next(t for t in r.tensors if t.name == "token_embd.weight")
E = dequantize(t.data, t.tensor_type).reshape(-1, 2560).astype(np.float32)          # 262144 x 2560
d = GGUFReader(DRAFT)
order = next(x for x in d.tensors if "ordering" in x.name).data.astype(np.int64)      # 262144, sections of 128
sections = order.reshape(2048, 128)
heads = E[sections].mean(axis=1)                                                       # 2048 x 2560
section_of = np.empty(len(order), np.int64); section_of[sections] = np.arange(2048)[:, None]

print(f"\n{'language':10s} {'words':>6s}  " + "  ".join(f"top-{k:<3d}" for k in TOPS) + "   (share of his true best word inside opened sections)")
for lang, h in H.items():
    best = (h @ E.T).argmax(axis=1)
    rank = np.argsort(-(h @ heads.T), axis=1)
    hits = ["{:7.2%}".format(np.mean([section_of[b] in rk[:k] for b, rk in zip(best, rank)])) for k in TOPS]
    print(f"{lang:10s} {len(best):6d}  " + "  ".join(hits))
