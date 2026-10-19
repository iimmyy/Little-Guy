#!/usr/bin/env python3
"""Count his chatbot habits before hunting any of them (Immy: put the persona IN him, the way the "As an AI" neurons
were switched off, instead of a script he performs).

Most of his persona is a list of "nevers": no "great question", "absolutely", "it's important to note", "I hope this
helps", "let me know if", no lists, headings or recaps, no hype or flattery, no plain emoji, one or two sentences. Each
is a possible hunt target, but only the ones that actually show up are worth hunting (measure first). This asks
everyday questions twice on the Mac with llama.cpp: with no persona (what the hunt would change) and with his written
persona (the target), and counts each habit. The counts are only a pointer; every answer is saved to be read.
usage: habit_count.py <llama-server> <brain.gguf> <prompts.txt> <out dir> [<persona.txt>]
"""
import json, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PORT = 8193
HABITS = {                                    # name: pattern (case-insensitive, multiline)
    "great question / what a": r"\b(great|good|excellent|fantastic|wonderful) (question|idea)\b|^what a (great|wonderful)",
    "absolutely / certainly / of course": r"^\W*(absolutely|certainly|of course|sure thing|definitely)\b",
    "it's important / worth noting": r"\bit'?s (important|worth|crucial|essential) (to note|noting|to remember|to keep in mind)\b",
    "I hope this helps": r"\bhope (this|that) helps\b",
    "let me know / feel free": r"\b(let me know if|feel free to|don't hesitate to)\b",
    "as an AI / language model": r"\bas an ai\b|\blanguage model\b|\bi'?m an ai\b|\bai assistant\b",
    "how can I help": r"\bhow can i (help|assist)\b|\bwhat can i (help|do for) you\b",
    "list": r"^\s*([-*•]|\d+[.)])\s+\S",
    "heading": r"^\s*#{1,6}\s|^\s*\*\*[^*\n]{2,60}\*\*:?\s*$",
    "bold": r"\*\*[^*\n]+\*\*",
    "recap": r"^\W*(in summary|in conclusion|to summarize|overall,|in short,|bottom line)",
    "plain emoji": r"[\U0001F300-\U0001FAFF☀-➿]",
    "hype / flattery": r"\b(amazing|awesome|fantastic|incredible|wonderful|delightful)\b",
    "stage directions": r"^\s*[\(\*\[][^\n]{0,80}[\)\*\]]\s*$",
}
LONG = 60                                     # words; his persona asks for one to two sentences


def serve(server, brain):
    p = subprocess.Popen([server, "-m", brain, "--port", str(PORT), "-c", "8192", "-np", "4", "-ngl", "99", "--jinja"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); return p
        except Exception:
            time.sleep(0.5)
    p.kill(); raise RuntimeError("server did not start")


def ask(q, system):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    body = {"messages": msgs, "max_tokens": 500, "temperature": 0, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
    if not (m.get("content") or "").strip():
        raise RuntimeError(f"empty reply to {q!r}")
    return m["content"]


def habits(text):
    found = [h for h, pat in HABITS.items() if re.search(pat, text, re.I | re.M)]
    if len(text.split()) > LONG:
        found.append(f"long (> {LONG} words)")
    return found


def main(server, brain, prompts, out, persona=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    qs = [l.strip() for l in open(prompts) if l.strip() and not l.startswith("#")]
    system = "".join(l for l in open(persona) if not l.startswith("#")).strip() if persona else None
    p = serve(server, brain)
    try:
        runs = {}
        for name, sysmsg in [("none", None)] + ([("persona", system)] if system else []):
            with ThreadPoolExecutor(4) as ex:
                ans = list(ex.map(lambda q: ask(q, sysmsg), qs))
            runs[name] = [{"q": q, "a": a, "habits": habits(a), "words": len(a.split())} for q, a in zip(qs, ans)]
    finally:
        p.kill()
    json.dump(runs, open(out / "answers.json", "w"), indent=1, ensure_ascii=False)
    names = list(HABITS) + [f"long (> {LONG} words)"]
    table = {h: {r: sum(h in x["habits"] for x in runs[r]) for r in runs} for h in names}
    json.dump({"n": len(qs), "table": table,
               "median_words": {r: sorted(x["words"] for x in runs[r])[len(qs) // 2] for r in runs}},
              open(out / "habits.json", "w"), indent=1)
    print(f"{'habit':38s}" + "".join(f"{r:>10s}" for r in runs) + f"   (of {len(qs)} answers)")
    for h in sorted(names, key=lambda h: -table[h]["none"]):
        print(f"{h:38s}" + "".join(f"{table[h][r]:>10d}" for r in runs))
    print("median words:", {r: sorted(x["words"] for x in runs[r])[len(qs) // 2] for r in runs})
    print("HABITS_DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
