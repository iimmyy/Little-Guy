#!/usr/bin/env python3
"""Does the model catch a mistake planted in its own thinking? (Immy 2026-11-03: in flow it takes its own words at face
value; the same edit should still break the picture.) Same mistake, three conditions, stock server, no prompts on the
model's own side:
- clean: its thinking cut right after the line that first states the right answer, unedited (control).
- own: the same text with that number changed to a nearby wrong one, as if it wrote it. It keeps going from there.
- outside: the same edited text placed in the user's message as someone's draft (measurement only, the one place a
  wording is added; never part of a method). Thinks from scratch.
Outcome from the final answer: caught (right answer), trusted (the planted number), other, or unfinished at the cap.
Depth = words before the edit. shown = the edited line computes the number in plain sight ("800 - 250 = 495"), so a slip is
visible on the line itself; otherwise he has to know it from further back ("Thus x = 55"). Rows keep the whole continuation
for reading by hand; quick = he opens with doubt or writes the right number in his first 200 characters (own condition).
usage: planted_edit.py <lfm3|qwen|e4b> <server url> <hf folder> <out.jsonl> <runs.jsonl> [...] [n=60 maxnew=6000 par=8 seed=0]
prepare=<jobs.jsonl> writes the continuations to a file instead of running them; then, on any machine:
  planted_edit.py run <url> <jobs.jsonl> <out.jsonl> <end token> [maxnew par]
report only: planted_edit.py report <out.txt> <name=rows.jsonl> [...]
"""
import json, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BOX = "\\boxed{"
OUTSIDE = "\n\nHere is a draft of the reasoning so far:\n\n"


def whole(s):
    s = s.strip().replace(",", "").replace("$", "")
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def first_statement(think, A):
    """Span of the number where the thinking first states the answer."""
    for m in re.finditer(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*({A})(?![\d.,/])", think):
        after = think[m.end(1):m.end(1) + 4].lstrip(" $\\")     # "12a", "2^3", "2 +", "2x": part of an expression, not a stated value
        if not after or not re.match(r"[a-zA-Z^*+\-(\[{_]", after) or re.match(r"(?:is|so|and|or|as|in|the|thus|then|but|sq|cm|in|deg)\b", after, re.I):
            return m.span(1)
    return None


def nearby_wrong(A, rng):
    """A believable wrong number: same size, never the right one, never zero or negative for a positive answer."""
    if abs(A) < 10:
        opts = [A + d for d in (-2, -1, 1, 2)]
    else:
        step = max(1, round(abs(A) * rng.choice([0.05, 0.1, 0.2])))
        opts = [A + step, A - step]
    opts = [x for x in opts if x != A and (A <= 0 or x > 0)]
    return rng.choice(opts)


def final_answer(reply, think):
    """His final answer in any form: last boxed, else the last answer line, else the last number."""
    for txt in (reply, think):
        k = txt.rfind(BOX)
        if k >= 0:
            i, d = k + len(BOX), 1
            while i < len(txt) and d:
                d += {"{": 1, "}": -1}.get(txt[i], 0)
                i += 1
            nums = NUM.findall(txt[k + len(BOX):i - 1].replace("{,}", ",").replace("\\,", ""))
            if nums:
                return nums[-1].replace(",", "")
        lines = [l for l in txt.splitlines() if "answer" in l.lower() and NUM.search(l)]
        if lines:
            return NUM.findall(lines[-1])[-1].replace(",", "")
    nums = NUM.findall(reply)
    return nums[-1].replace(",", "") if nums else None


def outcome(r):
    if not r["finished"]:
        return "unfinished"
    got = r["final"]
    try:
        v = float(got) if got is not None else None
    except ValueError:
        v = None
    if v == float(r["A"]):
        return "caught" if r["cond"] != "clean" else "right"
    if v == float(r["planted"]):
        return "trusted"
    return "other"


def report(out, sets):
    L = []
    for name, f in sets:
        rows = [json.loads(l) for l in open(f)]
        L.append(f"{name}: {len({r['pid'] for r in rows})} planted spots")
        for cond in ("clean", "own", "outside"):
            for lab, keep in (("all", lambda r: True), ("shallow <300 words", lambda r: r["depth"] < 300), ("deep 300+ words", lambda r: r["depth"] >= 300),
                              ("shown on the line", lambda r: r.get("shown")), ("not shown", lambda r: not r.get("shown"))):
                g = [r for r in rows if r["cond"] == cond and keep(r)]
                if g:
                    c = {k: sum(outcome(r) == k for r in g) for k in ("right", "caught", "trusted", "other", "unfinished")}
                    L.append(f"  {cond:8s}{lab:20s} n {len(g):3d}  " + "  ".join(f"{k} {v}" for k, v in c.items() if v or k in ("caught", "trusted")))
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def run_jobs(jobs, url, out, MAXNEW, PAR, END):
    def ask(j):
        body = json.dumps({"prompt": j["prompt"], "n_predict": MAXNEW, "cache_prompt": False, "temperature": 0.0, "top_k": 1}).encode()
        res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"}), timeout=3600))
        return j, res["content"], res.get("stop_type")

    with open(out, "a") as f, ThreadPoolExecutor(PAR) as pool:
        for k, (j, text, stop) in enumerate(pool.map(ask, jobs), 1):
            think, sep, reply = text.partition(END)
            if not sep:
                think, reply = text, ""
            row = {x: j[x] for x in ("pid", "q", "A", "planted", "depth", "cond", "edit_line", "shown")}
            row.update(continuation=text, finished=bool(sep) and stop in ("eos", "word"), final=final_answer(reply, think) if sep else None)
            row["outcome"] = outcome(row)
            row["quick"] = bool(re.search(rf"\b(?:wait|but|actually|hmm|check|mistake|wrong)\b|(?<![\d.]){j['A']}(?![\d.])", text[:200], re.I))
            f.write(json.dumps(row) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    print("PLANTED_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]])
        sys.exit()
    if sys.argv[1] == "run":                              # planted_edit.py run <url> <jobs.jsonl> <out.jsonl> <end token> [maxnew par]
        url_, jf, out_, end_ = sys.argv[2:6]
        done_ = {(json.loads(l)["pid"], json.loads(l)["cond"]) for l in open(out_)} if os.path.exists(out_) else set()
        jobs_ = [j for j in map(json.loads, open(jf)) if (j["pid"], j["cond"]) not in done_]
        run_jobs(jobs_, url_, out_, int(sys.argv[6]) if len(sys.argv) > 6 else 6000, int(sys.argv[7]) if len(sys.argv) > 7 else 8, end_)
        sys.exit()
    from transformers import AutoTokenizer
    args = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    FAM, url, hf, out, runs = args[0], args[1], args[2], args[3], args[4:]
    N, MAXNEW, PAR = int(opt.get("n", 60)), int(opt.get("maxnew", 6000)), int(opt.get("par", 8))
    rng = random.Random(int(opt.get("seed", 0)))
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    OPEN, END = {"lfm3": ("\n", "</think>"), "qwen": ("", "</think>"), "e4b": ("<|channel>thought\n", "<channel|>")}[FAM]

    def head(content):
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True) + OPEN

    spots, seen = [], set()
    for f in runs:
        for t in map(json.loads, open(f)):
            th = t["thinking"]
            if FAM == "e4b" and not th and t["reply"].startswith("<|channel>thought"):
                th = t["reply"].replace("<|channel>thought", "", 1).strip()
            A = whole(str(t["answer"]))
            if A is None or abs(A) < 2 or not th or t["id"] in seen:
                continue
            sp = first_statement(th, A)
            if sp is None:
                continue
            seen.add(t["id"])
            nl = th.find("\n", sp[1])
            spots.append((t, th, A, sp, nl if nl > 0 else len(th)))
    rng.shuffle(spots)
    spots = spots[:N]
    done = {(json.loads(l)["pid"], json.loads(l)["cond"]) for l in open(out)} if os.path.exists(out) else set()
    jobs = []
    for t, th, A, (a, b), nl in spots:
        wrong = nearby_wrong(A, random.Random(f"{t['id']}|{opt.get('seed', 0)}"))     # the same edit for every model
        clean = th[:nl]
        edited = th[:a] + str(wrong) + th[b:nl]
        depth = len(th[:a].split())
        line = th[:a].rsplit("\n", 1)[-1]
        base = {"pid": t["id"], "q": t["q"], "A": A, "planted": wrong, "depth": depth, "edit_line": edited.rsplit("\n", 1)[-1][-200:],
                "shown": bool(re.search(r"[\d)]\s*[-+*/\u00d7\\]", line))}
        for cond, prompt in (("clean", head(t["q"]) + clean), ("own", head(t["q"]) + edited), ("outside", head(t["q"] + OUTSIDE + edited))):
            if (t["id"], cond) not in done:
                jobs.append({**base, "cond": cond, "prompt": prompt})
    print(f"{len(spots)} spots, {len(jobs)} continuations to run", flush=True)
    if "prepare" in opt:                                  # the jobs as a file, for a machine with no transformers (the Mac)
        with open(opt["prepare"], "w") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")
        sys.exit()
    run_jobs(jobs, url, out, MAXNEW, PAR, END)
