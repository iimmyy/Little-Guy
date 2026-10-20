#!/usr/bin/env python3
"""His prompts from practice days, asked again of another brain (through brainstems, his live engine), so that brain's
own lines can be measured or trained on (ftpo.py pairs). The coaching sheet comes out of every prompt first (--coach),
so these are the prompts he sees live. Writes lines like steps.jsonl ({"op": "think", "req", "out", "from"}).

usage: replay.py OUT.jsonl --urls http://127.0.0.1:8781,... --key KEYFILE --days DIR... [--coach coach.txt] [--seed 7]
"""
import argparse, json, os, queue, threading, urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--urls", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--coach")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    key = open(a.key).read().strip()
    coach = "".join(l for l in open(a.coach) if not l.startswith("#")).strip() if a.coach else None
    jobs = queue.Queue()
    for d in a.days:
        for k, l in enumerate(open(os.path.join(d, "steps.jsonl"))):
            s = json.loads(l)
            if s["op"] == "think" and s["out"].get("text"):
                req = dict(s["req"])
                if coach:
                    req["messages"] = [dict(m, content=m["content"].replace("\n\n" + coach, "").replace(coach, ""))
                                       for m in req["messages"]]
                req["seed"] = (int(req.get("seed", 0)) + a.seed) % 2 ** 31
                jobs.put((f"{d}#{k}", req))
    lock, out, done = threading.Lock(), open(a.out, "x"), [0]

    def worker(url):
        while True:
            try:
                src, req = jobs.get_nowait()
            except queue.Empty:
                return
            body = json.dumps(req).encode()
            r = urllib.request.Request(url + "/think", data=body, headers={"X-LG-Key": key,
                                                                             "Content-Type": "application/json"})
            try:
                res = json.loads(urllib.request.urlopen(r, timeout=300).read())
            except OSError as e:
                res = {"error": str(e)}
            with lock:
                out.write(json.dumps({"op": "think", "from": src, "req": req, "out": res}, ensure_ascii=False) + "\n")
                out.flush()
                done[0] += 1
                if done[0] % 200 == 0:
                    print(json.dumps({"done": done[0]}), flush=True)
    ts = [threading.Thread(target=worker, args=(u,)) for u in a.urls.split(",")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print(json.dumps({"done": done[0]}))


if __name__ == "__main__":
    main()
