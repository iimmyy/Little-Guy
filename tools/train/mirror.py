#!/usr/bin/env python3
"""The mirror: which words and phrases Little Guy says far more often than people do (his mannerisms), counted against
real recorded human conversation. No AI judges anything here (Immy, 2026-10-19: "AI cannot grade something on making it
sound not AI... the habits are built into you"): the yardstick is how often people actually say a phrase.

Method (Antislop, Paech et al. 2025, arXiv 2510.15061, section 3.1): for each word, two-word and three-word phrase p,
  ratio(p) = his frequency of p / people's frequency of p
over everything he said out loud or thought in some practice days. A phrase counts as a mannerism only if he says it
in several different moments (not one long ramble), and not when it is an echo: a phrase that is in what he was just
told (the message, the time, his task, his persona) is the topic, not a habit.

  mirror.py human CORPUS.csv OUT.json         people's counts from a transcript set (Switchboard as on Hugging Face:
                                              Aman279/Cleaned_Switchboard_Corpus, one conversation per row)
  mirror.py him HUMAN.json OUT_DIR DAY_DIR...  his profile from practice days (each day's steps.jsonl), written as
                                              OUT_DIR/mirror.json (every scored phrase) and OUT_DIR/mirror.md (the top of
                                              each list, for Immy to read and strike)
  env: MIN_MOMENTS (default 5) different moments a phrase must turn up in; COACH (the coaching sheet) is taken out of
       prompts first, so the sheet's own examples count as echoes only where he really heard them
"""
import collections, csv, json, math, os, re, sys

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def words(text):
    text = text.lower().replace("’", "'").replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re") \
        .replace(" 'm", "'m").replace(" 've", "'ve").replace(" 'll", "'ll").replace(" 'd", "'d")
    return WORD.findall(text)


def grams(ws, n):
    return [" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def spoken(text):
    """what he said or thought, without his faces ([ curious ]), *asides* and TASK lines (screens and his to-do list)"""
    text = re.sub(r"(?m)^\s*TASK:.*$", " ", text)
    return re.sub(r"\[[^\]\n]*\]|\*[^*\n]*\*", " ", text)


def human(path, out):
    csv.field_size_limit(10 ** 9)
    c = {1: collections.Counter(), 2: collections.Counter(), 3: collections.Counter()}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        for u in json.loads(row["turns"])["utterance"]:
            ws = words(re.sub(r"<[^>]*>|\bpause\b", " ", u))        # transcriber marks, not speech
            for n in c:
                c[n].update(grams(ws, n))
    json.dump({"source": os.path.basename(path), "totals": {n: sum(c[n].values()) for n in c},
               "counts": {n: dict(c[n]) for n in c}}, open(out, "w"))
    print(json.dumps({"source": path, "words": sum(c[1].values())}))


def his_moments(day_dirs, coach):
    """(what he said, what he was told) for every word he produced in these days"""
    for d in day_dirs:
        for l in open(os.path.join(d, "steps.jsonl"), encoding="utf-8"):
            x = json.loads(l)
            if x["op"] != "think" or not x["out"].get("text"):
                continue
            # what he heard, not his own earlier words (his replies in the conversation, his notebook lines in the
            # persona: those are his habit feeding itself, not the topic)
            told = "\n".join(m["content"] for m in x["req"]["messages"] if m["role"] != "assistant")
            told = "\n".join(l for l in told.splitlines() if not l.lstrip().startswith("- ["))
            if coach:
                told = told.replace("\n\n" + coach, "")
            yield spoken(x["out"]["text"]), told


def him(human_path, out_dir, day_dirs):
    H = json.load(open(human_path))
    coach_path = os.environ.get("COACH")
    coach = "\n".join(l for l in open(coach_path, encoding="utf-8").read().splitlines()
                      if not l.startswith("#")).strip() if coach_path else ""
    min_moments = int(os.environ.get("MIN_MOMENTS", "5"))
    said = {n: collections.Counter() for n in (1, 2, 3)}       # every time he said it
    moments = {n: collections.Counter() for n in (1, 2, 3)}    # how many different moments
    echoes = {n: collections.Counter() for n in (1, 2, 3)}     # moments where it was in what he heard
    n_moments, lengths, examples = 0, [], collections.defaultdict(list)
    for text, told in his_moments(day_dirs, coach):
        ws, heard = words(text), set()
        tw = words(told)
        if not ws:
            continue
        n_moments += 1
        lengths.append(len(ws))
        for n in said:
            g = grams(ws, n)
            said[n].update(g)
            heard = set(grams(tw, n))
            for p in set(g):
                moments[n][p] += 1
                echoes[n][p] += p in heard
                if len(examples[p]) < 3:
                    examples[p].append(" ".join(text.split())[:160])
    rows = {}
    for n in said:
        tot_him, tot_h = sum(said[n].values()), H["totals"][str(n)]
        hc = H["counts"][str(n)]
        scored = []
        for p, k in said[n].items():
            if moments[n][p] < min_moments:
                continue
            f_him = k / tot_him
            f_h = (hc.get(p, 0) + 0.5) / tot_h                      # half a count where people never said it
            scored.append({"phrase": p, "ratio": round(f_him / f_h, 1), "him": k, "moments": moments[n][p],
                           "people": hc.get(p, 0), "echo": round(echoes[n][p] / moments[n][p], 2),
                           "examples": examples[p]})
        scored.sort(key=lambda r: -r["ratio"])
        rows[n] = scored
    os.makedirs(out_dir, exist_ok=True)
    lengths.sort()
    stats = {"days": len(day_dirs), "moments": n_moments, "words": sum(lengths),
             "median words": lengths[len(lengths) // 2] if lengths else 0, "human source": H["source"],
             "human words": H["totals"]["1"]}
    json.dump({"stats": stats, "phrases": rows}, open(os.path.join(out_dir, "mirror.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "mirror.md"), "w") as f:
        f.write(f"# His mannerisms, counted against people\n\n{json.dumps(stats)}\n\n"
                "ratio = how many times more often he says it than people do. echo = share of his moments where the\n"
                "phrase was already in what he heard (high echo = the topic, not a habit).\n")
        for n, title in ((1, "Words"), (2, "Two-word phrases"), (3, "Three-word phrases")):
            f.write(f"\n## {title}\n\n| ratio | phrase | his count (moments) | people's count | echo | e.g. |\n|---|---|---|---|---|---|\n")
            for r in [r for r in rows[n] if r["echo"] < 0.5][:40]:
                f.write(f"| {r['ratio']} | {r['phrase']} | {r['him']} ({r['moments']}) | {r['people']} | {r['echo']} | "
                        f"{r['examples'][0].replace('|', '/')} |\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    if sys.argv[1] == "human":
        human(sys.argv[2], sys.argv[3])
    else:
        him(sys.argv[2], sys.argv[3], sys.argv[4:])
