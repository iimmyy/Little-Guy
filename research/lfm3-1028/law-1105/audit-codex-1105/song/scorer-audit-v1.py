#!/usr/bin/env python3
"""Read-only source audit with synthetic probes and isolated scorer outputs."""
import builtins
import collections
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
LAW = ROOT.parent
SCI = LAW.parent / "science"
WORK = ROOT / "song" / "scorer-tests-v1"
FLIP_WORK = ROOT / "flip" / "scorer-tests-v1"
WORK.mkdir(exist_ok=True)
FLIP_WORK.mkdir(exist_ok=True)
SOURCES = {
    "audit_tool": ROOT / "audit_tool.py",
    "keyed_grade": SCI / "grades-v4/keyed_grade.py",
    "base_grader": SCI / "grades-v4/sci_grade2.py",
    "keys": SCI / "grades-v4/keys-v2.json",
    "song_score": LAW / "mix1/f5song_score.py",
    "flip_score": LAW / "flip1/flip_score.py",
    "song_goal": LAW / "mix1/GOAL-f5song-v1.md",
    "flip_goal": LAW / "flip1/GOAL-flip-v1.md",
}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]
def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
def write_jl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
before = {k: sha(p) for k, p in SOURCES.items()}
sys.path.insert(0, str(SCI / "grades-v4"))
from keyed_grade import keyed_grade
audit = runpy.run_path(str(SOURCES["audit_tool"]), run_name="audit_under_test")
redirects = []
def run_script(path, args, output_root):
    original_open = builtins.open
    old_argv = sys.argv[:]
    out = io.StringIO()
    err = io.StringIO()
    def safe_open(file, mode="r", *a, **kw):
        if isinstance(file, (str, bytes, os.PathLike)) and any(c in mode for c in "wax+"):
            target = Path(file).resolve()
            if not any(target.is_relative_to(ROOT / name) for name in ("song", "flip")):
                if target != (LAW / "flip1/flips-to-read-v1.txt").resolve():
                    raise RuntimeError("Unexpected scorer write: " + str(target))
                destination = output_root / "flips-to-read-v1.txt"
                redirects.append({"from": str(target), "to": str(destination)})
                file = destination
        return original_open(file, mode, *a, **kw)
    try:
        sys.argv = [str(path), *map(str, args)]
        builtins.open = safe_open
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            namespace = runpy.run_path(str(path), run_name="__main__")
    finally:
        builtins.open = original_open
        sys.argv = old_argv
    return namespace, out.getvalue(), err.getvalue()

# Independently recompute counts from source attempts and the locked hand map.
stats = {}
attempts_by_audit = {}
for name in ("song", "flip"):
    p = ROOT / name
    manifest = json.loads((p / "manifest-v1.json").read_text())
    mapping = json.loads((p / ".map-v1.json").read_text())
    verdicts = {r["item"]: r for r in jl(p / "verdicts-v1.jsonl")}
    assert sha(p / "verdicts-v1.jsonl") == (p / "verdicts-v1.sha256").read_text().strip()
    assert sha(p / ".map-v1.json") == manifest["map_sha256"]
    assert all(sha(f) == h for f, h in manifest["inputs"].items())
    A = audit["attempts"](name)
    attempts_by_audit[name] = A
    index = {(m["arm"], m["run"], m["id"]): item for item, m in mapping.items()}
    assert len(index) == len(mapping)
    assert len(A) == len({(a["arm"], a["run"], a["id"]) for a in A})
    counts = collections.defaultdict(lambda: collections.Counter())
    discrepancies = collections.defaultdict(collections.Counter)
    flags = collections.defaultdict(collections.Counter)
    grades = {}
    for a in A:
        k = (a["arm"], a["run"], a["id"])
        raw = keyed_grade(a["finish"], a["reply"], a["id"], a["answer"], a["scale"])
        machine = "wrong" if raw == "sign" else raw
        item = index.get(k)
        hand = verdicts[item]["verdict"] if item else "cut"
        assert (item is None) == (a["finish"] != "stop")
        group = a["arm"] if name == "song" else a["run"] + "/" + a["arm"]
        for label, grade in (("machine_raw", raw), ("machine", machine), ("hand", hand)):
            counts[label + "/" + group][grade] += 1
        if machine != hand:
            discrepancies[a["arm"]][machine + "->" + hand] += 1
        if item and hand == "right":
            v = verdicts[item]
            flags[a["arm"]][v["flag"] or "none"] += 1
            if v["flag"] != "bad_reasoning" and "bad" in v["note"].lower() and "reasoning" in v["note"].lower():
                flags[a["arm"]]["additional_bad_reasoning_in_key_issue_note"] += 1
        grades[k] = {"machine": machine, "hand": hand, "machine_raw": raw, "item": item}
    stats[name] = {
        "attempts": len(A), "finished_read": len(mapping),
        "counts": {k: dict(v) for k, v in counts.items()},
        "disagreements": {k: dict(v) for k, v in discrepancies.items()},
        "right_flags": {k: dict(v) for k, v in flags.items()},
        "verdict_sha256": sha(p / "verdicts-v1.jsonl"),
        "map_sha256": sha(p / ".map-v1.json"),
        "source_input_hashes": manifest["inputs"],
    }
    if name == "song":
        ids = sorted({a["id"] for a in A})
        assert len(ids) == 81 and len(A) == 648
        import numpy as np
        rng = np.random.default_rng(1105)
        intervals = {}
        for label in ("machine", "hand"):
            for kind in ("right", "wrong", "cut"):
                x = np.array([sum(grades[("song", u, i)][label] == kind for u in audit["RUNS"]) -
                              sum(grades[("f5", u, i)][label] == kind for u in audit["RUNS"]) for i in ids])
                sampled = [int(x[rng.integers(0, len(ids), len(ids))].sum()) for _ in range(4000)]
                intervals[label + "/" + kind] = {"delta": int(x.sum()), "percentile_2_5": float(np.percentile(sampled, 2.5)), "percentile_97_5": float(np.percentile(sampled, 97.5))}
        stats[name]["intervals_reproduced"] = intervals
        stats[name]["whitespace_words"] = {arm: sum(len((a["thinking"] + " " + a["reply"]).split()) for a in A if a["arm"] == arm) for arm in ("f5", "song")}
        stats[name]["tokens"] = {arm: sum(a["tokens"] for a in A if a["arm"] == arm) for arm in ("f5", "song")}
    else:
        assert len(A) == 316
        assert all(sum(a["run"] == s and a["arm"] == arm for a in A) == n for s, n in (("wrong",78),("right",80)) for arm in ("now","wait"))

# Probe the actual unchanged keyed grader, including repaired-key controls.
probes = []
def probe(name, finish, reply, item_id, answer, scale, expected):
    got = keyed_grade(finish, reply, item_id, answer, scale)
    probes.append({"name": name, "reply": reply, "expected_endpoint": expected, "actual": got, "agrees": got == expected})
probe("inclusive_two_percent_boundary", "stop", "Final answer: 980 m/s.", "probe", "1000", 0, "right")
probe("cut_with_correct_answer_stays_cut", "length", "Final answer: 1000 m/s.", "probe", "1000", 0, "cut")
probe("incidental_number_false_credit", "stop", "Intermediate value 12. Final answer: 19 N.", "probe", "12", 0, "wrong")
probe("unsupported_unit_scale_false_credit", "stop", "Final answer: 100 N.", "probe", "1", 0, "wrong")
probe("no_answer_false_credit", "stop", "The supplied input is 50. I have not calculated an answer.", "probe", "50", 0, "wrong")
probe("real_kcal_to_kJ_conversion_missed", "stop", "Final answer: 113 kJ/mol.", "probe", "27", 0, "right")
probe("exact_repaired_form_missed", "stop", r"Final answer: \boxed{1+\sqrt{2}}.", "scibench-class-6b2db3c107", "5.828427125", 0, "right")
probe("brief_integer_rounding_missed", "stop", "Final answer: 1.414 percent.", "probe", "1", 0, "right")
probe("repaired_activation_ratio", "stop", "Final ratio: 1.00185.", "scibench-thermo-b03357fa26", "0.15", 0, "right")
for number in (2,4,6):
    probe("repaired_xenon_alternative_" + str(number), "stop", f"Final answer: {number} mol.", "jee-chem-cebd780d88", "4", 0, "right")
probe("repaired_frequency_ignores_old_scale", "stop", "Final frequency: 1.59155 Hz.", "scibench-class-6b8e42b185", "6.9", -2, "right")
probe("repaired_activity", "stop", "Final answer: 7410 decays per minute.", "probe-b730f0057a", "1.43e24", 0, "right")
probe("repaired_energy_unit", "stop", "Final answer: 7.1 kJ per mole.", "probe-dd1f6a95dc", "7", 0, "right")
probe("repaired_key_cut_stays_cut", "length", "Final ratio: 1.00185.", "scibench-thermo-b03357fa26", "0.15", 0, "cut")
probe("known_unclear_overlay", "stop", "Final answer: 123.", "probe-825b5ed6dd", "123", 0, "unclear")
probe("opposite_sign_is_triaged", "stop", "Final answer: -2 N.", "probe", "2", 0, "sign")

# All standalone song outputs stay in a fixture directory inside song/.
song_dir = WORK / "actual-song"
song_dir.mkdir(exist_ok=True)
held = jl(LAW / "mix1/sci-held-81.jsonl")
write_jl(song_dir / "sci-held-81.jsonl", held)
for u in audit["RUNS"]:
    write_jl(song_dir / f"f5song-{u}.jsonl", jl(LAW / f"mix1/f5song-{u}.jsonl"))
mapping = json.loads((ROOT / "song/.map-v1.json").read_text())
verdicts = {r["item"]: r["verdict"] for r in jl(ROOT / "song/verdicts-v1.jsonl")}
full_new_corrections = {f"{m['run']}:{m['id']}": verdicts[k] for k,m in mapping.items() if m["arm"] == "song"}
write_json(song_dir / "full-hand-song.json", full_new_corrections)
ns, song_out, song_err = run_script(SOURCES["song_score"], [song_dir, "f5song", song_dir / "full-hand-song.json"], song_dir)
(song_dir / "stdout.txt").write_text(song_out)
assert "f5         right 207" in song_out and "f5song     right 194" in song_out
assert "f5song hand right 186" in song_out and "wrong 85" in song_out
assert "VERDICT: IT HURTS" in song_out
g_globals = ns["G"].__globals__
old_fix = g_globals["fix"]
g_globals["fix"] = {"g:probe-cut": "right"}
cut_override = ns["G"]({"id":"probe-cut","finish":"length","reply":"unfinished","answer":"2","scale":0}, "g")
g_globals["fix"] = old_fix

missing_dir = WORK / "missing-song-problem"
missing_dir.mkdir(exist_ok=True)
write_jl(missing_dir / "sci-held-81.jsonl", held)
removed = held[0]["id"]
for u in audit["RUNS"]:
    write_jl(missing_dir / f"f5song-{u}.jsonl", [r for r in jl(song_dir / f"f5song-{u}.jsonl") if r["id"] != removed])
_, missing_out, _ = run_script(SOURCES["song_score"], [missing_dir, "f5song"], missing_dir)
(missing_dir / "stdout.txt").write_text(missing_out)
assert "problems in both 80 of 81 runs 320" in missing_out

# Flip sign classification and zero-data behavior; redirect its legacy write.
flip_rows = LAW / "flip1/e4b-flip-srv-v1.jsonl"
_, flip_out, _ = run_script(SOURCES["flip_score"], [flip_rows], FLIP_WORK)
(FLIP_WORK / "actual-machine-stdout.txt").write_text(flip_out)
_, flip_hand_out, _ = run_script(SOURCES["flip_score"], [flip_rows, ROOT / "flip/corrections-flip-v1.json"], FLIP_WORK)
(FLIP_WORK / "actual-hand-stdout.txt").write_text(flip_hand_out)
empty = FLIP_WORK / "empty.jsonl"
empty.write_text("")
_, empty_out, _ = run_script(SOURCES["flip_score"], [empty], FLIP_WORK)
(FLIP_WORK / "empty-stdout.txt").write_text(empty_out)
assert "THE SENTENCE HOLDS AT THE FINAL ANSWER" in empty_out
assert "AN EMPTY WORD LEAVES IT ALONE" in empty_out

# A normal EOS answer without the channel separator is incorrectly cut.
raw_flip_rows = jl(flip_rows)
eos_unseparated = []
for row_number, r in enumerate(raw_flip_rows, 1):
    if r["done"] and "<channel|>" not in (r["text"] or ""):
        text_grade = keyed_grade("stop", r["text"], r["id"], r["answer"], r.get("scale"))
        eos_unseparated.append({"source_line":row_number,"id":r["id"],"arm":r["arm"],"set":r["set"],"stop":r["stop"],"tokens":r["tokens"],"source_text_sha256":hashlib.sha256(r["text"].encode()).hexdigest(),"packet_grade":"cut","whole_text_machine_grade":text_grade,"hand_verdict_after_unblinding":"right","hand_note":"Explicit final 67.0 J/K equals 25000/373.15=66.996 J/K. Normal EOS; no channel separator. Format issue, not token-limit cutoff."})
assert len(eos_unseparated) == 1
assert eos_unseparated[0]["id"] == "scibench-atkins-57b974868f"
assert eos_unseparated[0]["arm"] == "now" and eos_unseparated[0]["set"] == "right"
assert eos_unseparated[0]["whole_text_machine_grade"] == "right"
write_json(ROOT / "flip/post-lock-issues-v1.json", {
    "locked_verdicts_edited":False,
    "parser_omissions":eos_unseparated,
    "limit_runs":sum(r["stop"] == "limit" for r in raw_flip_rows),
    "locked_hand_right_set_now":{"right":71,"wrong":3,"cut":6},
    "after_parser_correction_hand_right_set_now":{"right":72,"wrong":3,"cut":5},
    "flip_rates_and_fixed_verdicts_change":False,
})
# Exercise the exact 15%-5%=10-point flip boundary.
bar_rows = []
for group in ("wrong","right"):
    for arm in ("now","wait"):
        flips = (3 if arm == "wait" else 1) if group == "wrong" else 1
        for j in range(20):
            grade = ("right" if j < flips else "wrong") if group == "wrong" else ("wrong" if j < flips else "right")
            bar_rows.append({"id":f"probe-bar-{group}-{j}","set":group,"arm":arm,"done":True,"text":"<channel|>Final answer: "+("10." if grade == "right" else "19."),"answer":"10","scale":0})
bar_path = FLIP_WORK / "exact-threshold.jsonl"
write_jl(bar_path, bar_rows)
_, flip_boundary_out, _ = run_script(SOURCES["flip_score"], [bar_path], FLIP_WORK)
(FLIP_WORK / "exact-threshold-stdout.txt").write_text(flip_boundary_out)
assert "wrong wait flips 3/20 = 15.0%" in flip_boundary_out
assert "WAIT: IN BETWEEN" in flip_boundary_out
flip_boundary = {"wait_W_to_R":0.15,"wait_R_to_W":0.05,"now_W_to_R":0.05,"expected":"AN EMPTY DOUBT WORD REACHES WHAT HE HAD","actual":"IN BETWEEN","reason":"0.05 + 0.10 evaluates to 0.15000000000000002","affects_current_78_problem_denominator":False}

# Isolated lock/integrity probes use only fabricated items, never source runs.
integrity_root = WORK / "integrity-fixture"
d = integrity_root / "song"
d.mkdir(parents=True, exist_ok=True)
A = []
mapping = {}
V = []
for n, (arm, u) in enumerate((a,u) for a in ("f5","song") for u in audit["RUNS"]):
    item = "P001-" + chr(65+n)
    mapping[item] = {"arm":arm,"run":u,"id":"probe-only"}
    V.append({"item":item,"verdict":"right","flag":"","note":"Synthetic probe."})
    A.append({"arm":arm,"run":u,"id":"probe-only","finish":"stop","reply":"Final answer 10.","answer":"10","scale":0,"tokens":1})
write_jl(d / "verdicts-v1.jsonl", V)
(d / "items.txt").write_text("\n".join(mapping)+"\n")
write_json(d / ".map-v1.json", mapping)
source_stub = d / "synthetic-input.txt"
source_stub.write_text("original")
write_json(d / "manifest-v1.json", {"inputs":{str(source_stub):sha(source_stub)},"map_sha256":sha(d/".map-v1.json")})
globals_ = audit["lock"].__globals__
old_root = globals_["ROOT"]
old_song_attempts = globals_["song_attempts"]
globals_["ROOT"] = str(integrity_root)
globals_["song_attempts"] = lambda: A
captured = io.StringIO()
try:
    with contextlib.redirect_stdout(captured):
        audit["lock"]("song")
        source_stub.write_text("changed after lock")
        (d / ".map-v1.json").write_text((d / ".map-v1.json").read_text() + " ")
        audit["score"]("song")
        source_and_map_change_accepted = True
        with (d / "verdicts-v1.jsonl").open("a") as f:
            f.write(" ")
        try:
            audit["score"]("song")
            verdict_change_rejected = False
        except SystemExit:
            verdict_change_rejected = True
finally:
    globals_["ROOT"] = old_root
    globals_["song_attempts"] = old_song_attempts
(d / "probe-stdout.txt").write_text(captured.getvalue())
assert source_and_map_change_accepted and verdict_change_rejected

# Fixed-bar checks are exact, not tuned to results.
thresholds = [
    {"delta_right":r,"delta_wrong":w,"actual":audit["bar"](r,w),"expected":e}
    for r,w,e in [(16,-16,"BOTH MOVE THE RIGHT WAY"),(16,0,"MORE RIGHT"),(0,-16,"FEWER WRONG"),(15,-15,"NOTHING"),(-16,0,"IT HURTS"),(16,16,"IT HURTS"),(-9,19,"IT HURTS")]
]
assert all(t["actual"] == t["expected"] for t in thresholds)
after = {k: sha(p) for k,p in SOURCES.items()}
assert before == after
for name in ("song","flip"):
    p = ROOT / name
    manifest = json.loads((p / "manifest-v1.json").read_text())
    assert all(sha(f) == h for f,h in manifest["inputs"].items())
    assert sha(p / ".map-v1.json") == manifest["map_sha256"]
    assert sha(p / "verdicts-v1.jsonl") == (p / "verdicts-v1.sha256").read_text().strip()
results = {
    "sources_before":before,"sources_after":after,"source_code_keys_goals_unchanged":True,
    "source_rows_maps_locked_verdicts_unchanged":True,"counts":stats,
    "grader_probes":probes,"threshold_probes":thresholds,"flip_exact_boundary":flip_boundary,
    "standalone_song_full_hand_table":{"f5":"machine only: 207 right,53 wrong,64 cut","song":"186 right,85 wrong,53 cut; folds 4 unclear into wrong"},
    "cut_correction_override":{"expected":"cut","actual":cut_override,"actual_saved_corrections_override_any_cut":False},
    "missing_song_problem":{"removed_id":removed,"actual_first_line":missing_out.splitlines()[0],"rejected":False},
    "empty_flip_data":{"rejected":False,"stdout":empty_out},
    "audit_integrity":{"changed_source_and_map_hash_accepted":source_and_map_change_accepted,"changed_verdict_hash_rejected":verdict_change_rejected},
    "eos_without_separator":eos_unseparated,
    "redirected_legacy_flip_writes":redirects,
}
write_json(ROOT / "song/scorer-checks-v1.json", results)
write_json(ROOT / "flip/scorer-checks-v1.json", {"counts":stats["flip"],"source_hashes":before,"source_code_keys_goals_unchanged":True,"source_rows_maps_locked_verdicts_unchanged":True,"empty_data":results["empty_flip_data"],"exact_threshold":flip_boundary,"eos_without_separator":eos_unseparated,"redirects":redirects,"full_checks":"../song/scorer-checks-v1.json"})
print(json.dumps({"probes":len(probes),"intentional_endpoint_disagreements":[p["name"] for p in probes if not p["agrees"]],"fixed_bar_checks":len(thresholds),"source_hash_checks":"passed","cut_correction_override":cut_override,"missing_song_problem":results["missing_song_problem"],"locked_hand_counts":{n:s["counts"] for n,s in stats.items()}},indent=2))
