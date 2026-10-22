#!/usr/bin/env python3
"""Render Immy's group map (groupmap.yaml) as a readable page, and check it before anything multiplies.

The page has three parts:
  1. the whole map at a glance: groups down the side, chemical systems across the top, one symbol per cell
  2. the checks: groups held up by fewer than two systems (Immy's rule: a feeling is never one chemical), groups
     resting only on theory, groups with no cells at all, and how many cells carry each evidence mark
  3. every group in full: which systems, which way, which receptor, how sure, and the source
The checks fail loudly on a malformed cell (unknown group, system, level or evidence mark).
usage: render_groupmap.py <groupmap.yaml> <out.md>
"""
import sys
from collections import Counter
import yaml

LEVELS = {"up": "▲", "down": "▼", "burst": "↑", "mixed": "±", "none": "0"}
EVIDENCE = ("checked", "known", "theory")

src, out = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(src))
groups, systems, cells = m["groups"], m["systems"], m["cells"]

# an unquoted comma in a flow-style label silently splits it into a stray key
for g, info in groups.items():
    assert set(info) <= {"label", "kind", "parked"}, f"group {g}: stray keys {set(info)} (quote the label?)"
by_group = {g: [] for g in groups}
for s, row in cells.items():
    assert s in systems, f"unknown system {s}"
    for g, (level, receptor, ev, note) in row.items():
        assert g in groups, f"{s}: unknown group {g}"
        assert level in LEVELS, f"{s}.{g}: unknown level {level}"
        assert ev in EVIDENCE, f"{s}.{g}: unknown evidence {ev}"
        by_group[g].append((s, level, receptor, ev, note))

L = ["# His chemistry: the group map", "",
     f"Generated from `{src.split('/')[-1]}` by `render_groupmap.py`; edit the YAML, not this page.", "",
     "Symbols: ▲ up · ▼ down · ↑ burst · ± mixed (depends on receptor, timing or person) · 0 shown not involved · "
     "blank = unknown (never guessed). Every cell pulls both ways (foundations doc I.2).", ""]

cols = list(systems)
L += ["## 1. The whole map", "",
      "| group | " + " | ".join(systems[s]["label"] for s in cols) + " |",
      "|---|" + "---|" * len(cols)]
for g, info in groups.items():
    have = {s: lv for s, lv, *_ in by_group[g]}
    name = info["label"] + (" *(parked)*" if info.get("parked") else "")
    L.append(f"| {name} | " + " | ".join(LEVELS[have[s]] if s in have else "" for s in cols) + " |")

ev_count = Counter(ev for row in by_group.values() for _, _, _, ev, _ in row)
live = {g: r for g, r in by_group.items() if not groups[g].get("parked")}
empty = [g for g, r in live.items() if not r]
thin = [g for g, r in live.items() if len(r) == 1]
theory_only = [g for g, r in live.items() if r and all(ev == "theory" for *_, ev, _ in r)]
L += ["", "## 2. Checks", "",
      f"- cells: {sum(ev_count.values())} ({', '.join(f'{ev_count[e]} {e}' for e in EVIDENCE)})",
      f"- groups with **no cells at all** ({len(empty)}): " + (", ".join(groups[g]["label"] for g in empty) or "none"),
      f"- groups held up by **one system only** ({len(thin)}): "
      + (", ".join(f"{groups[g]['label']} ({systems[live[g][0][0]]['label']})" for g in thin) or "none"),
      f"- groups resting **only on theory** ({len(theory_only)}): "
      + (", ".join(groups[g]["label"] for g in theory_only) or "none"),
      "- parked groups: " + (", ".join(f"{i['label']} ({i['parked']})" for i in groups.values() if i.get("parked"))
                             or "none")]

L += ["", "## 3. Every group in full", ""]
for g, info in groups.items():
    rows = by_group[g]
    L.append(f"### {info['label']}" + (f" *(parked: {info['parked']})*" if info.get("parked") else ""))
    if not rows:
        L += ["", "No system placed yet: unknown.", ""]
        continue
    L += ["", "| system | way | receptor | how sure | source and note |", "|---|---|---|---|---|"]
    for s, level, receptor, ev, note in sorted(rows, key=lambda r: cols.index(r[0])):
        L.append(f"| {systems[s]['name']} | {LEVELS[level]} {level} | {receptor} | {ev} | {note} |")
    L.append("")

open(out, "w").write("\n".join(L) + "\n")
print(f"{sum(ev_count.values())} cells, {len(empty)} empty groups, {len(thin)} one-system groups, "
      f"{len(theory_only)} theory-only groups -> {out}")
