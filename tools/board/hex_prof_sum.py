"""Sum GGML_HEXAGON_PROFILE=1 'profile-op' lines (llama-bench -v -r 1) per operation kind, for the LAST graph run only
(the first half is llama-bench's warm-up). Matrix multiplies are split by weight type and shape.
usage: python3 hex_prof_sum.py <log> [<log> ...]   (several logs side by side)"""
import re, sys
from collections import defaultdict

def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = re.search(r"profile-op (\S+): (.*) : usec (\d+)", line)
        if not m:
            continue
        op, rest, us = m.groups()
        f = rest.split(" : ")          # names : dims : types : strides
        dims, types = f[1], f[2]
        key = op
        if op == "MUL_MAT":
            key = f"MUL_MAT {types.split(' x ')[0]} {dims.split(' x ')[0]}"
        rows.append((key, int(us)))
    return rows[len(rows) // 2:]          # warm-up first, measured run second

logs = sys.argv[1:]
sums = []
for p in logs:
    d = defaultdict(lambda: [0, 0])
    for k, us in parse(p):
        d[k][0] += us; d[k][1] += 1
    sums.append(d)
keys = sorted(set().union(*sums), key=lambda k: -max(s[k][0] for s in sums))
print(f"{'operation':44}" + "".join(f"{p.split('/')[-1][:14]:>16}" for p in logs))
for k in keys:
    print(f"{k[:44]:44}" + "".join(f"{s[k][0] / 1000:10.2f} ms x{s[k][1]:<4}" if k in s else f"{'-':>16}" for s in sums))
print(f"{'TOTAL on the NPU':44}" + "".join(f"{sum(v[0] for v in s.values()) / 1000:13.2f} ms" for s in sums))
