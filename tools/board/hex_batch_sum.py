"""Per-batch NPU profile (GGML_HEXAGON_PROFILE=1, -v): for each batch, which model it belongs to (drafter steps carry
names like inp_xh / h_nextn / out_scaled), the NPU's busy time, the batch's round-trip time, and the host-side gap
before it. Summed per model and per batch shape. usage: python3 hex_batch_sum.py <server log>"""
import re, sys
from collections import defaultdict

def ts(line):   # "0.08.523.875 D" = minutes.seconds.milliseconds.microseconds -> seconds
    m = re.match(r"\s*(\d+)\.(\d+)\.(\d+)\.(\d+)", line)
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6 if m else None

DRAFT = re.compile(r"inp_xh|h_nextn|out_scaled|pre_proj|masked|inp_embd_target")
batches, ops, last_end = [], [], None
for line in open(sys.argv[1], errors="replace"):
    if "profile-op" in line:
        m = re.search(r"profile-op (\S+): (.*?) : ", line)
        ops.append((m.group(1), m.group(2)) if m else ("?", ""))
    elif "profile-batch" in line:
        m = re.search(r"n-ops (\d+) batch-dur-usec (\d+) htp-ops-usec (\d+)", line)
        t = ts(line)
        who = "drafter" if any(DRAFT.search(n) for _, n in ops) else "brain"
        kinds = "+".join(sorted(set(o for o, _ in ops)))[:60]
        batches.append(dict(who=who, n=int(m.group(1)), dur=int(m.group(2)), htp=int(m.group(3)), t=t, kinds=kinds))
        ops = []
agg = defaultdict(lambda: [0, 0, 0, 0])
for i, b in enumerate(batches):
    gap = (b["t"] - b["dur"] / 1e6 - batches[i - 1]["t"]) * 1e6 if i and b["t"] and batches[i - 1]["t"] else 0
    k = (b["who"], b["n"] if b["n"] < 20 else "many")
    agg[k][0] += 1; agg[k][1] += b["dur"]; agg[k][2] += b["htp"]; agg[k][3] += max(gap, 0)
print(f"{'model':8} {'ops/batch':>9} {'batches':>7} {'round trip ms':>13} {'NPU busy ms':>11} {'host gap before ms':>18}")
for (who, n), (c, d, h, g) in sorted(agg.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
    print(f"{who:8} {str(n):>9} {c:7} {d / 1000:13.1f} {h / 1000:11.1f} {g / 1000:18.1f}")
