import json, os, re
from collections import Counter
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.expanduser("~/.littleguy/hf/lfm3-1.7b/tokenizer.json"))
rows = [json.loads(l) for f in ("screen-plain-sci.jsonl", "screen-plain-math.jsonl") for l in open(f)]
C = Counter(); n = 0
for r in rows:
    ids = tk.encode(r.get("thinking") or "", add_special_tokens=False).ids
    C.update(ids); n += len(ids)
top = C.most_common()
txt = lambda i: tk.decode([i])
isword = lambda s: re.fullmatch(r"\s*[A-Za-z']+\s*", s) is not None
words = [(i, c) for i, c in top if isword(txt(i))]
print("thinking tokens", n, "distinct", len(top))
print("share that are alphabetic English tokens %.1f%%" % (100 * sum(c for _, c in words) / n))
for k in (50, 100, 200, 500, 1000):
    print(f"top {k} English tokens cover {100*sum(c for _, c in words[:k])/n:.1f}% of thinking")
for k in (50, 200):
    print(f"top {k} of all tokens cover {100*sum(c for _, c in top[:k])/n:.1f}%")
print("top 40 English:", [txt(i).strip() for i, _ in words[:40]])
print("top 20 overall:", [txt(i) for i, _ in top[:20]])
