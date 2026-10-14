"""Turn a plain text into a Gemma chat transcript for scoring instruction-tuned models (LG_PPL_SPECIAL=1):
paragraphs become alternating user / model turns in Gemma 4's format.

usage: make_chat_eval.py <in.txt> <out.txt>
"""
import sys
paras = [p.strip() for p in open(sys.argv[1]).read().split("\n\n") if p.strip()]
out = []
for i in range(0, len(paras) - 1, 2):
    out.append(f"<|turn>user\n{paras[i]}<turn|>\n<|turn>model\n{paras[i + 1]}<turn|>\n")
open(sys.argv[2], "w").write("".join(out))
print(f"{len(out)} exchanges")
