"""Greedy replies from a running llama-server, saved as JSON, so two engine settings can be compared word for word.

usage: python greedy_same.py <port> <out.json> [n_predict]
       python greedy_same.py --compare a.json b.json
"""
import json, sys, urllib.request

PROMPTS = [
    "Explain how a refrigerator keeps food cold.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main causes of the French Revolution?",
    "Give me a recipe for a simple vegetable soup.",
    "I take Dayvigo to sleep. Is it okay to have grapefruit juice with it?",
    "Write a Python function that checks whether a string is a palindrome.",
    "What is the difference between weather and climate?",
    "My friend is upset with me and won't say why. What should I do?",
]


def ask(port, prompt, n):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_k": 1,
                       "max_tokens": n, "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]


if sys.argv[1] == "--compare":
    a, b = (json.load(open(x)) for x in sys.argv[2:4])
    same = 0
    for p in PROMPTS:
        x, y = a[p], b[p]
        if x == y:
            same += 1; print(f"SAME     ({len(x):>4} chars) {p[:60]}")
        else:
            i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            print(f"DIFFERS at char {i:>4}: {p[:60]}\n   a: ...{x[max(0, i - 60):i + 60]!r}\n   b: ...{y[max(0, i - 60):i + 60]!r}")
    print(f"{same}/{len(PROMPTS)} identical")
else:
    port, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    res = {}
    for p in PROMPTS:
        res[p] = ask(port, p, n)
        json.dump(res, open(out, "w"), indent=1)   # saved after every reply
    print("wrote", out)
