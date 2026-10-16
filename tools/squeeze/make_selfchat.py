"""Calibration text that looks like the model's real job: the model answers varied everyday prompts itself
(self-generated, so it carries no outside data and none of the eval texts or behavior-test questions), written as
Gemma chat turns. Run against a llama-server holding the full-precision or 8-bit model.

usage: make_selfchat.py <server url> <out.txt> [n prompts]
"""
import itertools, json, random, sys, urllib.request

TOPICS = ["cooking dinner on a budget", "a broken bike chain", "learning guitar chords", "a job interview tomorrow",
          "saving money each month", "a houseplant with yellow leaves", "a long road trip", "sleeping better",
          "writing a cover letter", "a noisy neighbour", "starting to run", "a cat that scratches furniture",
          "fixing a slow laptop", "planning a birthday party", "learning Spanish", "a flat tyre", "making bread",
          "studying for an exam", "moving to a new city", "a leaking tap", "taking better photos with a phone",
          "a friend who stopped texting back", "the stock market basics", "a first camping trip", "mixing a song",
          "how vaccines work", "black holes", "the water cycle", "compound interest", "why the sky is blue",
          "a small vegetable garden", "a messy closet", "a rainy weekend", "public speaking nerves",
          "a python list comprehension", "sorting an array", "an excel formula", "a git merge conflict",
          "a recipe for two", "a dog that pulls on the leash", "choosing a laptop", "an argument with a sibling",
          "the french revolution", "photosynthesis", "a solar panel", "a cold that won't go away", "jet lag",
          "learning to swim as an adult", "a wedding speech", "volunteering", "a tight deadline at work"]
TASKS = ["Can you help me with {t}?", "Explain {t} like I'm twelve.", "Give me three quick tips about {t}.",
         "What's the most common mistake people make with {t}?", "I'm stressed about {t}. Any advice?",
         "Write a short friendly message to a friend about {t}.", "Make a simple step-by-step plan for {t}.",
         "What do I need to know before dealing with {t}?", "Is it worth spending money on {t}? Be honest.",
         "Tell me something surprising about {t}.", "Summarize {t} in two sentences.",
         "My friend says {t} is easy. Is that true?"]


def main():
    url, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 240
    prompts = [task.format(t=topic) for task, topic in itertools.product(TASKS, TOPICS)]
    random.Random(0).shuffle(prompts)
    turns = []
    for i, p in enumerate(prompts[:n]):
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 220, "temperature": 0.8, "top_p": 0.95,
                "seed": i}
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"].strip()
        turns.append(f"<|turn>user\n{p}<turn|>\n<|turn>model\n{reply}<turn|>\n")
        if i % 20 == 0:
            print(i, len(reply), flush=True)
    open(out, "w").write("".join(turns))
    print(f"{len(turns)} exchanges, {sum(map(len, turns))} chars")


if __name__ == "__main__":
    main()
