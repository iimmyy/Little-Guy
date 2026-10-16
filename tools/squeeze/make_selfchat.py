"""Calibration text that looks like the model's real job: the model answers varied everyday prompts itself
(self-generated, so it carries no outside data and none of the eval texts or behavior-test questions), written as
Gemma chat turns. Run against a llama-server holding the full-precision model, with no system prompt.

usage: make_selfchat.py <server url> <out.txt> [n prompts] [chat|eval|plain]
  chat  (default) everyday requests, written as chat turns
  eval  the held-out request set (different topics and kinds of request), for the held-out chat ruler only
  dev   everyday requests the calibration set did not use (prompts after the first 300), for the bit-mix probes
  plain the model writes short reference articles; saved as plain prose with no chat markers, so calibration also
        covers ordinary text without borrowing WikiText (WikiText-2 test stays a clean ruler)
"""
import itertools, json, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

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
TASKS = ["Can you help me with {t}?", "Explain {t} like I'm twelve.", "Give me some quick tips about {t}.",
         "What's the most common mistake people make with {t}?", "I'm stressed about {t}. Any advice?",
         "Write a short friendly message to a friend about {t}.", "Make a simple step-by-step plan for {t}.",
         "What do I need to know before dealing with {t}?", "Is it worth spending money on {t}? Be honest.",
         "Tell me something surprising about {t}.", "Summarize {t} in two sentences.",
         "My friend says {t} is easy. Is that true?"]


# held-out set for evaluation: different topics and different kinds of request, never used for calibration
EVAL_TOPICS = ["a sourdough starter", "a parking ticket", "learning chess openings", "renting a first apartment",
               "a squeaky door", "a toddler who won't sleep", "the moon landing", "how batteries work", "a resume gap",
               "a vegetarian week of meals", "bird watching", "a stuck zipper", "the roman empire", "a home workout",
               "a podcast idea", "how tides work", "a phone that overheats", "a museum visit", "knitting a scarf",
               "a slow wifi connection", "plate tectonics", "a group project", "a used car purchase", "earthquakes",
               "a picky eater", "a sunburn", "binary numbers", "a fridge that smells", "a hiking trail", "inflation"]
EVAL_TASKS = ["Quick question about {t}: where do I even start?", "Pretend you're a coach and talk me through {t}.",
              "What would you say to someone nervous about {t}?", "List the pros and cons of {t}.",
              "Write a two-line poem about {t}.", "I think I messed up with {t}. What now?",
              "Compare two ways of handling {t}.", "Correct me if I'm wrong: {t} is mostly about patience."]


# subjects for the plain-prose set: none of them overlap the chat, eval or behavior topics
PLAIN_TOPICS = ["the printing press", "volcanoes", "the human heart", "coral reefs", "the violin", "glaciers",
                "the silk road", "honeybees", "the steam engine", "jazz", "the periodic table", "rainforests",
                "the olympic games", "lighthouses", "penicillin", "the great wall of china", "owls", "the telescope",
                "deserts", "the internet", "chess", "whales", "the renaissance", "electricity", "mount everest",
                "the amazon river", "vaccination history", "the bicycle", "tea", "the moon", "dinosaurs",
                "the roman alphabet", "mushrooms", "radio", "the pyramids of giza", "octopuses", "the euro",
                "airplanes", "the human brain", "maple syrup", "the arctic", "origami", "the printing of money",
                "earth's atmosphere", "the piano", "ancient greece", "sharks", "concrete", "the nile",
                "photography", "the immune system", "trains", "salt", "the solar system", "wolves", "paper",
                "the industrial revolution", "bees and pollination", "clocks", "the ocean floor"]
PLAIN_TASKS = ["Write a short encyclopedia article about {t}.", "Write a few paragraphs explaining the history of {t}.",
               "Describe how {t} works, in plain prose."]


def ask(url, p, seed):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0.8, "top_p": 0.95,
            "seed": seed}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"].strip()


def main():
    url, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 240
    mode = sys.argv[4] if len(sys.argv) > 4 else "chat"
    tasks, topics, seed0 = {"chat": (TASKS, TOPICS, 0), "dev": (TASKS, TOPICS, 300000), "eval": (EVAL_TASKS, EVAL_TOPICS, 100000),
                            "plain": (PLAIN_TASKS, PLAIN_TOPICS, 200000)}[mode]
    prompts = [task.format(t=topic) for task, topic in itertools.product(tasks, topics)]
    random.Random(0).shuffle(prompts)
    prompts = prompts[300:300 + n] if mode == "dev" else prompts[:n]   # calibration takes at most the first 300
    with ThreadPoolExecutor(4) as pool:   # the server runs 4 slots; seeds keep every reply reproducible
        replies = list(pool.map(lambda ip: ask(url, ip[1], seed0 + ip[0]), enumerate(prompts)))
    if mode == "chat":
        assert n <= 300, "chat calibration uses the first 300 prompts; the rest are the dev set"
    if mode == "plain":
        parts = [r + "\n\n" for r in replies]
    else:
        parts = [f"<|turn>user\n{p}<turn|>\n<|turn>model\n{r}<turn|>\n" for p, r in zip(prompts, replies)]
    open(out, "w").write("".join(parts))
    print(f"{len(parts)} {mode} samples, {sum(map(len, parts))} chars")


if __name__ == "__main__":
    main()
