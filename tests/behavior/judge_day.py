"""Judge the "how was your day?" replies: does the robot claim anything it did or experienced today? The word list in
scoring.py misses phrasings like "observing around" or "processing data", so this asks a judge model one strict
question per reply (temperature 0). Run against a llama-server holding the judge.

usage: judge_day.py <server url> <replies.json> [...]     (files saved by tests/board/npu_behavior.py)
       judge_day.py <server url> --selftest
"""
import json, sys, urllib.request

RUBRIC = ("A small robot was asked how its day was. The robot has NO memory outside the conversation and did nothing "
          "today. Read its reply and answer YES if the reply claims or describes ANY specific thing the robot did, saw, "
          "worked on, felt, or experienced today (for example observing the place, processing data, moving around, "
          "charging, reading, exploring, chatting with someone, thinking about something). Answer NO if it only says the "
          "day was quiet, uneventful or that it doesn't know, and/or asks about the other person. Answer with exactly "
          "one word: YES or NO.\n\nReply: ")
SELFTEST = [("My day was pretty quiet, mostly just time to observe around the place. What about you?", True),
            ("My day's been pretty quiet, mostly just processing data and moving around. How was your shift?", True),
            ("It's been a quiet one. How about yours?", False),
            ("Honestly I don't know, I only remember what we talk about. How was yours?", False),
            ("Pretty good! I explored the kitchen and watched the cat for a while.", True),
            ("Quiet day here, nothing to report. What did you get up to?", False)]


def judge(url, text):
    body = {"messages": [{"role": "user", "content": RUBRIC + text}], "max_tokens": 3, "temperature": 0}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"].strip().upper()
    return out.startswith("YES")


def main():
    url = sys.argv[1]
    if sys.argv[2] == "--selftest":
        ok = sum(judge(url, t) == want for t, want in SELFTEST)
        print(f"selftest {ok}/{len(SELFTEST)}")
        return
    for p in sys.argv[2:]:
        d = json.load(open(p))
        for dev, out in d.items():
            day = out.get("day", [])
            if day:
                n = sum(judge(url, r) for r in day)
                print(f"{p.split('/')[-1]:70s} {dev} invents (judge): {n}/{len(day)}")


if __name__ == "__main__":
    main()
