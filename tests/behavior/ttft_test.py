"""Time-to-first-word for a multi-turn Little Guy conversation, three ways.

cold:      prompt cache off, so every turn re-reads the whole history
cached:    llama-server keeps the conversation's KV cache between turns
prewarmed: cached, plus the user's message is fed in chunks "while they talk"
           (n_predict=0 requests), so only the last chunk is left when they stop

Usage: python3 ttft_test.py [port]
"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path

from common import BRAINS, LOGS, SERVER
MAIN = BRAINS / "qat-Q4_0-f16proj.gguf"
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8091

SYSTEM = ("You are Little Guy, a small companion robot who lives on Immanuel's desk. You can see through a camera, "
          "hear through a microphone, and move with sixteen servos, but a safety gate must approve every movement. "
          "You remember earlier conversations from your memory notes. Keep replies short, warm and specific. ") * 6
HISTORY_TURNS = [
    "Morning Little Guy, I slept badly again.",
    "I have a shift at the pharmacy later, then I want to work on your NPU code.",
    "Remind me to buy thermal paste for your board.",
    "What did we figure out about the memory bandwidth yesterday?",
    "Okay, I'm heading out now, see you tonight.",
]
USER_TURN = ("I'm back. Long day. The pharmacy was packed and the fridge alarm kept going off, "
             "and then on the way home I kept thinking about how to make you answer faster. "
             "Anyway, how are you, and did anything happen while I was gone?")


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def ask(messages, cache=True, n_predict=24):
    t0 = time.time()
    r = post("/completion", {"prompt": render(messages), "n_predict": n_predict, "cache_prompt": cache,
                             "temperature": 0, "stream": False})
    t = r.get("timings", {})
    return {"wall": time.time() - t0, "prompt_ms": t.get("prompt_ms", 0), "prompt_n": t.get("prompt_n", 0),
            "first_token_s": (t.get("prompt_ms", 0) + (t.get("predicted_ms", 0) / max(t.get("predicted_n", 1), 1))) / 1000,
            "text": r.get("content", "")}


def render(messages, open_turn=True):
    """Gemma 4 chat format: <|turn>role\n text <turn|>\n, then <|turn>model\n to start a reply."""
    out = []
    for role, text in messages:
        out.append(f"<|turn>{'model' if role == 'assistant' else role}\n{text}<turn|>\n")
    if open_turn:
        out.append("<|turn>model\n")
    return "".join(out)


def render_partial(messages, partial):
    """History plus an unfinished user turn, exactly as it will start once the user stops talking."""
    return render(messages, open_turn=False) + f"<|turn>user\n{partial}"


server = subprocess.Popen([str(SERVER), "-m", str(MAIN), "-ngl", "99", "-c", "8192", "--port", str(port),
                           "--no-webui", "-np", "1"],
                          stdout=open(LOGS / "ttft_server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(240):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)

    convo = [("system", SYSTEM), ("user", HISTORY_TURNS[0])]
    for turn in HISTORY_TURNS[1:]:  # build a realistic history with real replies
        convo.append(("assistant", ask(convo)["text"].strip() or "Okay!"))
        convo.append(("user", turn))
    convo.append(("assistant", ask(convo)["text"].strip() or "See you!"))
    final = convo + [("user", USER_TURN)]

    cold = ask(final, cache=False)
    print(f"cold      : re-read {cold['prompt_n']:4d} tokens, first word after {cold['first_token_s']:.2f} s", flush=True)

    ask(convo + [("user", "x")], n_predict=1)  # warm the cache with the history only
    cached = ask(final)
    print(f"cached    : re-read {cached['prompt_n']:4d} tokens, first word after {cached['first_token_s']:.2f} s", flush=True)

    ask(convo + [("user", "x")], n_predict=1)  # reset to history-only again
    words = USER_TURN.split()
    for i in range(1, 6):  # feed the message in 5 chunks while "they talk"
        partial = " ".join(words[: len(words) * i // 6])
        post("/completion", {"prompt": render_partial(convo, partial),
                             "n_predict": 0, "cache_prompt": True})
    pre = ask(final)
    print(f"prewarmed : re-read {pre['prompt_n']:4d} tokens, first word after {pre['first_token_s']:.2f} s", flush=True)
finally:
    server.terminate()
    server.wait()
