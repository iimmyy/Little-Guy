#!/usr/bin/env python3
"""The Mac's side of the yes/no page on bymartins (api.md): send a batch up, bring Immy's answers home.

usage: sync.py status
       sync.py push items-<batch>.jsonl
       sync.py pull OUT_DIR        (saves every pull as labels-<time>.jsonl, and the latest answer per item in labels.jsonl)
key: read from the macOS Keychain each time (service ca.bymartins.yesno-api, your account), never printed or saved
env: YESNO_URL (default https://bymartins.ca/yesno/api), YESNO_KEY (a key file instead, for a test server)
"""
import getpass, json, os, subprocess, sys, time, urllib.error, urllib.request

URL = os.environ.get("YESNO_URL", "https://bymartins.ca/yesno/api")


def secret():
    """the API key, held in memory only: from the macOS Keychain (where the site's key lives), or from a file if
    YESNO_KEY names one (a test server)"""
    if os.environ.get("YESNO_KEY"):
        return open(os.path.expanduser(os.environ["YESNO_KEY"])).read().strip()
    r = subprocess.run(["security", "find-generic-password", "-s", os.environ.get("YESNO_KEYCHAIN", "ca.bymartins.yesno-api"),
                        "-a", os.environ.get("YESNO_ACCOUNT", getpass.getuser()), "-w"], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        raise SystemExit("no yes/no key in the Keychain (service ca.bymartins.yesno-api)")
    return r.stdout.strip()


PAGE_FIELDS = ("id", "batch", "n", "kind", "when", "heard", "reply", "cut")   # where it came from stays on the Mac


def call(method, path, body=None, ctype=None):
    req = urllib.request.Request(URL + path, body, method=method, headers={
        "Authorization": "Bearer " + secret(), "User-Agent": "LittleGuy-YesNo/1.0",
        **({"Content-Type": ctype} if ctype else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if "json" not in r.headers.get("Content-Type", ""):   # a sign-in page or a bot check, not the API
                raise SystemExit(f"{path}: got {r.headers.get('Content-Type')} instead of JSON (is the route behind the sign-in?)")
            return r.read().decode()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{path}: {e.code} {e.read().decode()[:300]}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "status"
    if what == "status":
        print(call("GET", "/status"))
    elif what == "push":
        rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
        body = "".join(json.dumps({k: (bool(r[k]) if k == "cut" else r[k]) for k in PAGE_FIELDS}, ensure_ascii=False) + "\n"
                       for r in rows)
        print(call("POST", "/items", body.encode(), "application/x-ndjson"))
    elif what == "pull":
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        text = call("GET", "/labels")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        open(os.path.join(out, f"labels-{stamp}.jsonl"), "w").write(text)       # every pull kept, as it came
        latest = {}
        for name in sorted(f for f in os.listdir(out) if f.startswith("labels-")):
            for l in open(os.path.join(out, name)):
                if l.strip():
                    a = json.loads(l)
                    if a["id"] not in latest or a["t"] >= latest[a["id"]]["t"]:
                        latest[a["id"]] = a
        with open(os.path.join(out, "labels.jsonl"), "w") as f:
            for a in sorted(latest.values(), key=lambda a: a["id"]):
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        yes = sum(a["label"] == "yes" for a in latest.values())
        print(json.dumps({"judged": len(latest), "yes": yes, "no": len(latest) - yes,
                          "notes": sum(bool(a.get("note")) for a in latest.values()), "saved": f"labels-{stamp}.jsonl"}))
    else:
        raise SystemExit(__doc__)
