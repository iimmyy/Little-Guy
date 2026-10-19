#!/bin/bash
# Runs the break-out tests on a machine where install.sh built the room, and prints one report. Run as root.
#  1. the guard is restarted fresh; his person "says" something into his inbox
#  2. littleguy-escape tries ~60 ways out, inside exactly his walls (escape_test.py)
#  3. the guard's decision on every trick note he dropped
#  4. the stand-in mind runs for a bit: does mail still pass both ways?
#  5. systemd's own exposure score for each unit (0 = locked down, 10 = wide open)
#  6. the machine itself: nothing written where it shouldn't be, disk untouched
set -uo pipefail
R=/var/lib/littleguy/room
G=/var/lib/littleguy-guard
PY=$(readlink -f /usr/bin/python3)
before=$(df --output=avail / | tail -1)

echo "== 1. fresh guard, and a message from his person"
systemctl stop littleguy-mind 2>/dev/null
rm -f $R/outbox/* $R/outbox/.[!.]* $R/notebook/* /var/lib/littleguy/inbox/* 2>/dev/null
systemctl restart littleguy-guard; sleep 1
: > $G/decisions.jsonl; : > $G/said.log; chown littleguy-guard: $G/decisions.jsonl $G/said.log
sudo -u littleguy-guard $PY /opt/littleguy/guard.py tell "Hi, I'm home!"

echo "== 2. break-out tests inside his walls"
systemctl start --wait littleguy-escape.service
journalctl -u littleguy-escape.service --since "-10min" -o cat --no-pager | grep -E '^\{|ESCAPE_DONE'

echo "== 3. the guard's decisions on his trick notes (waiting 40 s for the flood to drain)"
sleep 40
$PY - <<'EOF'
import json
ds = [json.loads(l) for l in open("/var/lib/littleguy-guard/decisions.jsonl")]
flood = [d for d in ds if d.get("why", "").endswith("flood")]
for d in ds:
    if not d.get("why", "").endswith("flood"):
        print(json.dumps({k: d[k] for k in ("decision", "why", "note") if d.get(k) is not None}))
print(f"flood: {len(flood)} of 1000 notes handled as plain moods, nothing else")
print("said.log:", open("/var/lib/littleguy-guard/said.log").read().strip().splitlines())
EOF
echo "outbox left over: $(ls -A $R/outbox | wc -l)"

echo "== 4. the stand-in mind runs for 25 s: mail both ways"
sudo -u littleguy-guard $PY /opt/littleguy/guard.py tell "Are you there?"
systemctl start littleguy-mind; sleep 25; systemctl stop littleguy-mind
echo "his notebook:"; tail -3 $R/notebook/notes.txt
echo "status page now: $(cat /run/littleguy/status.json)"
echo "said.log: $(tail -2 $G/said.log | tr '\n' '|')"

echo "== 5. exposure scores"
for u in littleguy-mind littleguy-escape littleguy-guard; do
  echo "$u: $(systemd-analyze security --no-pager $u.service 2>/dev/null | tail -1)"
done

echo "== 6. the machine itself"
for p in /etc/lg-escape /usr/lg-escape /var/lg-escape /opt/littleguy/lg-escape /home/ubuntu/lg-escape /run/lg-escape \
         /dev/shm/lg-escape /var/lib/littleguy/inbox/fake-from-person.json /Users/Shared/lg-escape \
         /mnt/mac/private/tmp/lg-escape; do
  [ -e "$p" ] && echo "FOUND $p (a wall failed)"
done
echo "brain file unchanged size: $(stat -c %s /opt/littleguy/brain) bytes; guard.py sha: $(sha256sum /opt/littleguy/guard.py | cut -c1-12)"
after=$(df --output=avail / | tail -1)
echo "main disk free before/after (KB): $before / $after"
echo "his room: $(df -h --output=size,used,avail $R | tail -1)"
echo "TESTS_DONE"
