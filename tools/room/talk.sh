#!/bin/bash
# Talk to Little Guy in his room from the Mac: puts your message in his inbox (through the guard) and prints his reply
# when the guard passes it on. usage: tools/room/talk.sh "Hey, I'm home!"     (MACHINE: default lgroom)
#   tools/room/talk.sh --notes [n]   his last n notebook notes (default 5)
#   VIA (default terminal): the way in stamped on the message (his ear learns who writes through each way)
set -euo pipefail
M=${MACHINE:-lgroom}
G=/var/lib/littleguy-guard
if [ "${1:-}" = --notes ]; then
  orb -m "$M" -u root tail -n "${2:-5}" /var/lib/littleguy/room/notebook/notes.txt; exit
fi
[ -n "${1:-}" ] || { sed -n 2,4p "$0"; exit 1; }
n0=$(orb -m "$M" -u root bash -c "wc -l < $G/chat.jsonl 2>/dev/null || echo 0")
orb -m "$M" -u root sudo -u littleguy-guard python3 /opt/littleguy/guard.py tell "$1" "${VIA:-terminal}"
for _ in $(seq 60); do
  sleep 2
  n=$(orb -m "$M" -u root bash -c "wc -l < $G/chat.jsonl")
  if [ "$n" -gt "$n0" ]; then
    orb -m "$M" -u root tail -n 1 $G/chat.jsonl | python3 -c 'import json,sys; print(json.load(sys.stdin)["reply"])'
    exit
  fi
done
echo "(no reply in 2 minutes: is the brainstem up? launchctl list | grep littleguy)"
