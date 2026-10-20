#!/bin/bash
# Puts the porch (tools/room/porch.py) in ~/.littleguy and starts it as a login service (LaunchAgent), next to the
# brainstem (tools/mind/deploy_mac.sh, run that first: the porch uses its small Python). Then point the menu bar app
# at it: LITTLEGUY_BOARD_URL=http://127.0.0.1:8080 in littleguy.env and rerun tools/menubar/build.sh.
#   DEST (default ~/.littleguy), PORT (default 8080), MACHINE (default lgroom), VOICE (a `say -v` voice, default the
#   Mac's own), ELEVEN_VOICE (his ElevenLabs voice, default Archer - Conversational, Immy's pick; needs the key in
#   ~/.littleguy/elevenlabs.key from a paid account: free accounts can't use library voices through the API)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST=${DEST:-$HOME/.littleguy}; PORT=${PORT:-8080}
LABEL=com.littleguy.porch; PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
mkdir -p "$DEST/code"
cp -f "$HERE/porch.py" "$DEST/code/"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$DEST/venv/bin/python</string><string>$DEST/code/porch.py</string><string>$PORT</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>MACHINE</key><string>${MACHINE:-lgroom}</string><key>VOICE</key><string>${VOICE:-}</string>
    <key>ELEVEN_VOICE</key><string>${ELEVEN_VOICE:-Fahco4VZzobUeiPqni1S}</string>
    <key>ORB</key><string>$(command -v orb)</string>
  </dict>
  <key>WorkingDirectory</key><string>$DEST</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$DEST/porch.log</string>
  <key>StandardErrorPath</key><string>$DEST/porch.log</string>
</dict></plist>
EOF
: > "$DEST/porch.log"
launchctl bootstrap "gui/$(id -u)" "$PLIST"
for i in $(seq 1 15); do curl -sf "http://127.0.0.1:$PORT/api/status" >/dev/null && break; sleep 1; done
tail -2 "$DEST/porch.log"; curl -s "http://127.0.0.1:$PORT/api/status" | head -c 300; echo
