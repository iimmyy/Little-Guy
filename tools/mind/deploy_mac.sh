#!/bin/bash
# Puts a frozen copy of Little Guy's brainstem in ~/.littleguy and starts it as a login service (LaunchAgent).
# A background service on macOS may not open files in Documents or on external drives without a privacy prompt, so
# everything it needs is copied out: the brain, its dials, lg-mind with its libraries, the two Python files, and a
# small Python environment. Rerun after changing any of them.
#   BRAIN (default: the baked brain with 160 neurons off), DIALS (dials-core.npz), DEST (default ~/.littleguy),
#   PORT (default 8771), LCPP (llama.cpp tree, for gguf-py and the libraries; default ../../.run/lcpp-src)
#   LORA: what training added (an adapter .gguf from tools/train/choices.py), loaded on top of his brain; he then
#   keeps feeling on the scale his untrained brain measured (calibration.json, kept, not measured again). Leave it out
#   to go back to his untrained brain.
#   MOST_WORDS: the most one answer may be (default 300; raise it with the room's REPLY_WORDS, install.sh)
#   EARS: his ears (the audio tower that came with his brain, an mmproj .gguf); a /think message can then carry a
#   sound. Leave it out and he has none (replies without sound are the same either way).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
MOODS=/Volumes/Music/LittleGuy/weights/experiments/moods
BRAIN=${BRAIN:-$MOODS/e4b-grid4-off160.gguf}; DIALS=${DIALS:-$MOODS/dials-core.npz}
DEST=${DEST:-$HOME/.littleguy}; PORT=${PORT:-8771}
LCPP=${LCPP:-$REPO/.run/lcpp-src}; LIBS=$LCPP/build-mac/bin
LABEL=com.littleguy.brainstem; PLIST=$HOME/Library/LaunchAgents/$LABEL.plist

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
mkdir -p "$DEST/bin" "$DEST/brain" "$DEST/code"; chmod 700 "$DEST"

echo "== lg-mind and its libraries"
"$REPO/engine/lg-mind/build.sh" >/dev/null
cp -f "$REPO/engine/lg-mind/build/llama-lg-mind" "$DEST/bin/"
for lib in $(otool -L "$DEST/bin/llama-lg-mind" "$LIBS"/libggml*.0.dylib "$LIBS"/libllama*.0.dylib | awk '/@rpath/{print $1}' \
             | sed 's|@rpath/||' | sort -u); do
  cp -fL "$LIBS/$lib" "$DEST/bin/$lib"
done
cp -fL "$LIBS"/libggml-*.so "$DEST/bin/" 2>/dev/null || true          # backends loaded at run time, if built that way
for r in $(otool -l "$DEST/bin/llama-lg-mind" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}'); do
  install_name_tool -delete_rpath "$r" "$DEST/bin/llama-lg-mind"
done
install_name_tool -add_rpath @loader_path "$DEST/bin/llama-lg-mind"
for f in "$DEST"/bin/*; do
  for r in $(otool -l "$f" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}' | grep -v loader_path); do
    install_name_tool -delete_rpath "$r" "$f" 2>/dev/null || true
  done
  codesign -f -s - "$f" 2>/dev/null
done

echo "== his brain and dials"
cmp -s "$BRAIN" "$DEST/brain/brain.gguf" 2>/dev/null || cp -f "$BRAIN" "$DEST/brain/brain.gguf"
cp -f "$DIALS" "$DEST/brain/dials.npz"
ENV="<key>LG_MIND</key><string>$DEST/bin/llama-lg-mind</string><key>LG_MOST_WORDS</key><string>${MOST_WORDS:-300}</string>"
if [ -n "${LORA:-}" ]; then
  [ -f "$DEST/calibration.json" ] || { echo "no calibration.json yet: deploy once without LORA first" >&2; exit 1; }
  cp -f "$LORA" "$DEST/brain/adapter.gguf"
  ENV="$ENV<key>LG_LORA</key><string>$DEST/brain/adapter.gguf</string>"
  ENV="$ENV<key>LG_CALIBRATION</key><string>$DEST/calibration.json</string>"
else
  rm -f "$DEST/brain/adapter.gguf"
fi
if [ -n "${EARS:-}" ]; then
  cmp -s "$EARS" "$DEST/brain/ears.gguf" 2>/dev/null || cp -f "$EARS" "$DEST/brain/ears.gguf"
  ENV="$ENV<key>LG_EARS</key><string>$DEST/brain/ears.gguf</string>"
else
  rm -f "$DEST/brain/ears.gguf"
fi
PYTHONPATH="$LCPP/gguf-py" "$REPO/.run/mind-venv/bin/python" - "$DEST/brain/brain.gguf" <<'EOF'
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields["tokenizer.chat_template"]
open(sys.argv[1] + ".jinja", "w").write(bytes(f.parts[f.data[0]]).decode())
EOF

echo "== code and Python"
cp -f "$HERE/brain.py" "$HERE/brainstem.py" "$DEST/code/"
[ -x "$DEST/venv/bin/python" ] || uv venv -q "$DEST/venv" --python 3.12
uv pip install -q --python "$DEST/venv/bin/python" numpy jinja2
[ -f "$DEST/brain.key" ] || { python3 -c "import secrets;print(secrets.token_hex(32))" > "$DEST/brain.key"; chmod 600 "$DEST/brain.key"; }

echo "== login service"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$DEST/venv/bin/python</string><string>$DEST/code/brainstem.py</string>
    <string>$DEST/brain/brain.gguf</string><string>$DEST/brain/dials.npz</string><string>$DEST/brain.key</string>
    <string>$PORT</string>
  </array>
  <key>EnvironmentVariables</key><dict>$ENV</dict>
  <key>WorkingDirectory</key><string>$DEST</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$DEST/brainstem.log</string>
  <key>StandardErrorPath</key><string>$DEST/brainstem.log</string>
</dict></plist>
EOF
: > "$DEST/brainstem.log"
launchctl bootstrap "gui/$(id -u)" "$PLIST"
for i in $(seq 1 90); do grep -q BRAINSTEM_READY "$DEST/brainstem.log" && break; sleep 2; done
tail -2 "$DEST/brainstem.log"
