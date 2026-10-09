#!/bin/bash
# Builds LittleGuyBar.app into ~/Applications (menu bar only, no Dock icon).
set -e
cd "$(dirname "$0")"
APP=~/Applications/LittleGuyBar.app
mkdir -p "$APP/Contents/MacOS"
swiftc -O -parse-as-library -target arm64-apple-macos15 LittleGuyBar.swift -o "$APP/Contents/MacOS/LittleGuyBar"
cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>ca.bymartins.littleguybar</string>
  <key>CFBundleName</key><string>Little Guy</string>
  <key>CFBundleExecutable</key><string>LittleGuyBar</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSUIElement</key><true/>
  <key>NSLocalNetworkUsageDescription</key><string>Little Guy reads his status from the board on your home network.</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
EOF
codesign --force --sign - "$APP"
echo "built $APP"
