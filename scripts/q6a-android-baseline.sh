#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
requested_serial="${1:-${ANDROID_SERIAL:-}}"

find_adb() {
    if [[ -n "${ADB_BIN:-}" ]]; then
        printf '%s\n' "$ADB_BIN"
    elif command -v adb >/dev/null 2>&1; then
        command -v adb
    elif [[ -n "${ANDROID_HOME:-}" && -x "$ANDROID_HOME/platform-tools/adb" ]]; then
        printf '%s\n' "$ANDROID_HOME/platform-tools/adb"
    elif [[ -n "${ANDROID_SDK_ROOT:-}" && -x "$ANDROID_SDK_ROOT/platform-tools/adb" ]]; then
        printf '%s\n' "$ANDROID_SDK_ROOT/platform-tools/adb"
    else
        printf '%s\n' "$HOME/Library/Android/sdk/platform-tools/adb"
    fi
}

adb_bin="$(find_adb)"

if [[ ! -x "$adb_bin" ]]; then
    echo "ADB was not found at: $adb_bin" >&2
    echo "Set ADB_BIN to the full path of adb and try again." >&2
    exit 2
fi

device_lines="$($adb_bin devices | awk 'NR > 1 && $2 == "device" { print $1 }')"
device_count="$(printf '%s\n' "$device_lines" | awk 'NF { count++ } END { print count + 0 }')"

if [[ -n "$requested_serial" ]]; then
    if ! printf '%s\n' "$device_lines" | grep -Fxq "$requested_serial"; then
        echo "ADB device '$requested_serial' is not connected and authorized." >&2
        "$adb_bin" devices -l >&2
        exit 2
    fi
    device_serial="$requested_serial"
elif [[ "$device_count" -eq 1 ]]; then
    device_serial="$(printf '%s\n' "$device_lines" | awk 'NF { print; exit }')"
else
    echo "Expected exactly one authorized ADB device; found $device_count." >&2
    echo "Pass a serial as the first argument when more than one device is connected." >&2
    "$adb_bin" devices -l >&2
    exit 2
fi

safe_serial="$(printf '%s' "$device_serial" | tr -c 'A-Za-z0-9._-' '_')"
report_dir="${REPORT_DIR:-$project_dir/.run/device-reports}"
timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
report_path="$report_dir/q6a-baseline-${safe_serial}-${timestamp}.txt"
mkdir -p "$report_dir"

section() {
    printf '\n## %s\n\n' "$1"
}

shell_capture() {
    local title="$1"
    local command="$2"
    section "$title"
    if ! "$adb_bin" -s "$device_serial" shell "$command" 2>&1; then
        echo "[command unavailable on this image]"
    fi
}

property() {
    local key="$1"
    local value
    value="$($adb_bin -s "$device_serial" shell getprop "$key" 2>/dev/null | tr -d '\r')"
    printf '%-34s %s\n' "$key" "${value:-[unset]}"
}

{
    echo "Little Guy Q6A Android baseline"
    echo "Captured UTC: $timestamp"
    echo "ADB serial: $device_serial"

    section "Host connection"
    "$adb_bin" version
    "$adb_bin" devices -l

    section "Android identity"
    property ro.product.manufacturer
    property ro.product.model
    property ro.product.device
    property ro.product.board
    property ro.hardware
    property ro.soc.manufacturer
    property ro.soc.model
    property ro.build.fingerprint
    property ro.build.version.release
    property ro.build.version.sdk
    property ro.product.cpu.abilist
    property ro.boot.slot_suffix
    property ro.boot.verifiedbootstate
    property ro.boot.flash.locked

    shell_capture "Kernel" "uname -a"
    shell_capture "Memory" "cat /proc/meminfo"
    shell_capture "Mounted storage" "df -h /data /sdcard"
    shell_capture "Block devices and UFS clues" \
        "for node in /sys/class/block/sd* /sys/class/block/mmcblk*; do [ -e \"\$node\" ] || continue; echo \"-- \$node\"; cat \"\$node/device/model\" 2>/dev/null; cat \"\$node/size\" 2>/dev/null; done"
    shell_capture "GPU renderer" \
        "dumpsys SurfaceFlinger | grep -E 'GLES|OpenGL|Vulkan' | head -40"
    shell_capture "OpenCL and Qualcomm runtime libraries" \
        "find /vendor /odm /system -type f \( -iname '*OpenCL*' -o -iname '*Qnn*' \) 2>/dev/null | sort | head -250"
    shell_capture "Thermal service" "dumpsys thermalservice"
    shell_capture "Battery and external power" "dumpsys battery"
    shell_capture "USB service" "dumpsys usb"
    shell_capture "SELinux" "getenforce"
    shell_capture "Installed packages" "pm list packages -f | sort"
} > "$report_path"

report_hash="$(shasum -a 256 "$report_path" | awk '{ print $1 }')"
echo "Saved read-only device report: $report_path"
echo "SHA-256: $report_hash"
echo "Raw reports stay under .run/ and are intentionally excluded from Git."
