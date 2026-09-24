#!/usr/bin/env python3

"""Continuously display Q6A Android thermals and optionally request shutdown."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


DEFAULT_SOC_LIMIT_C = 85.0
DEFAULT_BATTERY_LIMIT_C = 50.0
DEFAULT_SKIN_LIMIT_C = 60.0
CRITICAL_ANDROID_STATUS = 4
STATUS_NAMES = {
    0: "none",
    1: "light",
    2: "moderate",
    3: "severe",
    4: "critical",
    5: "emergency",
    6: "shutdown",
}


def find_adb() -> str:
    explicit = os.environ.get("ADB_BIN")
    candidates = [
        explicit,
        str(Path(os.environ["ANDROID_HOME"]) / "platform-tools/adb")
        if os.environ.get("ANDROID_HOME")
        else None,
        str(Path(os.environ["ANDROID_SDK_ROOT"]) / "platform-tools/adb")
        if os.environ.get("ANDROID_SDK_ROOT")
        else None,
        shutil.which("adb"),
        str(Path.home() / "Library/Android/sdk/platform-tools/adb"),
    ]
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("ADB not found. Set ADB_BIN to the full path of adb.")


def run(adb: str, serial: str, *arguments: str) -> str:
    completed = subprocess.run(
        [adb, "-s", serial, *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=8,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or "ADB command failed")
    return completed.stdout.replace("\r", "")


def select_serial(adb: str, requested: str | None) -> str:
    completed = subprocess.run(
        [adb, "devices"], check=True, text=True, capture_output=True, timeout=8
    )
    devices = [
        line.split()[0]
        for line in completed.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
        and not line.split()[0].startswith("emulator-")
    ]
    if requested:
        if requested not in devices:
            raise RuntimeError(f"ADB device {requested!r} is not connected and authorized.")
        return requested
    if len(devices) != 1:
        raise RuntimeError(
            f"Expected one authorized ADB device; found {len(devices)}. "
            "Pass its serial as the first argument."
        )
    return devices[0]


def read_zones(adb: str, serial: str) -> list[tuple[str, float]]:
    command = (
        "for zone in /sys/class/thermal/thermal_zone*; do "
        "IFS= read -r name < \"$zone/type\" || continue; "
        "IFS= read -r value < \"$zone/temp\" 2>/dev/null || continue; "
        "printf '%s|%s\\n' \"$name\" \"$value\"; "
        "done"
    )
    output = run(adb, serial, "shell", command)
    zones: list[tuple[str, float]] = []
    for line in output.splitlines():
        if "|" not in line:
            continue
        name, raw_value = line.rsplit("|", 1)
        try:
            value = float(raw_value.strip())
        except ValueError:
            continue
        value /= 1000.0
        if 10.0 <= value <= 120.0:
            zones.append((name.strip() or "unnamed", value))
    return zones


def read_android_statuses(adb: str, serial: str) -> list[tuple[str, float, int]]:
    output = run(adb, serial, "shell", "dumpsys thermalservice")
    pattern = re.compile(
        r"mValue=([-+0-9.]+).*?mName=([^,}]+).*?mStatus=([0-9]+)"
    )
    return [
        (match.group(2).strip(), float(match.group(1)), int(match.group(3)))
        for match in pattern.finditer(output)
    ]


def limit_for(name: str, arguments: argparse.Namespace) -> float:
    lowered = name.lower()
    if "batt" in lowered:
        return arguments.battery_limit
    if "skin" in lowered:
        return arguments.skin_limit
    return arguments.soc_limit


def request_poweroff(adb: str, serial: str, reason: str) -> None:
    print(f"\nEMERGENCY: {reason}", flush=True)
    print("Requesting Android power-off. Physically disconnect power if it stays on.", flush=True)
    result = subprocess.run(
        [adb, "-s", serial, "shell", "reboot", "-p"], check=False, timeout=8
    )
    print(f"Shutdown command exit code: {result.returncode}. Power-off is not verified.", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch Q6A Android thermal sensors over ADB."
    )
    parser.add_argument("serial", nargs="?", help="ADB serial when multiple devices exist")
    parser.add_argument("--interval", type=float, default=2.0, help="poll seconds (default: 2)")
    parser.add_argument("--soc-limit", type=float, default=DEFAULT_SOC_LIMIT_C)
    parser.add_argument("--battery-limit", type=float, default=DEFAULT_BATTERY_LIMIT_C)
    parser.add_argument("--skin-limit", type=float, default=DEFAULT_SKIN_LIMIT_C)
    parser.add_argument(
        "--once", action="store_true", help="read one sample and exit"
    )
    parser.add_argument(
        "--auto-poweroff",
        action="store_true",
        help="request Android power-off when a limit or CRITICAL status is reached",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.interval <= 0:
        print("--interval must be positive", file=sys.stderr)
        return 2

    try:
        adb = find_adb()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as failure:
        print(f"Cannot start thermal monitor: {failure}", file=sys.stderr)
        return 2

    serial = None
    print(f"Q6A THERMAL MONITOR — polling every {arguments.interval:g}s. Stop with Ctrl-C.", flush=True)
    print("Waiting for a physical Android device. No temperature is known until a sample arrives.", flush=True)
    print(
        "Limits: "
        f"SoC {arguments.soc_limit:g} C, "
        f"battery {arguments.battery_limit:g} C, "
        f"skin {arguments.skin_limit:g} C."
    )
    print(
        "Auto power-off is ON."
        if arguments.auto_poweroff
        else "Auto power-off is OFF; add --auto-poweroff for guarded model tests."
    )

    while True:
        try:
            if serial is None:
                serial = select_serial(adb, arguments.serial)
            zones = read_zones(adb, serial)
            statuses = read_android_statuses(adb, serial)
        except (OSError, RuntimeError, subprocess.SubprocessError) as failure:
            serial = None
            print(f"{time.strftime('%H:%M:%S')}  TEMPERATURE UNKNOWN — no live telemetry; automatic shutdown unavailable. {failure}", flush=True)
            if arguments.once:
                return 2
            time.sleep(arguments.interval)
            continue

        print(f"\n{time.strftime('%H:%M:%S')}  {len(zones)} thermal zones", flush=True)
        breach: str | None = None
        for name, value in sorted(zones, key=lambda item: item[1], reverse=True):
            limit = limit_for(name, arguments)
            marker = " !!" if value >= limit else ""
            print(f"  {name:<30} {value:6.1f} C  limit {limit:5.1f}{marker}")
            if value >= limit and breach is None:
                breach = f"{name} reached {value:.1f} C (limit {limit:.1f} C)"

        if not zones and not statuses:
            print("  TEMPERATURE UNKNOWN — no readable sensors; automatic thermal shutdown unavailable.", flush=True)

        for name, value, status in sorted(
            statuses, key=lambda item: (item[2], item[1]), reverse=True
        ):
            severity = STATUS_NAMES.get(status, f"unknown-{status}")
            marker = " !!" if status >= CRITICAL_ANDROID_STATUS else ""
            print(f"  android:{name:<22} {value:6.1f} C  {severity}{marker}")
            if value >= limit_for(name, arguments) and breach is None:
                breach = f"{name} reached {value:.1f} C"

        critical = next(
            (
                f"Android marked {name} status {status} at {value:.1f} C"
                for name, value, status in statuses
                if status >= CRITICAL_ANDROID_STATUS
            ),
            None,
        )
        if arguments.auto_poweroff and (critical or breach):
            request_poweroff(adb, serial, critical or breach or "thermal limit reached")
            return 1

        if arguments.once:
            return 0

        time.sleep(arguments.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nThermal monitor stopped.")
