#!/usr/bin/env python3
"""Reconnect to the Q6A over persistent TCP ADB or paired wireless debugging."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def find_adb() -> str:
    candidates = [
        os.environ.get("ADB_BIN"),
        shutil.which("adb"),
        str(Path.home() / "Library/Android/sdk/platform-tools/adb"),
    ]
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("ADB not found; set ADB_BIN to its path")


def run(*command: str) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def main() -> int:
    adb = find_adb()
    cache = Path(__file__).resolve().parent.parent / ".run" / "q6a-host"
    host = os.environ.get("Q6A_HOST")
    if not host and cache.exists():
        host = cache.read_text().strip()
    if host:
        if not re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host):
            raise RuntimeError("Q6A_HOST must be an IPv4 address")
        address = f"{host}:5555"
        try:
            connection = run(adb, "connect", address).strip()
            state = run(adb, "-s", address, "get-state").strip()
            platform = run(adb, "-s", address, "shell", "getprop", "ro.board.platform").strip()
            if state == "device" and platform == "lahaina":
                cache.parent.mkdir(exist_ok=True)
                cache.write_text(host + "\n")
                print(address)
                return 0
            raise RuntimeError(f"Unexpected device at {address}: {state}, {platform}")
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(f"Persistent ADB unavailable at {address}: {error}", file=sys.stderr)

    services = run(adb, "mdns", "services")
    wanted = os.environ.get("Q6A_MDNS_ID")
    found: list[str] = []
    for line in services.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[1] not in ("_adb._tcp", "_adb-tls-connect._tcp"):
            continue
        if wanted and fields[0] != wanted:
            continue
        address = fields[2]
        if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}", address):
            found.append(address)

    if len(found) != 1:
        raise RuntimeError(
            f"Expected one Q6A ADB service on the LAN; found {len(found)}. "
            "Check power, Wi-Fi, and Q6A_HOST."
        )

    address = found[0]
    connection = run(adb, "connect", address).strip()
    if "connected to " not in connection:
        raise RuntimeError(connection or f"Could not connect to {address}")
    state = run(adb, "-s", address, "get-state").strip()
    if state != "device":
        raise RuntimeError(f"ADB connected to {address} but state is {state!r}")
    platform = run(adb, "-s", address, "shell", "getprop", "ro.board.platform").strip()
    if platform != "lahaina":
        raise RuntimeError(f"ADB device at {address} is not the expected Q6A platform")
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(address.split(":", 1)[0] + "\n")
    print(address)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Q6A connection unavailable: {error}", file=sys.stderr)
        raise SystemExit(2)
