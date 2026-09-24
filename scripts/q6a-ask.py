#!/usr/bin/env python3
"""Ask the Q6A debug runtime from a terminal, without a display."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a prompt to Little Guy over ADB.")
    parser.add_argument("prompt", help="text for the running robot service")
    parser.add_argument("--timeout", type=float, default=120, help="seconds to wait")
    args = parser.parse_args()
    if not args.prompt.strip() or len(args.prompt) > 500 or args.timeout <= 0:
        parser.error("prompt must contain 1-500 characters and timeout must be positive")

    script_dir = Path(__file__).resolve().parent
    serial = subprocess.check_output(
        [sys.executable, str(script_dir / "q6a-connect.py")], text=True, timeout=25
    ).strip()
    adb = os.environ.get("ADB_BIN") or shutil.which("adb") or str(
        Path.home() / "Library/Android/sdk/platform-tools/adb"
    )
    request_id = uuid.uuid4().hex[:12]
    lines: queue.Queue[str] = queue.Queue()
    logs = subprocess.Popen(
        [adb, "-s", serial, "logcat", "-v", "raw", "LittleGuyDebug:I", "*:S"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    def collect_logs() -> None:
        assert logs.stdout is not None
        for line in logs.stdout:
            lines.put(line)

    threading.Thread(target=collect_logs, daemon=True).start()
    try:
        service = subprocess.run(
            [
                adb, "-s", serial, "shell", "am", "start-foreground-service",
                "-n", "com.littleguy.robot/.LittleGuyService",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if service.returncode or "Error:" in service.stdout:
            print(service.stderr.strip() or service.stdout.strip(), file=sys.stderr)
            return 2
        broadcast = subprocess.run(
            [
                adb, "-s", serial, "shell", "am", "broadcast", "--receiver-foreground",
                "-n", "com.littleguy.robot/.DebugPromptReceiver",
                "--es", "request_id", request_id,
                "--es", "prompt", args.prompt,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if broadcast.returncode:
            print(broadcast.stderr.strip() or broadcast.stdout.strip(), file=sys.stderr)
            return 2

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=min(1, deadline - time.monotonic()))
            except queue.Empty:
                if logs.poll() is not None:
                    print("ADB log stream stopped before the reply.", file=sys.stderr)
                    return 2
                continue
            if f"{request_id} " not in line:
                continue
            message = line.split(f"{request_id} ", 1)[1].strip()
            if message == "START":
                continue
            if message == "DONE":
                return 0
            print(message, flush=True)
            if message.startswith("ERROR"):
                return 2
        print("Timed out waiting for Little Guy; check ADB and the thermal monitor.", file=sys.stderr)
        return 2
    finally:
        logs.terminate()
        try:
            logs.wait(timeout=3)
        except subprocess.TimeoutExpired:
            logs.kill()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Q6A prompt failed: {error}", file=sys.stderr)
        raise SystemExit(2)
