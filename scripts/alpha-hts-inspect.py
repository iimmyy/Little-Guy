#!/usr/bin/env python3
"""Read UBTECH Alpha 1S .hts action files without modifying them.

Record layout and time conversion cross-checked against Super169/MyAlphaRobotV2:
https://github.com/Super169/MyAlphaRobotV2/tree/master/MyAlphaRobot
Values describe file contents; they are not validated commands for our robot.
"""

import argparse
import csv
from pathlib import Path
import sys


RECORD_SIZE = 33
HEADER_SIZE = 33
TAIL_SIZE = 33


def decode(path: Path):
    data = path.read_bytes()
    payload_size = len(data) - HEADER_SIZE - TAIL_SIZE
    if payload_size < RECORD_SIZE or payload_size % RECORD_SIZE:
        raise ValueError("size is not a 33-byte header, poses, and 33-byte tail")

    for index in range(payload_size // RECORD_SIZE):
        start = HEADER_SIZE + index * RECORD_SIZE
        pose = data[start : start + RECORD_SIZE]
        if pose[:2] != b"\xfb\xbf" or pose[-1] != 0xED:
            raise ValueError(f"pose {index + 1} has invalid frame markers")
        wait_raw = int.from_bytes(pose[29:31], "big")
        yield (index + 1, pose[28] * 20, 200 + (wait_raw - 8) * 20, *pose[8:24])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Alpha 1S .hts file to inspect")
    args = parser.parse_args()

    try:
        rows = list(decode(args.file))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    writer = csv.writer(sys.stdout)
    writer.writerow(["pose", "move_ms", "wait_ms", *(f"servo_{i}" for i in range(1, 17))])
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
