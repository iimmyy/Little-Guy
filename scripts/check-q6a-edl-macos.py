#!/usr/bin/env python3

import plistlib
import subprocess
import sys
from collections.abc import Iterator
from typing import Any


QUALCOMM_EDL_VENDOR_ID = 0x05C6
QUALCOMM_EDL_PRODUCT_ID = 0x9008


def descendants(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for child in node.get("IORegistryEntryChildren", []):
            yield from descendants(child)
    elif isinstance(node, list):
        for child in node:
            yield from descendants(child)


def main() -> int:
    if sys.platform != "darwin":
        print("This detector is for macOS. On Linux, use lsusb and look for 05c6:9008.")
        return 2

    try:
        encoded_registry = subprocess.check_output(
            ["/usr/sbin/ioreg", "-a", "-r", "-c", "IOUSBHostDevice"],
            stderr=subprocess.STDOUT,
        )
        registry = plistlib.loads(encoded_registry)
    except (OSError, subprocess.CalledProcessError, plistlib.InvalidFileException) as failure:
        print(f"Could not read the macOS USB registry: {failure}")
        return 2

    matches = [
        device
        for device in descendants(registry)
        if device.get("idVendor") == QUALCOMM_EDL_VENDOR_ID
        and device.get("idProduct") == QUALCOMM_EDL_PRODUCT_ID
    ]

    if not matches:
        print("Q6A EDL device not detected (expected USB ID 05c6:9008).")
        print("Hold EDL, connect 12 V PD power, release EDL, then connect the data cable.")
        return 1

    for device in matches:
        name = device.get("USB Product Name") or device.get("IORegistryEntryName") or "Qualcomm EDL"
        location = device.get("locationID")
        location_text = f" at location 0x{location:08x}" if isinstance(location, int) else ""
        print(f"Detected {name}{location_text} (05c6:9008).")

    print("The USB/EDL connection is working. This check did not flash or modify the board.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
