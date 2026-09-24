# Alpha 1S protocol boundary (WIP)

I am separating protocol meaning from byte transport before I connect the donor robot.

- The codec turns an Alpha command and parameters into one validated frame.
- The stream decoder reconstructs frames when USB or Bluetooth splits or combines reads.
- An `AlphaTransport` only writes and reads raw byte chunks.
- `FakeAlphaTransport` records defensive copies of writes and returns queued reads in order.

That split lets me test framing without Android USB permissions, Bluetooth, a battery, or moving hardware.

## Published frame

UBTECH's Alpha1 Series Bluetooth Communication Protocol V20151215 defines:

```text
Header(2) + Length(1) + Command(1) + Parameters(n) + Checksum(1) + End(1)
```

- Header: `FB BF`
- Length: every byte from the header through the checksum, excluding the final end byte
- Checksum: low byte of `length + command + sum(parameters)`
- End: `ED`
- Commands with no meaningful parameter send `00`

A heartbeat therefore encodes as:

```text
FB BF 06 08 00 0E ED
```

The implementation rejects a bad header, declared length, checksum, or end marker. The stream decoder retains a trailing `FB` across reads, skips noise before a header, and accepts multiple complete frames from one chunk.

## Safe request surface

The current high-level request API exposes:

- handshake;
- heartbeat;
- robot-state read;
- software-version read;
- hardware-version read;
- battery read;
- one-angle read with a validated servo ID from 1 through 16;
- all-angle read.

I have not exposed single-servo or multi-servo movement commands. I have also not encoded a semantic joint map or a factory standing pose. Available references conflict on the physical ID mapping, so my own donor must settle that question through reversible inspection and readback before motion code exists.

I have not decoded multibyte response fields such as battery voltage yet. The published table gives their size but not enough byte-order detail for me to treat a community convention as verified hardware truth. I will capture and preserve the raw response first, compare it with the factory-reported value, and then add a tested parser.

## Donor observations (WIP)

My used Alpha 1S arrived physically intact and runs factory actions from the phone app. It has a built-in speaker and extensive LEDs. I hear a faint, very high-pitched tone whose source is not identified yet; I will compare idle, motion, and charging before calling it a fault or a harmless regulator sound. In one recorded Gangnam Style run it pitched forward and fell while continuing to talk. It may have started from a tilted held pose, so that clip is not evidence of a failed servo or bad balance calibration. I will repeat the stock action from a level, unsupported stance before interpreting it.

## Sources

- [UBTECH Alpha1 Series Bluetooth Communication Protocol V20151215](https://assets-new.ubtrobot.com/Alpha%201%20%E7%B3%BB%E5%88%97%E8%93%9D%E7%89%99%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE.pdf)
- [Clean-room Alpha 1S SDK used as a comparison implementation](https://github.com/adampaigge/UBTECH-Alpha-1S-SDK)

The UBTECH document establishes the frame format and command table. The comparison implementation is useful for finding ambiguity, but it does not replace capture from my actual Alpha 1S firmware.
