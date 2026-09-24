# Q6A bring-up gate (WIP)

I am treating first connection, system flashing, Android validation, and debloating as separate gates. A working cable is not authorization to write storage.

## Gate 1: detect EDL without writing anything

Radxa's documented sequence is:

1. Hold the Q6A EDL button.
2. Connect the 12 V USB-C PD power adapter.
3. Release the EDL button.
4. Connect a USB Type-A data cable to the board's USB 3.1 OTG Type-A port and the host.

The expected Qualcomm EDL USB identity is `05c6:9008`. On my Mac I can check it without a flashing tool:

```sh
./scripts/check-q6a-edl-macos.py
```

This only reads the macOS USB registry. It cannot erase, provision, or flash the Q6A.

The first real connection enumerated as:

```text
QUSB_BULK_CID:…_SN:… (05c6:9008)
```

That proves the 12 V PD supply, dual-Type-A data cable, OTG port, and EDL sequence work together.

## Gate 2: select the exact Android package

The official Radxa downloads currently list `Q6A-Android15-spi-ufs-boot-20260630-b1.7z` and `provision_ufs31.xml`. I must use the UFS instructions for the installed 256 GB UFS module rather than borrowing an eMMC command.

The native macOS ARM64 build of EDL-NG v1.6.0 is staged locally. Its SHA-256 matches the digest published on its GitHub release:

```text
531f32420698b657e44939653a7c18c34d2196f0807f2e3ca38840c56b8449ce
```

I will record the download URL, filename, size, and SHA-256 before any write. I will inspect the extracted package and preserve the exact console log. I will not reflash the preinstalled BIOS unless the board actually fails the documented boot gate; Radxa says BIOS is already installed and ordinarily does not need reflashing.

Flashing and UFS provisioning are destructive operations. They are deliberately absent from the repository scripts until the exact package and target are visible and confirmed.

## Gate 3: preserve the untouched Android baseline

Once Android boots and ADB is authorized, I will run:

```sh
./scripts/q6a-android-baseline.sh
```

If multiple ADB devices are connected, the script refuses to guess. I must pass the Q6A serial explicitly:

```sh
./scripts/q6a-android-baseline.sh SERIAL_FROM_ADB_DEVICES
```

The report captures build identity, ABI, memory, mounted storage, block-device/UFS clues, renderer information, visible OpenCL/QNN libraries, thermal state, battery/power state, USB state, SELinux, and installed packages. It uses read-only Android commands and stores the raw report under `.run/device-reports`, which Git ignores because reports can contain device-specific identifiers.

## Gate 4: validate before debloating

The Q6A now reconnects after a power-off without HDMI: non-root ADB listens on LAN port 5555, and `LittleGuyService` starts from `BOOT_COMPLETED`. I can use `little-guy-monitor` on the Mac while the robot runs at my desk. The open ADB port is temporary bring-up access and must be secured before taking the robot onto another network.

My 256 GB UFS module has not disappeared. The UFS user LUN reports 240,794,992,640 bytes and the `userdata` block device reports 221,205,311,488 bytes, but the mounted F2FS filesystem currently exposes only about 45 GiB. I have not resized or reformatted it. That operation needs a backup and a separate validation gate; the current space is enough to stage the 2.6 GB Gemma file.

The likely mismatch is filesystem size, not an absent UFS module: `userdata` reaches most of the UFS while `/data` remains image-sized. Android mounts `/data` through device mapper (`dm-47` in this boot), so I must not blindly run an F2FS resize command against the raw `sda11` partition. I need a restorable backup, a verified mapping, and a tested recovery route before changing it.

I do not need to carve Linux space out of UFS just to compare boot times. Radxa documents booting the Q6A from microSD and lists microSD ahead of UFS in boot priority. I can put a Q6A Radxa OS image on a separate microSD card, boot it, measure startup, then remove it to return to Android. This is a proposed comparison, not yet tested on my board; Linux model acceleration would need its own separate validation. I will leave the UFS layout alone until that comparison and the backup plan are settled.

### Keep live thermals visible

EDL mode does not run Android, so it does not expose Android thermal telemetry. I will keep the bare board open with its heatsinks installed while flashing. Before any sustained model inference, I will run the host-side terminal monitor:

```sh
little-guy-monitor
```

It polls Android's thermal service and readable sysfs thermal zones every two seconds. The initial host trip limits are deliberately conservative: 85 C for SoC-class sensors, 50 C for battery sensors, and 60 C for skin sensors. I will first inspect the Q6A's actual sensor names and idle/load behavior, then document any justified per-sensor changes rather than silently widening limits.

`--auto-poweroff` asks Android to power down if a host limit is crossed or Android reports a critical-or-worse thermal status. That is a safety layer, not a guarantee: if ADB disconnects or Android hangs, I must physically remove power.

Before disabling any package I need passing evidence for:

- reboot and recovery path;
- ADB reconnection;
- UFS capacity and writable data partition;
- GPU/OpenCL visibility;
- Gemma model deployment and one real response;
- cold and warm model initialization;
- time to first token and sustained decode speed;
- memory and thermal behavior;
- USB host enumeration;
- camera, microphone, speaker, Wi-Fi, and Bluetooth;
- Little Guy foreground-service survival.

Only after that baseline will I disable consumer packages in reversible batches and rerun the same gates. I will preserve Qualcomm/Radxa vendor libraries, HALs, Binder, ART, keystore, permissions, thermal and power services, and the minimal Android framework the headless runtime still depends on.

## Official references

- [Radxa: Dragon Q6A EDL button](https://docs.radxa.com/en/dragon/q6a/hardware-use/edl)
- [Radxa: Dragon Q6A resource downloads](https://docs.radxa.com/en/dragon/q6a/download)
- [Radxa: install a Q6A system through USB](https://docs.radxa.com/en/dragon/q6a/getting-started/install-system/usb-system)
- [Radxa: Q6A microSD boot and boot priority](https://docs.radxa.com/en/dragon/q6a/getting-started/quickly-start)
