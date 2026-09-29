# Getting the Q6A off Android and onto Linux (WIP)

So the reason I did this at all: Android was eating way too much of the Q6A's 12 GB of RAM, and when I tried loading Gemma 4 E4B, Android's low-memory killer just straight up killed my app (that whole story is in [MODEL_SELECTION.md](MODEL_SELECTION.md)). I wanted to leave my working Android install alone on the UFS module and boot Radxa's Ubuntu off a USB stick instead. I'm writing down everything that happened, including the stuff that went wrong, because this board is pretty niche and I honestly couldn't find most of this anywhere.

Full honesty up front, almost all of this was brand new to me. Before this I had never read a schematic, never used EDL, never touched boot firmware, never looked at a device tree, and never held a soldering iron. I learned it as I went, mostly by asking "okay but what is actually deciding this?" over and over. So this doc has every dead end in it, the wrong guesses (Claude's and mine), and a bloopers section at the end.

## How this board boots (read this first)

Everything below makes a lot more sense with this in your head. The board has three separate places it stores stuff:

- **The SPI flash chip**, 32 MB, soldered to the board. It holds the boot firmware, basically the board's BIOS. It also holds the firmware's settings in an area called `VarStore`. On an old PC the BIOS program lived on one chip and its settings lived in "CMOS" memory kept alive by a coin battery. On this board both live on the SPI chip, and flash memory keeps its data with the power off, so there's no battery and unplugging it never resets anything.
- **The UFS module**, 256 GB, the removable storage card. It's like a fast SSD. Android lived here.
- **The USB stick**, which is where I ran Ubuntu from while figuring all of this out.

And when you power it on, it goes in stages, like a relay race:

```
PBL          burned into the Qualcomm chip at the factory, can't be changed
  │          jumps to the SPI chip (or to EDL mode if the EDL button is held)
XBL          first stage on the SPI chip, reads the board config (CDT), sets up RAM
  │          loads the power manager, TrustZone, hypervisor
UEFI         the actual "BIOS", also on the SPI chip
  │          reads its settings, starts storage drivers, builds the device tree
  │          then looks for \EFI\BOOT\BOOTAA64.EFI on a drive
embloader    Radxa's bootloader on the drive, picks the kernel
  │
Linux        obeys the device tree it was handed
  │
Ubuntu       the blue LED starts blinking (heartbeat) once Linux is fully up
```

Two things that matter a lot later:

- **EDL** is an emergency mode built into the chip itself. Hold the EDL button while power comes on and the chip skips the SPI chip completely and waits for a computer over USB. From there `edl-ng` can read and write the SPI chip and the UFS directly. Because EDL is burned into the chip, nothing I write to the SPI chip can break it, which is why I could experiment with the firmware without being scared of bricking the board.
- **The device tree** is a map of the hardware that the firmware hands to Linux. Every piece of hardware has a `status`, `okay` or `disabled`. Linux can't go look at a socket and decide what's in it. It just does what the map says.

The light codes I figured out along the way: green only means powered but nothing running (or EDL). Solid blue means the firmware is up but Linux hasn't started. Blinking blue means Linux is running.

## Writing Ubuntu to a USB stick

I grabbed `radxa-dragon-q6a_noble_gnome_r2.output_512.img.xz` from the [Radxa Q6A downloads page](https://docs.radxa.com/en/dragon/q6a/download). Quick heads up, there are two versions of this image. The `_512` one is for microSD, USB, eMMC, or NVMe, and the `_4096` one is for UFS. Radxa doesn't post a checksum, so here's the SHA-256 of the one I used: `f5ba7d3b34006759b54e14e28f38ec9caaabf6b9a325f4c2c67e6d9fd4412132`.

I unpacked it (6,003,352,576 bytes), wrote it to a 64 GB USB stick, and checked it with `sudo cmp -n 6003352576 <image> /dev/rdiskN`. The stick ended up with three partitions, 16.8 MB, 1.07 GB for EFI, and 4.9 GB for the root filesystem, which lines up exactly with the partition table inside the image.

## First boot attempt, and it just didn't work

I powered off, pulled the UFS module out so Android stays safe on it, plugged the stick in, ran Ethernet to my router, and turned it on. I didn't have a keyboard or monitor hooked up at first.

Here's what I got:

- Only the green LED. When it boots Android I normally see green and blue.
- The USB stick's light was blinking like crazy the whole time.
- Nothing new with SSH showed up on my network, even after two 10 minute scans.
- I moved the stick from the blue USB 3.1 port to one of the black USB 2.0 ports and nothing changed. The blue one is the OTG port, the one I plug into my Mac for flashing, so I didn't trust it for booting anyway.
- Then I hooked it up to a TV over HDMI and got absolutely nothing. No picture, no logo, nothing.

When you don't even see a firmware logo, the problem is happening before the OS ever gets a chance to load. So the USB stick was fine, and the problem had to be the boot firmware sitting on the board's SPI flash chip.

## Why that happened

The boot firmware lives on its own little 32 MiB SPI chip, separate from whatever storage the OS is on. And Radxa's downloads page actually lists two different setups for it:

- For **Linux** you need the board BIOS, which is `dragon-q6a_flat_build_wp_260120.zip`. They say R1 or newer images need BIOS 20251230 or newer. This is a full UEFI firmware. It can boot from USB, microSD, NVMe, or UFS, and it shows stuff on HDMI.
- For **Android 15**, the page says it "only supports booting with BIOS firmware combined with eMMC module or UFS module". My Android package, `Q6A-Android15-spi-ufs-boot-20260630-b1`, comes with its own `spinor/` folder, and I flashed that when I first put Android on.

So my board was running Android's firmware, which only knows how to boot from the UFS module. I'd pulled the UFS out, so it had nothing to boot.

## Checking that and backing everything up first

I put the board into EDL mode and plugged its blue USB 3.1 port into my Mac. My script `scripts/check-q6a-edl-macos.py` saw it as `05c6:9008` (product `QCS_KODIAK`). I used Radxa's `edl-ng` 1.6.0 for macOS with the firehose loader from the Android package.

First I just read the SPI chip's partition table. This only reads, it doesn't change anything:

```sh
edl-ng --memory=Spinor --loader=prog_firehose_ddr.elf printgpt
```

There was exactly one partition on there, `XBL`, 6 MiB. That's the Android layout. Android's `spinor/rawprogram0.xml` only writes `XBL` and the two partition tables.

Then I backed up the whole chip, which took like a second and a half:

```sh
edl-ng --memory=Spinor --loader=prog_firehose_ddr.elf read-lun spinor-lun0-android-20260927.bin
```

The backup is 33,554,432 bytes (8,192 sectors of 4 KiB) with SHA-256 `feda01176176133cbf5035c2ef3e8c6e9f876331f21ef5cae6be41fee00f22c7`. I compared the `XBL` part of it at offset `0x6000` against the Android package's 3,665,840 byte `xbl.elf` and they're identical byte for byte, so that confirmed it, the board was definitely running Android's firmware. The backup lives in `.run/spi-backup/`, which Git ignores, and I kept a second copy outside the repo too.

## The Linux BIOS package

I downloaded it from `https://dl.radxa.com/dragon/q6a/images/dragon-q6a_flat_build_wp_260120.zip`. No checksum from Radxa again, so mine is `cf23a1742ae5d51c451947d82cb21fd3f1c10bbcd621d611f55520673e3f90ca` (11,434,972 bytes).

Its `prog_firehose_ddr.elf` is the exact same file as the Android one. The big difference is what it writes. Android writes one partition to the SPI chip. This writes 25. There's the actual UEFI firmware (`ImageFv`), TrustZone (`TZ`), the hypervisor (`HYP`), `AOP`, `XBL`, `XBL_CONFIG`, `SMBIOS`, the UEFI settings storage (`VarStore`), and a bunch more. I went through its `rawprogram0.xml` and everything goes to the SPI chip. There are no erase commands and it never touches UFS.

## Why I wasn't worried about bricking it

EDL mode lives in the chip's boot ROM, which is permanent and totally separate from that SPI chip. So no matter what I write to that SPI chip, I can always get back into EDL and flash it again. If I ever want Android back, I flash either the Android `spinor/` files or my full backup, put the UFS module back in, and I'm good. The Android install on the UFS never gets written to at all.

## Flashing the Linux BIOS

```sh
cd dragon-q6a_flat_build_wp_260120/flat_build/spinor/dragon-q6a
edl-ng --memory=Spinor --loader=prog_firehose_ddr.elf rawprogram rawprogram0.xml patch0.xml
```

I ran that from my Mac while the board was still in EDL from the backup. It finished with `'rawprogram' command finished successfully` after patching LUN 0. The full log is in `.run/spi-backup/flash-linux-bios-20260927.log`.

## Booting the USB stick again

UFS still out, stick in a black USB 2.0 port, Ethernet plugged into the router.

This time the LEDs went through what looked like a totally normal boot, and then the blue light came on. It never came on during any of the failed attempts, so right away I knew it was working.

The board got `192.168.0.198` from my router over Ethernet. It didn't show up until I pinged the whole subnet, and `radxa-dragon-q6a.local` didn't resolve, so don't count on finding it by name.

Port 22 opened, but the first few SSH connections got reset right away. Then it disappeared off the network for a bit. I'm pretty sure that's Radxa's first boot routine, where it expands the filesystem to fill the stick and then reboots, but I haven't confirmed that. About 72 seconds after it came back, it answered with `SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13.14`.

Heads up, the next time I tried to SSH in I got the big scary `REMOTE HOST IDENTIFICATION HAS CHANGED` warning. That's because the board made brand new SSH keys during that first boot setup, and my Mac still remembered the old ones from before the reboot. Since this is a freshly flashed board on my own home network, I cleared the old entries with `ssh-keygen -R 192.168.0.198` and trusted the new ED25519 key after checking its fingerprint matched the one in the warning. On any network I don't control, I'd take that warning seriously.

## First look at the system

| | |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS, kernel `6.18.2-3-qcom` |
| RAM | 11 GiB total, **10 GiB free sitting idle**, plus 5.6 GiB of zram swap |
| Storage | the USB stick on a USB 2.0 link (480 Mbit/s), filesystem expanded to 57 GB on first boot |
| NPU access | `/dev/fastrpc-cdsp` is there, it belongs to the `render` group, and my `radxa` user is in that group |
| NPU packages | `fastrpc 1.0.3`, `libcdsprpc1 1.0.3`, `radxa-firmware-qcs6490 0.2.29`, plus udev rules `60-qcom-dmaheap.rules` and `99-fastrpc.rules` |

That 10 GiB free is the whole reason I'm here. Android was taking several gigs of the same 12 just for itself.

One thing that might scare people: the kernel log says `no reserved DMA memory for FASTRPC`. I've seen people online connect that message to NPU error 14001. On this image though, that line is about the audio DSP (`3700000.remoteproc`). The NPU work runs on the compute DSP (`a300000.remoteproc`), and that one was totally fine.

## The board kept "dying" and it turned out it was just going to sleep

About 16 minutes after boot, right in the middle of my NPU tests, the board stopped answering on the network. The blue blinking light stopped and only the green one stayed on. I thought it shut off. I hit the power button, the blue light came back, and when I checked the uptime it had never actually rebooted.

It was asleep. Ubuntu's desktop is set by default to suspend after 900 seconds (15 minutes) of no activity. With no keyboard or mouse plugged in, the board thinks nobody's using it, even while the NPU is working its butt off. `/sys/power/suspend_stats/success` read `1`, and the kernel log had `PM: suspend entry (s2idle)` at about 989 seconds of uptime and then `PM: suspend exit` when I pressed the button. So it had nothing to do with heat or power.

For a robot with no screen this is a must, turn sleep off completely:

```sh
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
```

While I was in there I gave my `radxa` user passwordless sudo (`/etc/sudoers.d/90-radxa-nopasswd`) and added it to the `adm` and `systemd-journal` groups so I can read the logs. I'm only OK with that because the board's on my home network and I log in with an SSH key. I'll lock it down properly before the robot goes anywhere else.

## First NPU test, Llama 3.2 1B on the Hexagon

Before trying to compile anything of my own, I wanted to prove the NPU even works on this setup. Radxa has a prebuilt Llama 3.2 1B that's already compiled for this exact chip, so I used that:

```sh
pip3 install --user --break-system-packages modelscope
~/.local/bin/modelscope download --model radxa/Llama3.2-1B-4096-qairt-v68 --local_dir ~/npu/llama32-1b-4096
cd ~/npu/llama32-1b-4096 && chmod +x genie-t2t-run
export LD_LIBRARY_PATH="$PWD" ADSP_LIBRARY_PATH="$PWD"
./genie-t2t-run -c htp-model-config-llama32-1b-gqa.json --prompt_file prompt.txt --profile prof.json
```

A few things I noticed in the bundle:

- It's Genie `1.13.0`, built from QAIRT `2.40.1`.
- It comes with the v68 libraries it needs, and the config is set to `soc_id 35` and `dsp_arch v68`, which is this chip.
- The whole model is one 1,780,002,816 byte file called `weight_sharing_model_1_of_1.serialized.bin` (SHA-256 `d4a284cb9a40d1178daee0c8da2f6a6144d5835b32f5f05e659c3de918aff586`). That "weight sharing" part is a big deal for me. I'd read that you needed a newer chip for that, and here it is running on v68.
- The RoPE scaling factor is set to `32.0`, which is correct for Llama 3. Someone reported that Qualcomm's own export tool puts `8.0` there for this chip and the model spits out garbage ([that issue is here](https://github.com/qualcomm/ai-hub-models/issues/337)). Radxa's version doesn't have that problem.
- `genie-t2t-run` can take embeddings directly with `-e`. That's exactly the door I'll need later to feed in stuff from Gemma's vision and audio encoders.

I didn't need the `cdsprpcd` daemon, and I didn't have to copy any libraries into `/usr/lib/dsp` like some guides say. Pointing `ADSP_LIBRARY_PATH` at the bundle folder was enough.

I asked it, "In one sentence, what would a small companion robot say when carried into a coffee shop?" and it said, "Good morning, I'm here to help you with your coffee, I hope you don't mind if I take a seat next to you?"

Here are the numbers from Genie's built in profiler. No fan, open board, and the model sitting on the slow USB 2.0 stick:

| Run | Prompt tokens | Load | Prompt reading | First word | Tokens written | Writing speed |
| --- | --- | --- | --- | --- | --- | --- |
| short, 1st | 29 | 1.11 s | 94.9 tok/s | 0.306 s | 30 | 8.0 tok/s |
| short, 2nd | 29 | 1.14 s | 95.5 tok/s | 0.304 s | 30 | 2.5 tok/s (weird one) |
| Radxa's long `lprompt_1024.txt` | 823 | 1.11 s | **403.9 tok/s** | 2.04 s | 512 | **9.1 tok/s** |
| short, 4 more runs after turning sleep off | 29 | 1.11 to 1.14 s | 96 to 100 tok/s | 0.29 to 0.30 s | 30 | **9.7 to 10.2 tok/s** |

It only used about 2.6 GB of RAM at its peak. That 2.5 tok/s run never happened again once I turned sleep off, and I still don't know for sure what caused it. The NPU went from about 35 C to 54 C over those four short runs and was at 61 C right after the long one, all with no fan. And even loading off that slow USB stick only took about a second, so storage speed doesn't really matter once the model's running.

To put that next to what I had before: my Gemma 4 E2B on the Android GPU read prompts at 34 to 74 tok/s and wrote at about 6 to 7 tok/s. That's a bigger model, about twice the size, so it isn't a perfectly fair fight. But the NPU is reading prompts way faster, like 5 to 10 times faster, and the writing speed is right around what I'd expect from the board's roughly 22 GB/s of memory bandwidth for a model this size.

## Moving everything onto the UFS

The USB stick was slow, it's on a USB 2.0 link, so once I knew the NPU worked I wanted Linux on the UFS module. That meant wiping Android off the UFS, which I was fine with. I still have the Android files and my SPI backup if I ever want it back.

My first idea was to just write the image to the UFS from the Ubuntu that was already running off the USB stick. That didn't work. I plugged the UFS in and rebooted, and Linux still couldn't see it at all. Turns out in the USB version of the image, the UFS controller is switched off in the device tree (`ufshc@1d84000` has `status = "disabled"`). Funny thing is the UFS PHY right next to it is turned on. And there's no overlay in `/boot/dtbo` to turn it on. So from the USB image, the UFS basically doesn't exist.

Oh, and don't plug the UFS module in while the board is running. It won't show up until a full power cycle.

So I went back to EDL from my Mac, which is how Radxa does it anyway. I downloaded the UFS version of the image. It's hosted on GitHub. I tried `dl.radxa.com` first and got a 404.

```sh
curl -fLO https://github.com/radxa-build/radxa-dragon-q6a/releases/download/rsdk-r2/radxa-dragon-q6a_noble_gnome_r2.output_4096.img.xz
xz -dk radxa-dragon-q6a_noble_gnome_r2.output_4096.img.xz
```

The `.xz` was 1,578,528,436 bytes with SHA-256 `c83ce7f610104f6c0185f88bf10c405acb995f46e95a9c94419065db0cf964cb`, and unpacked it's 10 GiB. Its partition table uses 4 KiB sectors, which is what the UFS uses. That's the whole reason there's a separate `_4096` image.

I pulled the USB stick out, put the board in EDL, and first just read what was on the UFS:

```sh
edl-ng --memory=Ufs --loader=prog_firehose_ddr.elf printgpt
```

Android had split the UFS into eight sections, LUNs 0 to 7. LUN 0 is the big one, 58,787,840 sectors of 4 KiB, so about 240.8 GB. It had all of Android's partitions on it, `super`, `userdata` and the rest. The other LUNs are small boot and test areas, about 14 GB total, and I just left them alone.

Then I wrote the image to the start of LUN 0:

```sh
edl-ng --memory=Ufs --loader=prog_firehose_ddr.elf write-sector 0 radxa-dragon-q6a_noble_gnome_r2.output_4096.img --lun 0
```

It wrote the whole 10 GiB in 34 seconds. I read the partition table back and LUN 0 now has Radxa's layout, a 128 MiB partition, the 1 GiB EFI partition, and a roughly 9 GiB root partition that grows to fill the rest on first boot. Then I rebooted it out of EDL with `edl-ng --memory=Ufs --loader=prog_firehose_ddr.elf reset`. Except it didn't actually boot. It just went right back into EDL and sat there, and I was staring at my network waiting for it to show up. What fixed it was unplugging the USB cable from my Mac and power cycling the board normally, without holding the EDL button. So after flashing, just skip the `reset` command and do that.

## The UFS won't boot yet

So after all that, booting from the UFS by itself doesn't work. The green light and the blue light both come on solid, the Ethernet light blinks, and it just sits there. The blue light never starts blinking like it does when Linux is running, and the board never shows up on my network. I waited over 20 minutes.

If I put the USB stick back in with the UFS still installed, it boots the USB Ubuntu just fine. So the Linux BIOS and the USB stick are both good, and the problem is specifically booting Linux from the UFS.

Here's what I've figured out so far from the USB side:

- The firmware has no boot list. `efibootmgr` says "No BootOrder is set", so it just scans for something bootable, and it picks the USB stick over the UFS when both are in.
- There's no device tree in the Radxa boot entry, on the USB stick or in the UFS image, so Linux gets its hardware description from the firmware.
- When I boot from USB with the UFS module installed, that firmware description has the eMMC controller (`mmc@7c4000`) turned on and the UFS controller (`ufshc@1d84000`) turned off. The Q6A uses the same socket for eMMC and UFS modules, so my guess is the firmware thinks there's an eMMC module in there.
- The UFS image's root partition has its own ID (`6783c260-922f-46a9-ba5a-7c202d0552dd`), different from the USB stick's (`a03a5c05-3365-4811-a1dd-f1776983aa76`), so the two aren't getting mixed up.

My best guess is that when it boots from the UFS, Linux starts, can't see the UFS because the controller's off, and just sits there waiting for its own root drive. That's only a guess though. Radxa's guide makes it sound like you just flash the UFS, put it in, and power on. I need to actually see the screen during that boot to know what's going on.

## Messing around trying to get UFS to show up

Since there was nothing on Linux I cared about losing, I just started experimenting.

**Try 1, give Linux my own device tree.** From the USB system I dumped the firmware's device tree from `/sys/firmware/fdt`, flipped `ufshc@1d84000` to `okay` and `mmc@7c4000` to `disabled` (since it's a UFS module in the socket), stripped out the stuff the firmware fills in fresh every boot in `/chosen`, and compiled it with `dtc`. Then I added a `devicetree` line to the boot entry. It booted totally normally, and the UFS was still disabled. Turns out Radxa's bootloader is their own build of systemd-boot called `embloader`, and `bootctl status` straight up says `✗ Support Type #1 devicetree field`. It just ignores that line.

**Try 2, make the kernel load it.** This kernel has `CONFIG_EFI_ARMSTUB_DTB_LOADER=y`, so I put `dtb=/dtb/q6a-ufs.dtb` in the kernel options instead. That one hung with a solid blue light. So either my device tree was missing something the firmware normally adds, or turning on UFS that way just doesn't work. I never found out which.

**Fixing the stick from my Mac was its own adventure.** I figured I'd just open the USB stick's boot partition on the Mac and delete the `dtb=` part. macOS wouldn't mount it at all. EFI partitions need `sudo` to mount, and then `mount -t msdos` failed because the FAT driver wouldn't load (`Bad code signature` on `msdosfs.kext`). What worked was `mtools`, which reads and writes FAT by itself. I did a `sudo chmod o+rw /dev/rdisk4s2` so my user could get at the partition, then pulled the file out with `mcopy`, edited it, and put it back.

**The firmware settings.** I looked through the firmware's variables in `/sys/firmware/efi/efivars`. There's `DeviceTreeAutoFixup`, `LoadDeviceTree`, and `UfsFixRequired` (set to 0), but nothing that looks like an eMMC or UFS switch.

## The UFS layout, which might be the real problem

Then I found that Armbian had added a UFS setup file for this exact board, [`provision_ufs31_lun0_only.xml`](https://github.com/armbian/qcombin/pull/2). It sets the UFS up as one single LUN that fills the whole module, with `bBootEnable="0"`. My UFS still had Android's layout from when I first installed Android: eight LUNs, `bBootEnable="1"`, and two boot LUNs that still had Android's old bootloader in them. My guess is the firmware sees a UFS set up the Android way and treats it that way.

So I re-provisioned it from the Mac in EDL:

```sh
edl-ng --memory=Ufs --loader=prog_firehose_ddr.elf provision provision_ufs31_lun0_only.xml
```

All three UFS commands came back acknowledged, and it took effect right away. No power cycle needed. Reading the UFS afterwards showed only LUN 0, completely blank. Then I wrote the Radxa UFS image again (34 seconds again) and checked that its partitions were there.

Still solid blue. So the layout wasn't it either.

## Testing blind, one stage at a time

At this point I had no screen, and every test was "boot it and see if the light blinks". So I started designing tests where each possible result looks different from the outside.

**The reboot loop test.** I put `rootdelay=20 panic=10` in the UFS boot entry. Ubuntu's startup script, if it can't find the root drive, will reboot the board after 10 seconds when you give it `panic=10`. So if Linux was actually starting and just couldn't find the UFS, I'd see the board rebooting itself over and over. It didn't. Solid blue, no resets. So it wasn't even getting that far.

**The "borrow the USB stick's Linux" test.** I copied the USB stick's own kernel and startup image onto the UFS boot partition, pointed root at the USB stick, and renamed the stick's `\EFI\BOOT\BOOTAA64.EFI` to `BOOTAA64.BAK` so the firmware couldn't boot the stick directly. That kernel and startup image definitely work, they boot from that stick every day. If the firmware could load anything off the UFS, the board would've come up. It didn't. That told me the kernel was never the problem. It was stuck before Linux.

**Plot twist, PoE.** For a while I couldn't get the board into EDL no matter what I did, and my Mac wouldn't see it at all. Turns out the Ethernet cable was powering the board over PoE the whole time, so unplugging the power cable never actually turned it off. EDL only kicks in if you're holding the button the moment power comes on. And that means some of my earlier "power cycles" probably weren't real cold boots either. Lesson learned, pull power **and** Ethernet.

## Digging through the firmware's hidden settings

Somewhere in the middle of all that, I wanted to know what settings the firmware was actually running with. Linux shows firmware settings in `/sys/firmware/efi/efivars`, and I'd already seen things like `DeviceTreeAutoFixup`, `LoadDeviceTree` and `UfsFixRequired` in there, but nothing that looked like an eMMC or UFS switch.

The catch is that UEFI settings come with attributes. The "runtime" ones show up in Linux. The "boot services only" ones disappear the moment the OS starts, and the kind of thing you'd set in the F2 menu is often that type. So I dumped the whole SPI chip over EDL (read only, 1.5 seconds) and looked at the settings area directly. It starts at sector 138 and it's 640 KB.

It isn't the standard EDK2 format. It starts with a `PTBL` header, which is Qualcomm's own layout. Every record goes like this:

```
data size (4 bytes) | name size (2 bytes) | name in UTF-16 | value at the next 4 byte boundary
```

and every record is stored twice, like two save slots. Pulling every UTF-16 name out of that area showed a bunch of settings Linux never showed me, like `BootDevOrder`, `BootAskTimeoutSecs` and `BootAskValid`. `BootDevOrder` held `12, 9, 6, 2, 11` followed by zeros. Later the firmware's own log explained that list, it's USB, SD, NVMe, eMMC, UFS.

I also tried searching the firmware itself for text about how it picks storage. The UEFI part (`imagefv.elf`) is compressed, so plain text searches found nothing useful. The little FAT disk that gets flashed as the `PLAT` partition (`Fat12.bin`) has an `ACPI` folder and an empty `uefi_cfg.txt`, which looked exciting for about five minutes. And the `config` partition at the start of the USB stick is just Radxa's `rsetup` config file and the boot logo.

None of this found the UFS problem directly. But learning how to read the settings store is what saved me later, when I had to restore it by hand.

## Stepping back and thinking it through

This is where I stopped throwing experiments at it. Honestly, a big chunk of the afternoon had been Claude running test after test on its own while I watched, and I realized I hadn't learned anything from it. So I told it to stop, explain what was actually going on, and let me do the reasoning. That turned out to be the turning point.

We went back to basics and I reasoned through the boot order myself.

The board boots in stages: PBL (burned into the chip), then XBL and the UEFI firmware (both on the SPI chip), then embloader (the bootloader on the drive), then the Linux kernel, then Ubuntu.

Linux can't be the one deciding the UFS is off, because Linux just does what the device tree tells it. The device tree literally says `ufshc@1d84000 { status = "disabled"; }` and the kernel obeys. embloader can't be it either, it only picks which kernel to run and doesn't touch the device tree. PBL just jumps to the SPI chip. So the decision had to be happening in the firmware on the SPI chip, XBL or UEFI.

I compared the text strings inside the Linux BIOS's `xbl.elf` against the Android one, since XBL is the one piece that was on the SPI chip in both setups. They carry the same UFS driver messages, and both have `UfsSmmuConfigForOtherBootDev = 1`. The one real difference is `UEFIExitUfsSSURequired = 1`, which is only in Android's XBL. Watch out when you do this, `strings` grabs random junk bytes sometimes, so `-MmUFS 3.x Device Volt set to 2.5 volt` and `iUFS 3.x Device Volt set to 2.5 volt` look different but they're the same message.

But having the same strings only means the same code is in there. It doesn't mean it runs the same way. What I really needed was to see what the board actually did.

## Reading the firmware's own boot log

Those strings are log messages, and the firmware keeps its log in RAM. When I booted the USB stick (with the UFS still plugged in), the kernel log showed memory the firmware had set aside:

```
xbl@80600000           2 MB
xbl-uefi-res@80880000  64 KB
smem@80900000          2 MB
```

This kernel has `CONFIG_STRICT_DEVMEM` turned off, so `/dev/mem` is readable. A plain `dd` failed with "Bad address" because those areas are hidden from Linux's normal memory map, but mapping them with Python's `mmap` on `/dev/mem` worked. The UEFI boot log was sitting in `xbl-uefi-res`, 388 lines of it. And there it was:

```
UEFI Ver    : 6.0.260120.BOOT.MXF.1.0.1-00549-KODIAKWP-1
Boot Device : NVME
Card Detection FAILED! Handle 0, Slot 0
eMMC module not found
External SD card not found
UFS GPIO7 value is 1
Skipping UFS initialization
...
[QcomBds] Boot Device Order:
[QcomBds]   [0]: USB
[QcomBds]   [1]: SD
[QcomBds]   [2]: NVMe
[QcomBds]   [3]: eMMC
[QcomBds]   [4]: UFS
[QcomBds] Trying device 0: USB
```

So the firmware reads one pin, GPIO7, to decide if there's a UFS module in the socket. On my board it reads `1`, and the firmware goes "no UFS here" and never even starts the UFS. That one pin explains everything. It's why the firmware can't boot from the UFS, and it's why the device tree it gives Linux has UFS disabled.

It also cleared up that `BootDevOrder = 12, 9, 6, 2, 11` value I dug out of the firmware's settings earlier. It lines up with this list, USB, SD, NVMe, eMMC, UFS.

The dumps are saved in `.run/fwdump/`, and the readable log is `.run/fwdump/uefi-log-20260928.txt`.

## Is GPIO7 just a "no UFS" signal?

I went digging on GitHub and found [radxa-build/radxa-dragon-q6a#4](https://github.com/radxa-build/radxa-dragon-q6a/issues/4). That issue is actually about a boot loop after changing camera overlays in `rsetup`, which isn't my problem. But the person's serial log has the exact same two lines, `UFS GPIO7 value is 1` and `Skipping UFS initialization`, and they boot from NVMe. So on a board with no UFS, GPIO7 reads 1 and that's normal. On my board it reads 1 with a UFS module sitting right there. The firmware isn't broken, it's getting the wrong answer from the pin.

Their log also says `Boot Device : NVME`, and so does mine even though I don't boot from NVMe. So that line is probably just something this firmware always prints.

## Finding GPIO7 from Linux

Linux can read the same pin. `gpiodetect` lists the GPIO chips:

```
gpiochip0 [c440000.spmi:pmic@8:gpio@c000] (12 lines)
gpiochip1 [c440000.spmi:pmic@1:gpio@8800] (10 lines)
gpiochip2 [c440000.spmi:pmic@2:gpio@8800] (9 lines)
gpiochip3 [c440000.spmi:pmic@0:gpio@b000] (4 lines)
gpiochip4 [f100000.pinctrl] (176 lines)
gpiochip5 [33c0000.pinctrl] (15 lines)
```

The small ones are the power management chips, `f100000.pinctrl` is the main processor. This Ubuntu has the older GPIO tools, so it's `gpioget gpiochip4 7` without a `-c`. I read pin 7 on every chip that has one. Only the main processor's pin 7 read `1`, every power chip read `0`. So the firmware's "GPIO7" is the processor's GPIO 7, and I can check it from Linux any time.

I powered it all the way off, pulled the UFS module, checked it, and pushed it back in firmly. Still `1`.

## The power theory

Here's what I think is going on. For the module to say "I'm here" on that pin, it probably needs to be powered. If the firmware checks the pin before it powers the UFS, the module can't answer, the pin sits at its default `1`, and the firmware skips UFS forever. Android's firmware never checked this pin, it just powered the UFS and used it. EDL powers the UFS itself too. That's why both of those worked.

The UFS controller's device tree entry lists what power it needs:

| Supply | Rail | State in Linux |
| --- | --- | --- |
| `vcc` (2.96 V, the flash itself) | `vreg_l7b_2p96` | disabled |
| `vccq` and `vccq2` (1.2 V) | `vreg_l9b_1p2` | unknown |

I booted with `regulator_ignore_unused` so Linux wouldn't switch off rails it thinks nobody's using. `vcc` still showed disabled. Then I found out why that test doesn't prove anything. That same 2.96 V rail is **shared** with the eMMC controller (`7c4000.mmc` uses it as `vmmc`). The eMMC and UFS share a socket, so they share its main power pin. Linux's eMMC driver grabbed the rail, looked for an eMMC card, found nothing, and turned it off itself.

But it made the theory stronger. The firmware's log goes in this order: tries eMMC (`Card Detection FAILED`), `eMMC module not found`, then `UFS GPIO7 value is 1`. If it powers the socket down after the eMMC probe fails, the UFS module is unpowered right when GPIO7 gets read.

One more thing about the hardware. The socket on the board is actually two stacked connectors, the top one is UFS and the bottom one is eMMC. My official Radxa UFS module plugs into both, and the eMMC side on the module is a real, fully populated connector. So if the firmware only powers the parts it's using for its eMMC check, the UFS side might not be getting what it needs to answer.

## The firmware edits the device tree, even when you bring your own

Radxa's own overlay tool (`rsetup`, in `/usr/lib/rsetup/cli/edk2-menu.sh`) turned out to be the best documentation for how the bootloader works. It copies the kernel package's device tree to the EFI partition and writes two lines into the boot entry, `devicetree` and `devicetree-overlay`. So embloader does support `devicetree`. When my own device tree "got ignored" earlier, it actually got loaded and then edited.

The kernel package ships its own map for this board, `/usr/lib/linux-image-6.18.2-3-qcom/qcom/qcs6490-radxa-dragon-q6a.dtb`, and in that one `ufshc@1d84000` is `okay`. I copied it to `/boot/efi/RadxaOS/6.18.2-3-qcom/` and added `devicetree /RadxaOS/6.18.2-3-qcom/qcs6490-radxa-dragon-q6a.dtb` to the entry. After boot, UFS was still `disabled`. The embloader log explained it:

```
linux_try_load_dtb: load dtb /RadxaOS/6.18.2-3-qcom/qcs6490-radxa-dragon-q6a.dtb successfully
efi_dtfixup: applying efi dtfixup protocol
efi_dtfixup: efi dtfixup applied successfully
```

embloader loads your map, then asks the firmware to "fix it up", and the fixup switches UFS back off.

embloader is open source ([BigfootACA/embloader](https://github.com/BigfootACA/embloader)). In `embloader/src/linux/preboot.c` it only calls the fixup if the setting `efi.dtfixup` is true, which is the default. The settings live in an `embloader` folder at the root of the boot partition, which didn't exist, and that's why the log kept saying `embloader folder not found`. So I made `/boot/efi/embloader/config.static.yaml`:

```yaml
efi:
  dtfixup: false
```

The log showed it loaded the config and skipped the fixup. UFS was still `disabled`. So the firmware fixes up the device tree again on its own right at hand-off, after embloader is done. That's the `DeviceTreeAutoFixup` firmware setting.

## Turning off the firmware's auto fixup broke the boot

Firmware settings can be written from Linux on this board. `efivarfs` is mounted read-write, and the kernel sets up `qcom.tz.uefisecapp`, which is the bridge to the firmware's settings through TrustZone. The value is `07 00 00 00 01`, four bytes of attributes and then the setting. Each variable has Linux's immutable flag on it so you can't wipe one by accident.

```sh
F=/sys/firmware/efi/efivars/DeviceTreeAutoFixup-882f8c2b-9646-435f-8de5-f208ff80c1bd
sudo chattr -i $F
printf "\x07\x00\x00\x00\x00" | sudo tee $F >/dev/null
```

It read back as `07 00 00 00 00`. On the next boot the board went into a restart loop. Solid blue, the USB stick blinks, blue goes off, blue comes back, over and over. So the fixup is something the board actually needs.

To get back I restored just the settings area from the SPI dump I took earlier. In that settings store each record goes: data size (4 bytes), name size (2 bytes), the name in UTF-16, then the value at the next 4 byte boundary. Every setting is stored twice, like two save slots. I checked my saved copy had `DeviceTreeAutoFixup` at `01` in both slots, then put the board in EDL:

```sh
edl-ng --memory=Spinor --loader=prog_firehose_ddr.elf read-sector 138 160 varstore-before-restore-looping.bin
edl-ng --memory=Spinor --loader=prog_firehose_ddr.elf write-sector 138 varstore-20260928-fixup-on.bin
```

Read it back, it matched byte for byte, and the looping copy showed `00` in both slots, so the Linux write really had landed on the chip. It booted normally again.

Quick side note, my Ethernet cable is missing its clip and doesn't always seat all the way. Since the board runs on PoE, a loose cable can mean no network and flaky power. A couple of "host is down" moments were just that.

## What the fixup actually changes

I compared the kernel's map straight off the disk against the one Linux actually got (`/sys/firmware/fdt`), both run through `dtc -s` so the order matches. The whole fixup is just this:

| Where | Before | After |
| --- | --- | --- |
| `chosen` | nothing | boot options, memory map, `radxa,dtb-fixup-applied` |
| clock controller `gcc` | `protected-clocks` | the same list with three entries repeated |
| GPIO controller | `gpio-reserved-ranges = <0x0c 0x06>` | the same range twice |
| UFS crypto engine `1d88000` | on | `disabled` |
| UFS controller `ufshc@1d84000` | `okay`, linked to the crypto engine | `disabled`, link removed |
| watchdog `17c10000` | `reserved` | `okay` |

The watchdog is a hardware timer that resets the board if nothing keeps petting it. My guess is that's the restart loop. Without the fixup the watchdog stays `reserved`, Linux never takes care of it, and it keeps resetting the board.

## Where I'm at

The fixup isn't really the villain. It just writes down what the firmware found at boot, and what the firmware found is "no UFS" because GPIO7 read `1`. I could build a map with UFS on and the watchdog on and get Linux to use the UFS, but the firmware would still never start its own UFS driver, so it could never boot from it. So I'm not doing that. The real problem is why GPIO7 reads `1` with a UFS module installed, and that's what I'm going after next.

## EDL already proved the UFS itself is fine

The thing that kept bugging me is that I wrote to LUN 0 over EDL all day, 10 GiB in 34 seconds. So what state is the board in when that works? In EDL the chip skips the SPI chip completely. No XBL, no UEFI, no GPIO7 check. `edl-ng` uploads the firehose program into RAM and I tell it `--memory=Ufs`. So EDL gets told "this is UFS" and just powers it up and talks to it. The normal boot has to figure it out from GPIO7. That's the only difference.

I tried reading GPIO7's register straight from EDL with the open source [bkerler/edl](https://github.com/bkerler/edl) tool (`edl peekdword 0x0F107004`, since the GPIO block starts at `0x0F100000` and each pin gets `0x1000`). This firehose loader doesn't allow it: `Peek command isn't supported by edl loader`. But while it was starting up the storage it printed what's actually on the module:

```
mem_type:        UFS
manufacturer_id: 462   (Samsung)
prod_name:       KLUEG8UHDC-B0E1
num_physical:    1
```

A real Samsung 256 GB UFS 3.1 chip, one LUN from the Armbian layout. Then I pulled the UFS out and ran it again, and the loader crashed trying to start UFS (`USBError(5, 'Input/Output Error')`) and the board dropped off USB. So EDL really was talking to the module. The UFS works. Only the detection fails.

The chip's own boot log also has `Secure Boot: Off` and `Boot Interface: SPI`. So the `Boot Device : NVME` line in the UEFI log is just a label, the chip knows it booted from SPI.

## The schematic

The Q6A schematic shows the storage socket has three detect lines right next to each other:

| Connector pin | Signal |
| --- | --- |
| 19 | `UFS_DET`, goes to QCS6490 GPIO_7 |
| 20 | `FSPI_DET` |
| 21 | `EMMC_DET` |
| 16, 22, 24, 27, 30 | GND |

Each one has a pull-up on the board, and they're active low. A module tells the board what it is by tying exactly one of them to ground. So a UFS module should ground pin 19 and pull GPIO7 to `0`. That also kills my power theory. It's a plain wire to ground, so it doesn't need the module to be powered at all.

## Using the board as a multimeter

I don't own a multimeter. But Linux can show the level of every GPIO pin without touching any of them:

```sh
sudo cat /sys/kernel/debug/gpio
```

With the module in, GPIO 7, 8 and 9 all read high. 8 and 9 read high even though the chip's own weak pull is trying to hold them low, so something on the board is pulling them up. Those are probably the other two detect lines. Then I powered off, pulled the module, booted and took the same snapshot again:

```sh
diff gpio-module-IN.txt gpio-module-OUT.txt
```

Identical. All 231 lines. Taking the UFS module out doesn't change a single pin on the whole board. With the module in, the board literally can't tell it's there. (First time I ran it I got "no difference" because the board wasn't up yet and the OUT file was empty. Check that the file actually has 231 lines.)

## The module schematic

Radxa has a schematic for the UFS module too (`RADXA_UFS_MODULE_V1.1_SCH_20250423.pdf`). On the module side:

```
J1 pin 21  EMMC_DET  R11  0 ohm   (NC)  to GND
J1 pin 20  FSPI_DET  R10  0 ohm   (NC)  to GND
J1 pin 19  UFS_DET   R2   10k     (NC)  to VCC_1V8
                     R9   0 ohm         to GND
```

NC means not connected, those parts are left off on purpose. R9 is the only one that's actually there, a 0 ohm resistor that ties `UFS_DET` to ground. That's the whole "I'm a UFS module" signal. The ground symbols are drawn right on top of each other next to `VCC_1V8`, so at first it looked like R9 might go to 1.8 V. I pulled the line coordinates out of the PDF with PyMuPDF, and each resistor ends on its own symbol. R2 goes to 1.8 V, R9 goes to ground.

Also the second connector on the module (J2, the one that sits in the eMMC side) has nothing connected to any pin. It's just there to hold the module in place.

## Found it

So I took the module out and looked at it with my phone's macro camera. Right by the J1 connector, in a little group of four pads, there was a tiny black `0402` part ripped up on one end, sitting at an angle, barely attached. The pad under where it used to sit had torn solder and bare copper where the solder came off with it. That's R9.

![R9 knocked off the UFS module, next to the J1 connector](images/q6a-ufs-r9-knocked-off.jpg)

![Close up of R9 lifted off its pads](images/q6a-ufs-r9-closeup.jpg)

It got knocked off, so `UFS_DET` was never grounded, GPIO7 floated up to `1`, and the firmware skipped the UFS. Every single thing I saw today comes back to that one part. Android and EDL never cared because they don't check that pin.

I have no idea how it got knocked off. It sits right at the edge of the module next to the connector, so pushing on that corner or prying the module out could easily do it. From now on I only press straight down on the connector and lift it out evenly.

Putting it back was its own thing. Those four pads are two footprints, R9 and R2. To figure out which way they run I looked at the rest of the board. The other unfitted footprints right next to them (R10 and R11) are horizontal pairs, so these are too. The top pair has two neat little solder domes like R10 and R11, so that's the unfitted one, R2. The bottom pair is where the damage was, so that's R9. With the part laid flat across one horizontal pair, it can't bridge two footprints, which matters because that could short 1.8 V to ground.

I set it back on the bottom pair with tweezers, black side up. I recorded the whole thing, about five minutes, with my phone zoomed in as far as it goes. It's so small compared to my tweezers.

![R9 set back on its pads before soldering](images/q6a-ufs-r9-placed.jpg)

It's not soldered yet. I don't have a soldering iron, and nothing local had one, so I ordered a FNIRSI HS-02A USB-C iron, some low temperature Sn42/Bi58 solder paste, and 0.5 mm lead free rosin core wire. The plan is a pinhead of paste on each of the bottom pads, the part on top, and 1 to 2 seconds per end at about 250 to 280 C. The top pads are R2, and one of them is 1.8 V while R9's other pad is ground, so I have to make sure nothing bridges between the two rows before the module goes back in. When it's soldered, the module goes back in, and `gpioget gpiochip4 7` should finally read `0`.

## I couldn't wait (the unsoldered test)

Okay so… I couldn't help myself. I wanted to end the night on a verdict. My setup had the module lying connector side up, so I could lower the Q6A straight down onto it without flipping anything, and gravity would keep R9 sitting on its pads. Power and Ethernet off, lowered it very very carefully, sweating the whole time, then powered it up with the USB stick in.

```
GPIO7: 1
ufshc: disabled
UFS GPIO7 value is 1
Skipping UFS initialization
```

Still `1`. Which honestly makes sense. An unsoldered `0402` resting on pads has basically no pressure on it, and one of those pads is bare copper where the solder got ripped off. So it's probably just not touching both pads. That result doesn't prove or disprove anything, it just means it has to be soldered.

I did double check the placement though. Radxa's own product photo of the UFS module (V1.10) shows the connector side with the four empty pads along one edge (R10 and R11), and right below them a 2 by 2 block with one empty footprint (R2) and one fitted part (R9). Turn that photo a quarter turn and it lines up with my board exactly, chamfered corner and all, and the fitted part sits right where I put it, the bottom pair, lying horizontal. So the placement is right. It just needs solder.

## IT WORKS 🎉

Soooo… I couldn't leave it alone. I lifted the Q6A off, reseated it super carefully again, powered it back up, and ran the same check.

```
GPIO7:
gpioget: error reading GPIO values: Device or resource busy
ufshc: okay
NAME     SIZE TRAN
sda    238.4G
├─sda1   128M
├─sda2     1G
└─sda3   8.7G
sdb     58.6G usb
```

LET'S GOOOO. `ufshc: okay` means the firmware left the UFS on this time. `sda` is the UFS showing up in Linux with Radxa's three partitions, and the USB stick got bumped to `sdb`. And the firmware's own log:

```
UFS GPIO7 value is 0
UFS INQUIRY ID: SAMSUNG KLUEG8UHDC-B0E1 0200
UFS Boot LUN: 1
UFS GEAR: 4
```

GPIO7 finally reads `0`, the firmware found my Samsung UFS by name, and it's running at Gear 4, which is full UFS 3.1 speed. The pin dump shows GPIO7 as `low` now too. (`gpioget` saying "busy" is fine, it just means something's using that pin now.)

So that's it. The whole day, the black screens, the restart loop, the device tree fights, all of it came down to R9, a 0 ohm resistor about the size of a grain of sand that got knocked off the module. And I put it back by hand with tweezers, no soldering iron. It's only resting on its pads right now and it happened to make contact this time, so I'm not touching anything until it's soldered tomorrow. That makes it permanent.

## And then it booted from the UFS 😭🎉

Obviously I couldn't stop there. The UFS boot partition still had Claude's test version on it from earlier (the one with the USB stick's kernel and root pointed at the USB stick). But now that Linux could actually see the UFS, I didn't even need EDL to fix it.

I still had an untouched copy of the UFS image's original boot partition on my Mac from when we looked inside it. I checked it was byte for byte the same as partition 2 of the image, gzipped it (1 GiB down to 58 MB since it's mostly empty), copied it to the board, and made sure I was writing to the right disk first:

```
NAME   SIZE          LOG-SEC  MODEL
sda    255944818688  4096     KLUEG8UHDC-B0E1
└─sda2 1073741824    4096
```

That's the Samsung UFS, and `sda2` is exactly the size of the partition in the image. Then:

```sh
gunzip -c /tmp/ufs_efi.img.gz | sudo dd of=/dev/sda2 bs=4M conv=fsync
```

Checksum matched, and the boot entry pointed at the UFS's own root again (`root=UUID=6783c260-922f-46a9-ba5a-7c202d0552dd`). Shut it down, pulled power and Ethernet, pulled the USB stick out super gently so I wouldn't bump R9, and powered it back on with nothing but the UFS.

```
NAME   SIZE   MOUNTPOINTS
sda    238.4G
├─sda1 128M   /config
├─sda2 1G     /boot/efi
└─sda3 237.1G /
TARGET SOURCE    FSTYPE
/      /dev/sda3 ext4
```

IT BOOTED FROM THE UFS. Root is on `sda3`, it already grew itself to fill the whole 238 GB, and there's no USB stick anywhere. Little Guy is running off its own storage for the first time. This morning the board wouldn't even show a picture. Tonight it's booting Linux off a UFS module I fixed by hand. I recorded it too 😁.

## How I actually figured it out (the back and forth)

The full version of this, told like a story with the actual conversations mixed in, is in [my recap of 2026-09-28](2026-09-28.md). Here's the short version.

I want this part in here because honestly it's the part I'm proudest of. I didn't know anything about Linux boot chains, UFS, EDL or schematics before today. I didn't even know what UFS stood for (Universal Flash Storage, which is funny because it's the least universal thing on this board). What I did have was a way of thinking through it. Most of today was me and Claude going back and forth, and a lot of the breakthroughs came from me pushing back or asking the next question. Here's how it actually went, in order.

**"Did you actually look at the project?"** Before any of the hardware stuff, Claude suggested putting a tiny 1B model on the NPU. I pushed back hard. Little Guy's brain has to do audio, video, thinking and responding all on one board, so a small text model defeats the point. That's what locked in Gemma 4 E2B as the target, with E4B as the real goal. And when Claude pointed out nobody's gotten a model that size running on this NPU, my answer was basically, good, that's the challenge.

**"Stop, loop me in."** For a big chunk of the afternoon Claude was running test after test by itself, and I realized I wasn't learning anything. So I told it to stop and explain. It tried a quick three line summary and I called that out too, because after hours of work a summary isn't learning. We went through it piece by piece instead, starting from what the SPI chip even is. (I asked if it was like a CMOS chip, and then clarified I meant the CMOS battery, because I thought the battery controlled the BIOS. Turns out the battery only keeps the settings memory alive. On this board the settings live in flash, so there's no battery at all.)

**Walking the boot chain myself.** Once I had the boot stages in my head, I eliminated them one at a time. The Linux kernel can't be deciding, because it's already Linux running and it just obeys the device tree. embloader just picks which kernel to run. PBL just jumps to the SPI chip. So the decision had to be happening in the firmware on the SPI chip. Then my next question was, okay, so what exactly did we put on the SPI chip today? That's what got us listing every piece of the Linux BIOS and what order they run in.

**"UFS is basic hardware."** When I saw that XBL "sets up RAM and basic hardware", I figured a storage controller counts as basic hardware, so my money was on XBL. We compared the strings in the Android XBL and the Linux XBL, and they were almost identical. Claude was ready to clear XBL based on that. I pushed back, because having the same code doesn't mean it runs the same way on my board. I asked if there was a way to see how the board was actually handling it. That question is what led to reading the firmware's own boot log out of RAM, and that log is where `UFS GPIO7 value is 1` showed up. It turned out the check happens in UEFI, one stage after XBL, so XBL wasn't the culprit. But asking how the board actually behaved, instead of what the code looked like, is what cracked it open.

**"Read that again."** When Claude explained the device tree fixup, it said the firmware "takes the map, edits a few lines to match what it found at boot, and passes the edited version on." I stopped it right there and told it to read that again. To match what it found. So the fixup was never really the problem, it was just honestly reporting what the firmware found. Claude agreed that everything we'd been doing to get around the fixup was treating the symptom. The real question was why the firmware found "no UFS" in the first place.

**"You think I'd half ass this?"** At that point Claude laid out an easy way out, boot from USB forever with a hand made device tree and just use the UFS from Linux. I shut that down. It would work, but the firmware still wouldn't know the UFS exists and I'd have a USB stick in there as a crutch forever. Roadblocks are the fun part, that's where I actually learn. So we went after the root cause.

**"In what state were you able to write to LUN 0?"** This one mattered a lot. The UFS obviously worked, because we'd been writing to it over EDL all day. So I asked what's different about EDL. The answer was that in EDL we *tell* it "this is UFS" and it just powers it up and talks to it, while the normal boot has to *detect* it with GPIO7. That proved the UFS itself was fine and only the detection was failing.

**The chicken and the egg.** My theory was that something has to send power to the UFS before it can say "I'm here", so if the firmware checks the detect pin before powering the module, it can never answer. I'm not going to pretend I was 100% right. When I found the schematic myself, it showed `UFS_DET` is a plain wire to ground on the module and doesn't need any power at all, so the power part of my theory was wrong. But I was in the right ballpark. The detection was failing before anything got set up, it was a hardware signal, and chasing the power question is what got us reading the power rails. That's where we found out the eMMC and UFS share the same 2.96 V rail, and it's what sent me to the schematic in the first place.

**The schematic, my first one ever.** I found the `UFS_DET` to `GPIO_7` line myself, figured out it's active low with a pull up on the board, and worked out the logic. The pull up makes it read `1` if nothing's there, so the module has to pull it to ground, which means the chain between the module and GPIO7 had to be open somewhere. I came up with the plan to check continuity from the module's detect contact to ground, and from the connector to the board's net. I also asked if the fusing step from installing Android could be doing this, since I'd been told that's not reversible. It can't, fuses can't hold a pin at a voltage. And I noticed the socket is actually two stacked connectors, and wondered if the firmware was only powering one of them. (The module schematic later showed the second connector isn't even wired up, it's just there to hold the module in place.)

**Finding the evidence myself.** I dug up [radxa-build/radxa-dragon-q6a#4](https://github.com/radxa-build/radxa-dragon-q6a/issues/4), where a board with no UFS logs the exact same `GPIO7 value is 1` line. That's what told us `1` just means "no UFS detected". I spotted `Secure Boot: Off` in the chip's boot log right after Claude had guessed secure boot was on. I found the UFS module's schematic, pointed at the NC resistors, and then went looking at the actual module with my phone camera until I found R9 ripped up at an angle.

**Context over guessing.** When it came time to put R9 back, Claude guessed which pair of pads from the solder on them. I looked at the rest of the board instead. The other unfitted footprints right next to it all run horizontally, so these do too, which means the bottom pair. That was right, and Radxa's own product photo backed it up. Reading the situation around the problem beat staring at the four pads on their own.

**Healthy skepticism.** When the unsoldered test still read `1`, I asked if we were somehow screwing something up with the firmware, because if R9 was really the problem it should've worked. Claude walked through why the GPIO reading is a raw electrical fact the firmware can't fake. Then I reseated it, it made contact, and it worked.

Looking back, "go check the hardware" sounds obvious. It didn't feel obvious at 2 PM with a board that just sat on a solid blue light. What got us there was eliminating possibilities one at a time, questioning things instead of accepting them, using the context around each problem, and not settling for a workaround. I don't know Linux or UFS inside out. But I can reason through stuff I've never seen before, and today proved it.

## Decisions I made along the way

- **Linux over Android.** Android was using several gigabytes of the 12 GB just for itself and its low-memory killer ended my E4B test. Idle Ubuntu leaves about 10 GB free, and all the NPU tooling is documented for Ubuntu anyway. Android was only ever my test harness.
- **Keep Android recoverable.** Before touching the boot firmware I backed up the whole SPI chip and kept the Android package. Even after wiping the UFS I can put Android back.
- **Prove the NPU works before compiling anything.** Radxa's prebuilt Llama 1B was the fastest way to know the NPU and its drivers work on my exact board. It did, so the Gemma plan in [NPU_PORT.md](NPU_PORT.md) is worth chasing.
- **Wipe the UFS for Linux.** Android couldn't boot with the Linux BIOS on the SPI chip anyway, and the USB stick is on a slow USB 2.0 link.
- **Stop experimenting blind and reason it out.** Once I started asking what was actually making the decision at each boot stage, it went from guessing to narrowing it down.
- **Go after the root cause.** At one point there was an easy way out. Boot from the USB stick with a hand made device tree and use the UFS from Linux. It would have worked, but the USB stick would have stayed in forever as a crutch and the firmware still wouldn't know the UFS exists. I wanted to know why GPIO7 read `1`, and that's how I found R9.
- **OPSEC.** Before pushing any of this to GitHub, the board's serial number, MAC address and SSH key fingerprint came out of the docs, and I checked the photos for location data.

## Things we got wrong

This whole thing was a lot of wrong guesses that got corrected by testing. Some of them were Claude's, some were mine.

- **"The firmware thinks the UFS is an eMMC."** Close. It turned out the firmware never saw any module at all.
- **"The UFS layout is the problem."** Re-provisioning from Android's eight LUNs to one LUN changed nothing about the detection.
- **"embloader ignores the `devicetree` line."** Wrong. It loaded my device tree fine and then the firmware's fixup switched UFS back off. We only figured that out by reading embloader's log.
- **"It's XBL."** That was my theory. The string compare couldn't rule it in or out, and the firmware log later showed the GPIO7 check happens in UEFI, one stage later.
- **"The module needs power to announce itself."** Also my theory, and I liked it. The schematic killed it, `UFS_DET` is a plain wire to ground and needs no power.
- **"Secure boot is on, we can't touch the firmware."** Claude guessed that from the `PK_HASH` value EDL printed. The chip's own boot log says `Secure Boot: Off`. That hash is probably just Qualcomm's default for boards that aren't locked down.
- **"R9 goes on the left pair of pads."** Claude's guess from which pads had solder on them. I looked at the other footprints around it, saw they all run horizontally, and put it on the bottom pair. That also rules out the part bridging two footprints and shorting 1.8 V to ground.
- **Reading the settings store.** Claude's first read of `DeviceTreeAutoFixup` in the saved copy said `00` when it was actually `01`. It was reading the byte before the value, which sits on the next 4 byte boundary. Good thing we looked at the raw bytes before writing anything back.

## Bloopers

- I ran the `chattr` and `tee` commands for changing the firmware setting on my Mac instead of on the board. Nothing happened except `od` trying to open files called "read", "it" and "back", because my shell didn't treat the `#` comments as comments.
- Typed `eld` instead of `edl`. Twice-ish.
- Claude killed its own SSH session with `pkill -f "apt-get update"`, because the pattern matched the command line of the very SSH command it was running.
- Claude wanted to test a device tree without rebooting using `kexec`. `apt` hung trying to install it and we gave up on that idea.
- The first "identical" GPIO diff was fake. The board wasn't up yet, the OUT file was empty, and Claude caught it by counting lines.
- A bunch of "host is down" moments were just my Ethernet cable, which is missing its clip and doesn't always seat all the way.
- My Mac said it saw zero USB devices while the board was "in EDL". It wasn't in EDL. PoE was keeping it powered the whole time so it never actually restarted into EDL.
- `edl-ng reset` put the board straight back into EDL instead of booting it, and I sat there watching my network for a board that was never coming.
- My Mac wouldn't mount a FAT partition because its FAT driver failed its own code signature check. `mtools` saved me.
- Halfway through I switched my Mac to my phone's hotspot for a download and then couldn't reach the board, because it was on a different network.
- When Claude asked if I had a soldering iron, Claude Code's autocomplete had already typed "i dont have a soldering iron" into my message box. It knew.
- I did the R9 placement with my phone zoomed all the way in and tweezers that are enormous compared to that part. I recorded the whole thing. The first four minutes are me figuring out how to set up the camera.

## What I learned

- **How a Qualcomm board boots**, stage by stage, and how to work out which stage is making a decision.
- **EDL**, reading and writing the SPI chip and UFS directly, backing up firmware, and recovering from a bad setting.
- **Device trees**, what `okay` and `disabled` do, and that the firmware can rewrite the map you hand it.
- **UEFI settings**, which ones Linux can see, how to change one from Linux, and how to read the raw settings store.
- **How to read a schematic**, my first ever. Pull-ups, active low detect straps, 0 ohm resistors used as jumpers, and what NC means.
- **Using software as test equipment.** No multimeter, so the firmware's own log and a before and after GPIO snapshot did the job.
- **That the fixup was just the messenger.** It honestly reported what the firmware found. The real problem was a tiny part that got knocked off.
- **How to place an `0402`**, and next, how to solder one.

## Where everything stands right now

So future me doesn't have to dig through all of this:

- **SPI chip:** Radxa's Linux BIOS `dragon-q6a_flat_build_wp_260120`. The settings area (`VarStore`) is back to the known good copy with `DeviceTreeAutoFixup = 1`.
- **Backups:** the original Android SPI chip is `.run/spi-backup/spinor-lun0-android-20260927.bin` (SHA-256 `feda0117...`) with a second copy in `~/Documents`. The Linux BIOS SPI dump is `spinor-lun0-linuxbios-20260928.bin`. The settings area from the restart loop is saved as `varstore-before-restore-looping.bin`.
- **UFS module:** this is what the board boots from now 🎉. Armbian's single LUN layout, Radxa's UFS image on LUN 0 with its original boot partition, root on `sda3` grown to 237 GB. R9 is only resting on its pads, so soldering it is the next job, and until then I'm not bumping the board.
- **USB stick:** out of the board, now a spare. Its boot entry still has my test changes (`regulator_ignore_unused`, `devicetree /RadxaOS/6.18.2-3-qcom/qcs6490-radxa-dragon-q6a.dtb`, and `/boot/efi/embloader/config.static.yaml` with `dtfixup: false`), and the untouched entry is backed up at `~/entry.before-regtest.conf` on it. I'll clean it up or just reflash it.
- **Board setup:** the UFS install is fresh, so it has new SSH host keys and none of my setup yet. My SSH key (`~/.ssh/id_ed25519_q6a`, the `q6a` host alias on my Mac) is already copied over. Still to redo on it: mask sleep, passwordless sudo for `radxa`, and journal access. The USB stick install has all of that already.
- **Firmware log trick:** the UEFI boot log is in RAM at `xbl-uefi-res@80880000` and I can read it any time by mapping `/dev/mem` with Python. The copy from today is `.run/fwdump/uefi-log-20260928.txt`.
- **GPIO snapshots:** `.run/gpio-module-IN.txt` and `.run/gpio-module-OUT.txt`.

References: [Radxa downloads](https://docs.radxa.com/en/dragon/q6a/download), [Radxa SPI firmware flashing](https://docs.radxa.com/en/dragon/q6a/low-level-dev/spi-fw), [Q6A boot chain overview](https://rootcommit.com/2026/board-bring-up-radxa-dragon-q6a-with-mainline-linux-and-yocto/).
