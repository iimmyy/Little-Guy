# Review of the second supplied research report

The attachment titled “ALPHA 1S PCB Reconstruction: Resources & Strategy” is different from the first report. Its exact contents are preserved at `../reference/submitted-research/resources-strategy-report.md`, SHA256 `dbaec4852d2d46c1ba7446f420a241cc59dd019a2bda8b5b43c26ee546b053d5`.

It supplies no source URLs, document identifiers, donor pin-to-pin measurements, firmware files or PCB design data. Its suggestions can guide investigation but do not establish additional donor connections.

| Claim or suggestion | Assessment |
|---|---|
| 900 MHz Cortex-A7, 512 MB RAM and 4 GB flash, borrowed from Alpha 1E | Not evidence for this Alpha 1S V1.1 donor. U8 was read directly as STM32F103RDT6. ST identifies it as Cortex-M3, maximum 72 MHz. No additional Cortex-A7 host has been established on this board. |
| STM32 and ARM are separate confirmed processing parts | STM32F103 is itself an Arm-based MCU. The report has not established a second processor. |
| No detailed PCB images found | Incomplete relative to this project: donor photos and video, FCC mainboard photographs and community board images are already archived. None supplies the full donor circuit. |
| 3 W audio implies 3 ohm/1 W or 6 ohm/3 W speaker | Invalid inference. Power rating alone does not determine impedance. User now specifies their own speaker for V2; obtain its actual specification. |
| 7.4 V/2500 mAh pack and approximately 9.6 V adapter | Unverified for this physical donor from this uncited report. Record actual labels and connector assignments. Adapter input and regulated battery charging voltage are different quantities. |
| Battery positive to charger input as an example net | Do not assume a short or treat lack of continuity as a fault. BQ24133 has distinct adapter and battery paths; donor topology still needs recovery. |
| Four layers, one-ounce copper and internal 5 V or 12 V plane | Unsupported assumptions. Do not import into the original-board reconstruction. |
| Board dimensions from product documentation | Existing board dimensions come from user caliper readings, not robot enclosure dimensions or this report. |
| 433 MHz radio, extra CPU module, microphone/IR and joint driver ICs | No donor component evidence or mainboard net assignments supplied. Servo electronics must be distinguished from mainboard electronics. |
| Official manual/software, SD backups, harness map and continuity checklist | Useful categories, but no downloadable resources or completed measurements are supplied. The first donor continuity worksheet is already prepared. |
| Patents might reveal the circuit | A research lead only; no patent number or matching embodiment is supplied. |

Primary sources checked during this review:

- [ST STM32F103RD datasheet](https://www.st.com/resource/en/datasheet/stm32f103rd.pdf), first page: Cortex-M3 and maximum 72 MHz.
- [TI BQ24133 datasheet](https://www.ti.com/lit/ds/symlink/bq24133.pdf), first page: configurable one-to-three-cell charger and distinct adapter/battery paths. Its example is not the recovered donor schematic.

No schematic nets, PCB copper, component substitutions or power settings were changed based on this report. The subsequent user decisions—Jetson-hosted AI and a user-selected speaker—are recorded in `../v2-requirements.md`.
