# Phase 1 body and stand layout

User decision, 2026-10-21: **Jetson on the stand for phase 1; leave room to move it inside the torso later.** The Alpha remains the separate test body. The user's own speaker remains an unresolved packaging input.

## Placement now

| Assembly | Phase 1 placement | Layout consequence |
|---|---|---|
| Jetson Orin Nano Super developer kit | Stand-mounted removable tray | Keep cooling and connections accessible; its mass is outside the moving torso |
| USB servo adapter | Candidate: on the stand near Jetson | Short accessible USB link; servo bus passes through the supported body tether |
| Servo source protection and disconnect | Candidate: stand | Accessible disconnect and protection before the main trunk; actual components remain unselected |
| Servo distribution hub | Candidate: near torso, location unresolved | Eleven protected feeds and returns in the [harness draft](power/README.md); fit around the future bay and account for mass and waist cable motion |
| Shoulder, elbow, wrist and head joints | Moving upper body | Arm load budgets include their carried parts and cables |
| Waist bearing and drive | Stand-to-torso interface | Carry torso loads through the structure; verify support and overturning resistance |
| Future Jetson bay | Empty removable-tray region in torso | Do not occupy it permanently with the speaker, waist support or shoulder brackets |

These placements establish packaging intent, not attachment-hole coordinates or a completed harness. Allow service loops and strain relief across waist motion; the tether must not carry structural loads. Specify a bounded waist travel before routing it, rather than assuming unlimited rotation.

## Future torso space reservation

NVIDIA's [carrier-board specification, chapter 4](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf) gives the assembled developer kit as **103.00 ± 0.20 × 90.50 ± 0.20 × 34.77 ± 1.09 mm**, and 0.175 kg. The bare carrier is 100 × 79 mm and is not the right envelope for the assembled kit. NVIDIA's [FAQ](https://developer.nvidia.com/embedded/faq) states that the Super kit has no physical hardware/package differences from the original kit, so the drawing is an appropriate initial reference. The actual ordered unit, accessories and mounts still require checking.

Reserve a **130 × 120 × 65 mm provisional unobstructed bay** around a removable tray in the first torso layout. This is a design allocation, not a manufacturer enclosure recommendation or a fit guarantee. It exceeds the published assembled-kit dimensions but does not establish room for every plug, cable bend, storage accessory or fan airflow path. Model those separately from the exact installed hardware before fixing torso width/depth. If the clothing/shoulder layout cannot accommodate them, revise the body layout rather than silently shrinking the reservation.

The bay needs:

- A removable access panel and a clear tray-removal path, without dismantling the shoulder load path.
- Separate connector and cable-bend keepouts for the ports actually used, plus underside clearance for fitted storage and mounting hardware.
- Air inlet/exhaust paths that remain open with clothing fitted. Do not place fabric directly over the fan or treat empty space alone as thermal validation.
- A future internal cable route from Jetson to the servo adapter, sensors and audio. The phase 1 stand harness can be replaced or shortened when the computer moves.
- Mounting features tied to verified board/tray geometry, with insulation and clearance under the electronics. No mounting holes have been released yet.

When the Jetson moves inside, recalculate torso centre of mass, waist inertia, bearing load and stand stability with the complete tray, cabling, storage and cooling assembly. The manufacturer's 175 g kit mass does not include every future accessory. Retest power and cooling under the intended AI workload inside the clothed torso. Keep the Jetson supply correctly specified; moving it inside does not authorize connecting it to the servo rail.

## Dimensions still to establish

The approximate 55 cm finished robot height is retained. Shoulder spacing, torso height, neck position, exact arm lengths, speaker envelope, waist bearing size, stand footprint and ballast remain open. The compact 100 / 100 / 60 mm arm scenario from [the load study](mechanical-envelope.md) is a layout candidate only. It has little spare mass budget and is not a committed physical proportion.

The first [spatial packaging study](layout/README.md) now places the empty future bay, eleven servo envelopes and proposed neck/waist/frame corridors together. It uses a provisional 220 × 140 × 190 mm torso reference, and the resting arm envelopes reach about 279 mm across. Six limb/elbow intrusions into the rectangular torso reference remain unresolved; a shaped torso or revised joint placement is needed. Future Jetson fit is checked with the kit upright, not in its stand orientation. The user's speaker remains absent because its dimensions are unknown. The study provides a visible candidate to refine; it does not establish clothing fit, printable part splits or mechanical interfaces.

The [stand sensitivity study](layout/stand/README.md) now distinguishes that assembly shelf from a floor support footprint. It combines both mirrored arms, the five remaining servos and the stand-mounted Jetson, then compares candidate contact spans and conditional centered-mass requirements. Torso/head/stand/hub masses remain absent, and the calculation does not select a base or prove physical stability.
