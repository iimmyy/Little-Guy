# Little Guy V2, phase 1: the top half (head, arms, waist) on a stand

Goal: a ~55 cm Little Guy (baby 0-3 month clothes) built in two phases. Phase 1 is everything above the hips, on a
stand: he turns toward you, looks, gestures, bops, and wears a hoodie. Legs (phase 2) bolt onto the same torso later.
The Jetson (already ordered) is his brain; the Alpha 1S stays whole as the test body.

On 2026-10-21 the user selected this servo-adapter plan in place of the Alpha-derived PCB plan. See [the current implementation and remaining engineering work](README.md). The original PCB files are retained as an incomplete reference.

Prices checked live on 2026-10-20, before tax. "est." means I did not check that line yet.

| ✓ | Part | Pick | Where | Price | Note |
|---|---|---|---|---|---|
| ☐ | 3D printer | Bambu Lab A1 (256 mm cube build area) | ca.store.bambulab.com | $379 CAD (listed "from", was $559) | Earlier candidate. Required build volume depends on each final part and how the torso is split; overall robot height does not establish whether a 180 mm or 256 mm printer fits the design. |
| ☐ | 11 servos (head 2, shoulders 2x2, elbows 2, wrists 2, waist 1) | Feetech/Waveshare ST3215, 12 V: 10 kg.cm rated load, 30 kg.cm stall | Waveshare direct: US$21.99 each, or Polar Parts (Canada): $40.91 CAD each at 10+ | ~US$242 (Waveshare) + shipping, or ~$450 CAD (Polar Parts) | These are the earlier price observations; joint suitability still requires mass and lever-arm calculations. Select the 12 V variant explicitly. |
| ☐ | Servo interface | Waveshare Bus Servo Adapter (A) | Waveshare | US$4.99 | USB communication from Jetson to the servo bus; separate power-distribution design required. One spare was proposed. |
| ☐ | Power | 12 V bench supply and protected distribution, sizing unresolved | Earlier supply-only estimate $30-45 CAD | est. | 10 A remains a candidate, not a validated eleven-servo supply. Published rated-current sum is 9.9 A and all-stall sum 29.7 A. Adapter schematic marks MAX:6A at the servo output supply. See power notes below. |
| ☐ | Plastic | PETG (strong parts) + PLA (shell), 4 rolls | Bambu CA store | "4+ rolls from $19.99 CAD" each, ~$80 | |
| ☐ | Hardware | M2/M3 screws, heat-set inserts, a few bearings | est. $30-50 CAD | est. | Exact list comes from the CAD design |

## Totals (before tax and shipping)

| Path | Total |
|---|---|
| Waveshare servos + 2 adapters | $379 + ~$345 (US$252 converted, approx.) + $40 + $80 + $40 = **~$885 CAD** |
| Polar Parts servos + 2 adapters | $379 + $450 + ~$14 + $40 + $80 + $40 = **~$1,000 CAD** |

Without the printer (if you print through a service instead): subtract $379 and add the service's quote.

These earlier totals retain a $40 power placeholder. They exclude the unresolved distribution, branch protection and any revised supply; they are not a complete current build quote.

## Power review, 2026-10-21

The selected 12 V servo's [published specifications](https://docs.waveshare.com/ST3215_Servo) list 0.9 A rated current and 2.7 A stall current. Multiplying by eleven gives 9.9 A and 29.7 A respectively. Neither is measured consumption of this robot. The former leaves almost no margin on a 10 A supply; representative loaded motion and transients still need measurement.

The [adapter schematic](https://files.waveshare.com/wiki/Bus_Servo_Adapter_A/Bus_Servo_Adapter_A.pdf) marks MAX:6A beside DC_IN0 at the H3/H5 servo-output block. Do not route the full eleven-servo demand through that output path. Supply capacity, distribution, branch fuses, connectors and wiring remain to be designed together. The adapter is a USB bridge and bus interface; the schematic does not provide a controller that can independently enforce a host-loss timeout. The [distribution study](power-distribution.md) retains the 4 / 4 / 3 load grouping; the newer [harness candidate](power/README.md) uses eleven individual protected feeds and returns to avoid aggregate current in servo jumpers; the [mechanical study](mechanical-envelope.md) records current arm-load assumptions.

## Later (not phase 1)

- Mic array: reSpeaker Flex XVF3800, circular (73 mm mic board). Price to check when ordered.
- Phase 2 legs: 12 x STS3250 (50 kg.cm) + battery + charger. Price when the legs are designed.

## Not needed any more

- The Alpha 1S main board copy: the US$4.99 adapter does its job (talking to the servo bus) for the new servos.
