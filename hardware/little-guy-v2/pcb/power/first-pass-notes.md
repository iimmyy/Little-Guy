> Historical feasibility notes from before circuit capture. See README.md for current status.

# V2 power board: first design pass

2026-10-25. Status: requirements and component feasibility review. No schematic, PCB layout, physical test,
manufacturing release or order exists yet. The 60 x 80 mm / 40 g envelope remains an estimate.

This board follows the 41-servo walking-body requirements in [PCB-BRIEF.md](../../PCB-BRIEF.md) and
[BOM-full-robot.md](../../BOM-full-robot.md). The earlier eleven-servo harness is useful grounding/protection
background, but is not this board's load specification.

## Charging behaviour decided

Support sleeping and awake conversation while charging; sleep is the preferred default (owner, 2026-10-25).
Provide an always-powered charging/supervision domain and a button wake path independent of the Jetson.
Sleep entry must finish the supported sitting sequence and computer shutdown/suspend before removing their supplies.
Exact sleep mode and wake interface depend on the final carrier documentation. Do not equate sleep with opening
the main battery protection switch.

The design must reduce charge current to respect the negotiated input power, remaining system load, cell limits and
temperature limits. Wired input is the proposed preferred source if both inputs are present; this is a design proposal.
Use input selection and reverse blocking; do not simply join wireless output and USB VBUS or assume their powers add.

## Findings that affect the circuit

| Item | Evidence and consequence |
|---|---|
| Leg supply | Feetech STS3250 specification, edition A/0, electrical specifications on PDF page 3: rated input 6-12 V. A full 3S pack is 12.6 V. Direct connection is not supported by this specification; select a regulated/limited leg supply or obtain manufacturer evidence for the exact servo revision. Include regulation tolerance, overshoot and returned motor energy in the terminal-voltage budget. |
| Charger battery path | TI BQ25798 Rev. C, section 6.3: 6 A continuous RMS battery discharge, 10 A peak up to one second. The brief's estimated walking load is already roughly 65-80 W motors + 15-25 W Jetson + 8 W extras, before conversion losses. At 9 V, that is roughly 10-13 A before losses. Do not route the whole robot through this chip's internal battery path. |
| Computer feed | The proposed carrier's 9 V minimum input equals the brief's minimum pack voltage. A separate fuse or switch does not prevent common battery sag. Validate the carrier specification and reserve a regulated feed with sufficient input range, transient isolation and stored energy; size those from an explicit sag duration/current target. |
| Wireless input | A coil alone is not a charger input. Select matched transmitter, receiver/controller, rectification, ferrite and thermal control with documented delivered power. A 15 W receiver cannot meet a 2-2.5 h target for 108 Wh: even an ideal empty-to-full charge takes 7.2 h with all loads off. |
| Battery | 3S2P and 21700 identify topology and size, not cell current capacity, charge rate, exact energy or temperature limits. Select the exact cell before freezing BMS, current limits, fuse and thermistor thresholds. |
| Everyday charge cap | Approximately 90% is a state-of-charge policy. It is not 90% of 12.6 V. Determine it from the selected cells, gauge and charge profile. |

The load figures above remain estimates from the brief, not measured electrical demand. Servo stall-reference sums
are 50.4 A legs and 35.1 A upper body; each 6 V hand is 8.2 A. They are not selected fuse ratings or a requirement
to sustain all motors stalled. Define allowed simultaneous load, peak duration and trip behaviour before choosing
copper, connectors, switches or protection. Independently protected feeds must not be rejoined through servo V+ jumpers.

## Charging-time arithmetic

For 108 Wh added to a depleted pack, the absolute ideal lower bounds are 2.40 h at 45 W and 1.66 h at 65 W.
At an assumed 95% conversion efficiency, those become 2.53 h and 1.75 h, before charge taper or any running load.
The efficiency is an arithmetic assumption, not a verified value for our board. Thus the brief's 2-2.5 h target is
plausible to investigate at 65 W while sleeping, but is not established at 45 W or while awake. The wireless path
needs its own measured budget. Starting charge level and the everyday 90% endpoint also change elapsed time.

## Components under evaluation

| Function | Exact candidate | Current disposition |
|---|---|---|
| USB-C PD controller | TI TPS25751DREFR | Candidate; TI documents charger control and a TPS25751/BQ25798 evaluation design. Check sink-only configuration, dead-battery start, configuration storage and the selected charger pairing before schematic capture. |
| Compact charger | TI BQ25798RQMR | Comparison candidate; documented dual-input selection and integrated power path, but its internal discharge path cannot carry the whole robot's estimated walking load. Any split-domain topology needs a separate current-sensing, termination and backfeed review. |
| External-power-stage charger | TI BQ25731RSNR | Candidate to investigate for the higher-current system. TI lists 1-5 cells, external switching power stage and no battery MOSFET; do not assume the BQ25798's autonomous termination/temperature/power-path behaviour carries over. Establish battery-current sensing, supervision and the system connection from its own documentation. |
| Pack protection, balancing and temperature interlock | Open | Prefer a bought 3S BMS if a documented unit fits and supports the required current and all three segment temperatures. Charger control does not replace pack protection. |
| Fuel gauge; always-on supervisor; leg regulator; two hand supplies; computer feed; 5 V supply; switches and connectors | Open | Size and select after the rail/transient budget and cell choice. No placeholder component is released for purchase. |

Pricing has not been verified from a live populated listing. Search results showed inconsistent cached prices;
none are adopted here. Assembly-house availability, passive parts, external FETs/inductors, shipping and tax remain open.
This is not a completed BOM or a claim that the chips above are the final selection.

## Proposed schematic partition

1. USB-C protection/PD and a separately protected, reverse-blocked wireless-receiver input.
2. Input selection and charging stage, with current sensing that distinguishes system load from battery charging.
3. Battery interface to pack protection, cell taps/segment temperatures and pack-current/fuel-gauge sensing.
4. Hardware motor-cut path followed by separately protected/switchable legs, upper body and left/right 6 V hand feeds.
5. Computer and 5 V logic domains with defined sag protection, sequencing and a return layout that keeps motor
   current out of sensor/reference paths. Separate rails still share a source and require transient verification.
6. Always-powered supervisor, power button/status light and telemetry/control interface to the spinal cord.

Motor-cut behaviour must account for a supported sitting body and motion-energy discharge. It must not depend
only on Jetson software. Rail-off servo data behaviour and phantom powering need to be checked with the spinal-cord
interface. The transient/returned-energy handling also has to work when the battery BMS disconnects.

## Next design work

Complete exact-cell and electrical load/transient budgets; select the charger topology and a wireless receiver with
documented power; verify live component/assembly availability; then capture the schematic and design a short bench
test for its riskiest power section. Switching charger/high-current circuits need the manufacturer's evaluation board
or a correctly laid-out test PCB, rather than a solderless breadboard. Final layout and ordering follow bench evidence.

## Sources checked 2026-10-25

- [Feetech STS3250 A/0 manufacturer specification, distributor-hosted copy](https://www.gotronic.fr/pj2-38757-3355.pdf), PDF page 3. The manufacturer-hosted URL timed out during this review.
- [TI BQ25798 datasheet Rev. C](https://www.ti.com/lit/ds/symlink/bq25798.pdf), section 6.3 and power-path description.
- [TI TPS25751 product documentation](https://www.ti.com/product/TPS25751).
- [TI USB-PD-CHG-EVM-01](https://www.ti.com/tool/USB-PD-CHG-EVM-01), documented TPS25751/BQ25798 evaluation pairing.
- [TI BQ25731 datasheet Rev. A](https://www.ti.com/lit/ds/symlink/bq25731.pdf), initial feature/architecture review only.

The two project briefs are the source of requirements and estimated loads. They are not substitutes for manufacturer
electrical limits or physical validation.
