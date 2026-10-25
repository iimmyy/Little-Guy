# Little Guy V2: brief for designing our own circuit boards (2026-10-25)

Handoff brief for whoever designs the boards. Little Guy V2 is a ~60 cm, ~4.4 kg walking humanoid companion robot
(41 Feetech servos, Jetson Orin Nano Super brain). Parts, candidates and open selections are recorded in
[BOM-full-robot.md](BOM-full-robot.md); read it first, this brief adds what the boards need.

Implementation status, 2026-10-25: all four native KiCad schematic drafts now exist under [pcb/](pcb/README.md).
There are no layouts, bench results or fabrication releases. The component and envelope estimates are not frozen.

## Ground rules (from Immy, the owner)

- Breadboards only for short bench tests of the risky circuits; the real electronics are our own PCBs, made and
  assembled at JLCPCB or PCBWay. Immy knows both services.
- Nothing gets placed in the simulation (build step 3) until every part is real: all four boards designed and every
  bought part picked, so placement uses true sizes and weights.
- Prices only from live listings, with the date checked. Never quote a price from memory.
- Immy decides what (goals, trade-offs, looks); the designer decides how and writes all the files. Immy does not code
  or draw schematics, so ask him design questions in plain words, one at a time.
- The project goes public on GitHub: no personal data, home paths or account details in tracked files.
- Tool: KiCad 10 (kicad-cli 10.0.6 is installed on the Mac). Keep each board in its own folder under
  `hardware/little-guy-v2/pcb/<board>/`.

## Order and steps

Most-connected board first: 1 power, 2 spinal cord, 3 feet, 4 head mics. Each board: pick exact chips (live prices)
-> schematic -> short breadboard test of the riskiest part (Immy wires it) -> layout -> order.

## The four boards

### 1. Power board (chest, lying flat just above the battery stack; first estimate ~60 x 80 mm, ~40 g)

Placement update, 2026-10-25: the chest fit search (`tools/body/insides.py`, chest made 2 cm deeper) puts the
battery as three 2-cell sections stacked low in the chest and this board flat on top of them, under the Jetson on
the back wall. Its heat rises away from the cells, but it must not conduct heat into them: leave an air gap or an
insulating spacer between board and pack. The pelvis has no free room (waist and hip servos, gears).

- Input: built-in Li-ion pack, 21700 cells 3S2P, 9.0-12.6 V, ~108 Wh. Battery is NOT removable.
- Charging, two full paths: USB-C Power Delivery up to 20 V (45-65 W laptop chargers, ~2-2.5 h full) and a wireless
  coil in the seat (he sits on a stool dock). Charger must take PD in and charge 3 cells in series. Everyday charge
  capped ~90%; charge only in the cells' safe temperature range (thermistor per battery segment).
  - Owner decision, 2026-10-25: support both sleeping and staying awake/talking while charging; prefer sleep as the
    default. Charging supervision and the power-button wake path must work with the Jetson asleep or switched off.
    Awake use reduces the power available to charge; the time above is a target to validate, not a guarantee for
    either input or every operating mode. See [power-board first pass](pcb/power/README.md).
- Outputs, each fused and switchable:
  - 12 V leg bus: 12 x STS3250 (rated 6-12 V; check its tolerance to 12.6 V full charge), stall 4.2 A each.
    Must be switchable off while he sits (saves ~3.5 W).
  - 12 V upper-body bus: 13 x ST3215 12 V (6-12.6 V), stall 2.7 A each.
  - 6 V hand buses, left and right: each 1 x STS3032 + 7 x SCS0009.
  - Jetson feed (carrier takes 9-36 V).
  - 5 V for the screen board, USB hub and sensors.
- Separate high-current motor rails and clean regulated logic rails; size for motor current spikes, not averages.
  A leg-servo spike must never brown out the Jetson.
- Emergency cut, lit power button on the front door, fuel gauge (his "hunger" reading).
- Measured loads (simulation): servos 8.7 W idle, 12.7 W standing, 23.9 W gesturing, walking estimated 65-80 W;
  Jetson ~15 W typical, 25 W heavy; other electronics ~8 W.
- Earlier harness study (11 individually protected servo feeds): [power/README.md](power/README.md) and
  [power-distribution.md](power-distribution.md).
- BMS: buy an off-the-shelf 3S board if one fits the spine, else design it in.

### 2. Spinal-cord board (chest; first estimate ~50 x 70 mm, ~30 g)

- Real-time controller between the Jetson (USB) and the servos: Teensy 4.1 plugged in, or an RP2350 on the board.
  Holds his pose and fails safe if the Jetson stops.
- 4 half-duplex servo buses at 1 Mbps (74HC241-type drivers): legs (STS protocol), upper body (STS), left hand and
  right hand (6 V; SCS0009 uses the older SCS byte order).
- IMU (LSM6DSOX / BMI088 class).
- 4 x MPR121 touch chips on one I2C bus = 48 copper-tape pads inside the shell (layout in the BOM's touch row).
- I2C out to the foot boards, the BME688 room sensor and the desk-edge sensors.

### 3. Foot boards (one per foot; ~25 x 40 mm, ~5 g each)

- 4 load cells per foot (full bridges, ~40 kg, at the sole corners) into a 24-bit bridge ADC (ADS1220 / HX711 class).
- One down-looking time-of-flight sensor under the toe (VL53L4CD or VL6180X) for desk edges.
- One thin cable up the leg to the spinal-cord board. Reading the tiny bridge signals inside the foot keeps the noise
  off the long wire.

### 4. Head mic board (the diamond-shaped panel on top of the head; ~10 g)

- XMOS XVF3800 voice chip + 4 MEMS mics placed in the diamond's shape (beamforming, echo cancelling past his own
  speakers, voice direction). The round reSpeaker XVF3800 board does not fit the diamond.
- Audio to the Jetson over USB (through the internal USB hub) or I2S.
- **Superseded, 2026-10-25 afternoon:** front/back 62 mm in head plan view, left/right 100 mm, front about 12 mm
  higher (11.016-degree plane, 63.164 mm along it). A fit check of the real shell (`tools/body/mic_diamond.py`:
  rays onto the head mesh, 2 mm wall, 4 x 3 mm IM69D130 package) found the front and back packages touching the
  inside of the wall with no board edge left. [pcb/head/microphone-geometry.md](pcb/head/microphone-geometry.md)
  A0.2 still carries these old points.
- **Frozen mic locations (owner, 2026-10-25):** the front and back mics moved 2 mm toward the centre each. The
  current head markers (head frame, mm: x = robot left, y = robot back, z = up; acoustic openings on the outer
  surface, +0.5 mm marker offset as before):

  | Mic | x | y | z |
  |---|---:|---:|---:|
  | front | 0.0 | -29.0 | 43.50 |
  | back | 0.0 | 29.0 | 31.85 |
  | robot left | 50.0 | 0.0 | 36.99 |
  | robot right | -50.0 | 0.0 | 36.99 |

  Plan view front/back 58 mm, along the plane 59.2 mm; left/right 100 mm. Best-fit plane z = 37.333 - 0.20089 y,
  11.4 degrees front-up, every mic within 0.34 mm of it. Fit check results: a diamond board whose tips sit 3 mm
  past each mic centre clears the inside of the wall by 1 mm front and back and 4 mm at the sides. Between the
  mic plane and the outer surface the shell rises up to 0.24 mm above the plane and dips up to 1.22 mm below it
  (at the side tips, where the top rounds off). The board slab (2 mm wall + 0.5 mm gasket + 1.6 mm board + 1.1 mm mic
  under it) sits 3.7 to 6.4 mm below the mic plane and hits nothing in the head (screen panel, its driver board,
  ear speakers, cameras). The nearest part is the panel's top edge, about 1 mm below the front tip.
  Source: `.run/results/mind/body-v2/CURRENT.xml` as of this change. The earlier snapshot hashes no longer match.
  Board perimeter, port offsets, gaskets and mounts remain engineering work.
- The GNSS antenna for the LTE module's GPS can ride on this board (high, under plastic).

## Keep-outs that touch the boards

- No copper (touch pads, ground pours) over the seat charging coil, over the screen, or near the LTE / GNSS / Wi-Fi
  antennas.
- Battery sits below the Jetson (heat rises); the power board must not add heat to the battery segments.
