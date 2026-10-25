# Little Guy V2, full robot parts list (draft 1, 2026-10-25)

Step 1 of the full build plan: every part that goes inside him, before any learned motion (RL) work. The robot has to
learn with its real body, so every sensor and part is chosen, sized and weighed first, then placed in the simulation.

Each line gets: what it is, the pick, how many, size, weight, power, how it connects, price, source. Status:
**decided** = Immy picked it; **candidate** = my pick, not yet confirmed; **open** = not researched yet.
Prices only from live listings with the date checked; older prices are marked with their date and get re-checked
before ordering.

This replaces the phase-1 list ([phase1-parts.md](phase1-parts.md), upper body on a stand), which predates the
walking 60 cm body.

## 1. Moving: servos and their bus (41 servos, counted from the simulation model)

| Part | Status | Qty | Where | Notes |
|---|---|---|---|---|
| Feetech STS3250, 50 kg.cm stall, 75 rpm | decided | 12 | hips (yaw, roll, pitch), knees, ankles (2 per foot, rods) | price not checked live yet |
| Feetech ST3215 12 V, 30 kg.cm stall | decided | 13 | waist 2, shoulders 6, elbows 2, neck turn 1, neck nod+tilt 2 | US$21.99 each at Waveshare (checked 2026-10-20) |
| Feetech STS3032, 4.5 kg.cm | decided | 2 | forearm twist | price not checked |
| Feetech SCS0009, 2.3 kg.cm, 11-13 g | decided | 14 | wrists 4 (rods), fingers 10 (spools) | US$9 each at Seeed (checked 2026-10-25) |
| Spinal cord: real-time controller between the Jetson and the servos | candidate | 1 | chest/back | Teensy 4.1 (600 MHz Cortex-M7, 8 hardware serial ports; the reference board for on-MCU half-duplex servo loops, github.com/molanocortes/dynamixel-on-device) or an RP2350 on our own PCB (RP2350 servo bridges with a built-in IMU exist: link101, Axon). Holds his pose and fails safe if the Jetson stops; reads the IMU and, later, foot/touch/edge sensors. Talks to the Jetson over USB. Price not checked |
| Servo buses: 4, one per body region | candidate | 4 | - | legs 12 x STS3250 (12 V); upper body 13 x ST3215 (12 V); left and right forearm 1 x STS3032 + 7 x SCS0009 each (6 V). The 6 V servos can't share a 12 V line, and the SCS0009 speaks the older SCS byte order, so the hands get their own buses. Several buses in parallel is how RoboCup humanoids get fast loops (Bestmann et al., bit-bots.de dxl_paper); one sync write + one sync read for 12-13 servos at 1 Mbps is a few ms, so 100 Hz or more per bus. Half-duplex line driver per bus (74HC241-type) |
| Servo cables, horns, bearings, rods, ball ends, finger wire, spools | open | - | everywhere | lengths come from CAD |
| Feetech servo programmer (FE-URT-1, USB to servo bus) | candidate | 1 | bench tool | sets each servo's ID before it joins a shared bus and tests servos from the Mac; was on the old Notion shortlist (C$20.34 at RobotShop, Aug 2026, re-check) |

## 2. Power

| Part | Status | Qty | Notes |
|---|---|---|---|
| Battery pack | open (exact cell, BMS) | 1 | FIT DONE 2026-10-25 (`tools/body/insides.py`): sized on Molicel P45B 21700 cells (21.55 x 70.15 mm, 70 g max, datasheet v1.2); all three 2-cell sections fit stacked low in the chest, below the Jetson, after the chest was made 2 cm deeper (Immy: "2cm is fine"); as drawn before only 1 of 3 fit. ~155 g per section with holder and nickel strip, ~465 g pack. The pelvis has no free room (waist and hip servos, waist gears), so the spine stays in the chest. Immy's layout idea (2026-10-25): a BATTERY SPINE, cells stacked vertically from the bottom of the pelvis up to the neck like vertebrae, a protective sleeve that doesn't trap heat, and the wiring running up and down it like the spinal cord. Cells can't bend, so it splits into segments (pelvis, chest) joined by flexible cable through a hollow waist, the way discs sit between vertebrae. Fit and heat get checked in step 3 |
| Battery chemistry | decided | - | Round lithium-ion cells, 21700 size, 3S2P = 6 cells, ~108 Wh, ~0.42 kg (Immy, 2026-10-25, changed from LiFePO4 after seeing it adds ~1 kg: "we'll go with the round cells. As long as they don't puff up"; shorter life is fine, "by 500 to 1,000 charges, we'll have ... better robots"). Steel cans (no puffing up like a pouch); 3 in series top out at 12.6 V = the ST3215's max, so the 12 V servos run straight off it. No chemistry mix: two chemistries can't share one pack. Kept cool instead: placed BELOW the Jetson (heat rises), own temperature sensor, BMS charges in a safe temperature range and caps everyday charge ~90%. Layout: the battery spine, 3 vertebrae of 2 cells each (one per series group), all three stacked in the lower chest (fit 2026-10-25, see Chest insides in section 7) |
| Battery protection (BMS) + fall-safe enclosure | open | 1 | design rule: a fall must never make the battery dangerous. Built in, not removable (Immy, 2026-10-25, overriding the Aug planner's removable-pack rule: "as long as I can plug him in or he can charge on the wireless station, that's all that matters"); cells still reachable through the back for replacement at end of life |
| Battery fuel gauge | open | 1 | how full he is = his "hunger" (computer-body idea) |
| Charging: USB-C port on the front door, charger chip | decided (Immy, 2026-10-25: "wireless charging creates a lot of heat, so sometimes I just want to be able to plug him in ... type c") | 1 | a full-speed charging path, equal to the seat dock, not a trickle backup. USB-C Power Delivery up to 20 V; a 45-65 W laptop charger fills the ~110 Wh pack in about 2-2.5 h. Wired runs cooler than the coil (~95% vs ~90% efficient, and no warm coil in his pelvis). Charger chip open: must take USB-C PD in and charge 3 cells in series. Door already has the port and the power button |
| Seat charging coil in the pelvis + dock | decided | 1 | see Charging below (feet coils dropped). Backup from the old planner (Aug 29): hidden pogo-pin contacts in the seat, lighter, cooler and more efficient than a coil, if the coil runs warm |
| Power board: fuses per servo group, 12 V to the servos, 5 V for screen/USB, Jetson feed, emergency cut | open | 1 | our own PCB; the old harness study is the starting point ([power/](power/README.md)). Old planner rule kept: a separate high-current motor rail and clean regulated rails for the brain and sensors, sized for motor current spikes, not just averages |
| Lit power button | decided | 1 | on the front door |

### Power budget, servos + Jetson only (2026-10-25, Immy: "the servos and the Jetson to power it for 2 hours ... before
we even involve the speakers and the mics")

How it's worked out: every one of these servos pulls current in step with how hard it pushes (its torque constant,
from the maker's datasheet). The simulation says how hard each of the 41 servos pushes in a pose; that gives amps,
times volts gives watts. Each servo also draws a little just for being switched on (idle current).

| Servo | Volts | Idle | Amps per push | Source |
|---|---|---|---|---|
| STS3250 x12 | 12 V | 24 mA | 1 A per 11 kg.cm (stall 4.2 A) | Feetech spec sheet 2024-01-16 |
| ST3215 x13 | 12 V | 30 mA | 1 A per 11 kg.cm (stall 2.7 A) | Waveshare docs, DFRobot |
| STS3032 x2 | 6 V | 6 mA | 1 A per 3.5 kg.cm (stall 1.2 A) | Feetech spec sheet 2020-06-08 |
| SCS0009 x14 | 6 V | 6 mA | 1 A per 2.3 kg.cm (stall 1.0 A) | Feetech spec sheet 2020-11-23 |

Servos, measured in the simulation (CURRENT.xml, real servo limits; .run/results/mind/body-v2/power-1025.txt):
- all 41 switched on, nothing pushing: 8.7 W (just their electronics)
- standing still, holding his pose: 12.7 W (straight legs need very little push)
- standing and talking with his hands: 23.9 W
- walking: not simulated yet (no gait). ToddlerBot (3.4 kg) estimates 50-60 W of motor power walking; scaled to
  his 4.4 kg that is roughly 65-80 W (estimate)
- sitting relaxed: no seated pose yet; servos mostly resting, so close to the 8.7 W idle (estimate ~10 W)
- rerun on the real-insides body (4.25 kg, 2026-10-25, .run/results/mind/body-v2/retest-1025/): standing 12.8 W,
  talking with his hands 23.7 W, idle 8.7 W: unchanged

Jetson Orin Nano Super, measured by others: idle 4.4-7 W; 15 W mode thinking 13-15 W; 25 W mode thinking 20-28 W
(specpicks.com and edgeaistack.ai wall-plug measurements, 2026). Little Guy listens most of the time and thinks in
bursts, so ~15 W average is the working number, 25 W the heavy case.

Plus ~10% lost turning battery voltage into 12 V / 6 V / Jetson power (typical step-down converter).

| 2 hours of... | Servos | Jetson | Total with losses | Energy for 2 h |
|---|---|---|---|---|
| sitting with Immy (café) | ~10 W | 15 W | ~28 W | ~55 Wh |
| standing, talking with his hands | 24 W | 15 W | ~43 W | ~86 Wh |
| mixed: 1 h 45 sitting/standing + 15 min walking | | | | ~70-75 Wh |
| walking the whole time | ~70 W | 15 W | ~94 W | ~190 Wh |

Rough battery target: about 75-90 Wh. For scale, ToddlerBot's own pack is 4 x 21700 cells, 74 Wh, 330 g, and lasts
about 2 hours of on-and-off use (their paper). Airlines allow up to 100 Wh in carry-on without approval, so staying
under 100 Wh keeps him flyable. Battery gets chosen last, after everything else is in.

### Everything else switched on, and warming up (2026-10-25, Immy: "with our battery and speakers and all that ... as
he works, more and more heat builds up ... after an hour, he may be warm")

| Part | Draw | Source |
|---|---|---|
| Amp TAS5805M, idle | ~0.3 W | TI datasheet: 16.5 mA at 13.5 V (hybrid mode) + 18 mA at 3.3 V |
| Music through ears + woofer | ~2 W normal, ~5 W loud (average) | estimate: music averages about a tenth of its peaks |
| Mic processor XVF3800 | ~0.4-0.5 W | XMOS datasheet: 345-400 mW core |
| Screen + HDMI board | ~1.5-2 W | estimate: face is mostly black (AMOLED black = off); full white panel ~2.8 W (BOE 5.5" spec), LT6911-class board max 4 W |
| 2 cameras | ~0.5 W | estimate |
| NVMe SSD | ~0.5-1 W | estimate (idle low, bursts 2-4 W) |
| Fan | ~0.5-1 W | estimate |
| Spinal-cord controller | ~0.5 W | estimate |
| Wi-Fi / LTE (Quectel EG25-G class) | ~0.5-1 W | LTE idle 22 mA at 3.8 V (Quectel spec); data bursts more |
| Sensors (IMU, touch, feet, edge) | ~0.1 W | estimate |
| **All extras** | **~8 W** (11 W with loud music) | |

Warming up over an hour: the servos barely heat while sitting or gesturing (the hardest-working one pushes 0.15 Nm,
about 0.02 W of motor heat; their electronics add ~0.3 W each). The warm part is the chest: Jetson + amp + boards
~20 W. Hot chips leak a little more and the fan spins up, so add ~5% sitting; walking heats the leg servos for real
(hotter copper, weaker magnets), so add ~10% there. Walking is limited by heat before battery: ToddlerBot's walking
lasted 19 minutes before hot motors made it fall.

| 2 hours of... | Servos | Jetson | Extras | x1.1 converters x warm-up | Energy for 2 h |
|---|---|---|---|---|---|
| sitting with Immy (café) | ~10 W | 15 W | 8 W | ~38 W | ~76 Wh |
| standing, talking with his hands | 24 W | 15 W | 8 W | ~54 W | ~109 Wh |
| mixed: 1 h 45 sitting + 15 min walking | | | | | ~95 Wh |
| loud music on top | | | +3 W | | +6 Wh |

Updated target: ~90-110 Wh, right at the 100 Wh carry-on line. DECIDED (Immy, 2026-10-25): flying doesn't matter ("I haven't flown anywhere in years"), so size for the full 2 hours, ~110 Wh, even if that passes 100 Wh. Biggest lever found: switch the leg servos' power off
while he sits (the power board cuts that bus): saves their ~3.5 W of idle draw, ~7 Wh over 2 h.

## 3. Brain

| Part | Status | Notes |
|---|---|---|
| Jetson Orin Nano Super module (8 GB) | decided | from the dev kit (bought, arriving Oct 28 to Nov 6) |
| Compact carrier board | candidate | Forecr DSBOARD-ORNXS, 61 x 91 mm, 43 g, 9-36 V in (user manual rev 2.0). Has: 2x CSI 4-lane or 4x CSI 2-lane (up to 4 cameras), M.2 Key-M 2230 (SSD, PCIe x2), M.2 Key-E (Wi-Fi/BT), I2S, 2x UART, 2x I2C, SPI, CAN-FD, 5 V fan header, mini DisplayPort, temperature sensor. Only 2x USB 2.0: a small USB hub is decided (Immy) for the mic chip, LTE modem and spinal cord. Price not checked |
| NVMe SSD, M.2 2230 | Immy supplies it | the carrier only takes the short 2230 size; memory never forgets (notebook + RAG on disk). Size/price open |
| Heatsink + fan | open | exhaust through the upper-back grille, handles and underside vent; carrier has a 5 V PWM fan header with speed readback (= his "breathing") |
| Wi-Fi + Bluetooth card, M.2 Key-E | candidate | reuse the card that comes in the dev kit if there is one (check on arrival) |

## 4. Senses

| Part | Status | Notes |
|---|---|---|
| IMU (gyro + accelerometer) | candidate | on the spinal-cord board in the chest; LSM6DSOX / BMI088 class (the RP2350 robot bridges link101 and Axon use LSM6DSOX); already modelled in the sim (balance.py) |
| Cameras | decided (Immy: "some LiDAR cameras, some normal cameras") | 2x IMX219 (Raspberry Pi Camera v2 sensor) with wide lenses (77-160 deg versions exist) in the frame above the screen, as a stereo pair = depth like two eyes (proven on Orin Nano: Lenna stereo project). The carrier takes up to 4 CSI cameras, so 2 more are possible later. How many and where = Immy ("tons of cameras") |
| Microphones | decided (Immy: voice chip, echo + noise cancelling) | XMOS XVF3800 voice chip on our own small board + 4 MEMS mics in the diamond on top of the head: beamforming, echo cancelling (it hears past his own speakers), direction of a voice. 345-400 mW (XMOS datasheet). The round reSpeaker XVF3800 board doesn't fit the diamond |
| Touch | decided (Immy: "he should be able to feel things") | copper-tape pads inside the printed shell (embedded ~0.6 mm under the surface, or stuck behind 2-3 mm walls) read by MPR121 chips (12 pads each); proven on social robots (USC Blossom used MPR121 + Teensy 4.1) and pet robots through 3D-printed heads. Which places = Immy |
| Touch layout | decided: 48 (Immy left the count to Claude: "if it's not a hassle to install, go for it"; 48 = one I2C bus, past it every extra pad means a second bus and more wires to solder) | 48 pads = 4 MPR121 chips, the most one I2C bus holds (the chip has 4 addresses). Head 5 (top front, top back, left, right, back of head), chest 3, back 3, belly 2, butt 1, hip sides 2, and each side: shoulder 1, upper arm 2, forearm 2, hand 3 (palm, back, fingertips), thigh 3, knee 1, shin 2, top of foot 2. Pads sit INSIDE: copper tape stuck behind the 2-3 mm shell wall (or laid in during a paused print), invisible from outside; through-plastic sensing is proven (Blossom, pet robots). Weight (calculated, not weighed): 1-oz copper tape ~3.1 g per 100 cm2 copper + ~0.3 g adhesive; ~2000 cm2 covered = ~70 g solid, ~35 g with hatched (mesh-pattern) pads that sense the same for hand-sized touches; wires 48 x ~0.3 m x ~0.6 g/m (30 AWG) = ~9 g; 4 breakout boards ~8 g; total ~50-90 g, about one servo. Keep-outs: no copper over the butt charging coil (copper in the coil's field heats and blocks charging), over the screen, or near the LTE/GNSS/Wi-Fi antennas (copper detunes them). Superseded drafts: 71 pads (~1 per 40 cm2, .run/results/mind/body-v2/touch-area-1025.txt), then 21 big pads |
| Foot pressure | decided | 4 small steel load cells per foot (full Wheatstone bridge, ~40 kg rated) at the sole corners, like Rhoban's RoboCup champions (64 cm, 5.5 kg robot: same size class). They switched from glued gauges because those broke. Gives where his weight sits on each foot (centre of pressure) for balance |
| Desk-edge sensors | decided | ST VL53L4CD or VL6180X time-of-flight sensors under each toe, looking down (laser timing, so dark or shiny desks don't fool it, unlike IR). VL6180X also measures room light |
| Depth / LiDAR | decided (need) / candidate (parts) | two layers: (1) Arducam ToF camera in the upper chest, 240x180 depth points, 0-4 m, 70 deg, 38 x 38 mm, 50 g, CSI 2-lane, Orin Nano tested (Arducam docs); 3.5 W max, so it runs at a low frame rate unless he's exploring. (2) ST VL53L8CX multizone ToF (8x8 zones, ~4 m, a few mm big) in the band above the screen: coarse depth right in front of his face where the 38 mm board can't fit. Camera budget: 2 IMX219 + 1 ToF = 3 of the carrier's 4 CSI slots |
| GPS | decided (Immy) | the LTE module's built-in GNSS (Quectel EG25-G: GPS/GLONASS/BeiDou/Galileo/QZSS); needs its own small antenna high up (top of head or shoulders) under plastic, not metal |
| Room temperature + humidity | decided (Immy) | Bosch BME688-class: temperature, humidity, air pressure (weather, which floor he's on) and a gas sensor (rough sense of smell / air quality). Placed at a low side intake, away from the Jetson's warm exhaust |
| Battery temperature | candidate | thermistor on each battery vertebra, read by the BMS and the spinal cord |
| Servo heat, effort, voltage, current | comes with the servos | every Feetech servo reports these = his body sense |
| Finger squeeze | comes with SCS0009 | load reading on each finger servo |

## Charging

Owner decision, 2026-10-25: charging supports both sleep and awake conversation, with sleep preferred by default.
Keep charging supervision and button wake available while the Jetson sleeps or is off. Awake charging will be slower;
the power board must budget the running load before assigning battery charge current. Initial electrical checks are
in [the power-board first pass](pcb/power/README.md).

| Option | Notes |
|---|---|
| Wireless pads in the feet (Immy's first idea, 2026-10-25) | coil + ferrite in each sole adds weight at the very end of the leg (the costliest place for swing weight); Qi-class pads are ~15 W each |
| Contact pads under the feet (robot-vacuum style) | grams, most efficient, but bare metal on his soles |
| Charging through his seat when he sits on his dock | **DECIDED (Immy, 2026-10-25)**: his dock is a little stool; "if we can teach him to literally just turn around and sit on it, then he can just charge through his butt". Until he learns to sit, Immy sets him on it (carry handles). No coils in the feet, so the feet stay light. USB-C on the front door is the other full charging path (Immy: plug him in when he wants it cool) |
| (row kept for reference) Charging through his seat | wireless coil in the pelvis: weight in the trunk, sitting is already a core skill, and leg power is off while seated. Robot wireless docks (WPC Ki) run ~90% efficient aligned; a dock is a fixed spot, so alignment is easy |

## 5. Face and sound

| Part | Status | Notes |
|---|---|---|
| Screen EDO E555HBM2 5.55" AMOLED + HDMI board | decided | C$87.68 combo (AliExpress, 2026-10-25); 26 g panel |
| DisplayPort to HDMI (carrier has DP) | open | |
| Ear speakers, 2 x 40 mm full-range | decided (size) | driver not picked; Immy browses AliExpress, I check size/depth/specs |
| Woofer, 2" long-throw, sealed chamber, lower back | decided (size) | driver not picked |
| Amplifier | candidate | TI TAS5805M (2 x 23 W, I2S, built-in EQ); a 2.1 system needs a third channel: open |

## 6. Talking to the world

| Part | Status | Notes |
|---|---|---|
| Wi-Fi + Bluetooth card | open | depends on the carrier's M.2 slots |
| LTE modem + SIM | decided (need) | for going out with Immy; part open |
| Antennas | open | inside the shell, away from metal and the battery |

## 7. Body shell and mechanics

| Part | Status | Notes |
|---|---|---|
| Printer: Bambu Lab P2S | decided, not bought | about C$1,200 (2026-10-21), re-check before buying |
| Filament: PETG-CF shells, PAHT-CF near heat, TPU rubber parts | decided (plan) | rubber: ear rings, screen rim, chest bumper |
| Screws, heat-set inserts, bearings | open | from CAD |
| Neck ball joint + push rods, neck slip clutch | open | clutch not designed yet |
| Carry handles (fold, detent), front door + latch | handles decided; latch open | |
| Getting-up shape | open (CAD rule) | from the old planner's self-righting requirement: no shell shape that traps him flat on his back; rounded contact surfaces that let him rock into a pose his legs can push up from |
| Touchable-surface temperature | open (CAD rule) | from the old planner: no metal radiator where hands grab him; conservative comfort limit on the shell, measured on the built body |

### Chest insides and whole-robot weight (2026-10-25, Immy: "start to wire in things that already exist, like the Jetson")

The chest used to weigh ToddlerBot's chest scaled up (1341 g), which already counted THEIR computer, battery, boards
and shell. It is now built from our own parts (`tools/body/insides.py`): the printed shell and frame, every bought
part that is picked as a box with its maker's size and weight, and a kept space for every board not made or picked
yet. A fit search places each part inside the shell, clear of the shoulder and neck servos and the woofer, and keeps
the battery below the Jetson (heat rises). The chest is 2 cm deeper front to back than before (1 cm each way; Immy:
"2cm is fine"): the door, power button and charge port moved forward with the front wall, the grilles, handles and
woofer back with the back wall (`tools/body/deepen.py`); shoulders, neck and arms did not move.

| Part | Grams | Where (his frame: front / left / up from the waist) | Source |
|---|---:|---|---|
| Printed shell, 2 mm PETG-CF | 228 | 912 cm2, measured off the shell's own shape | calculated (1.25 g/cm3) |
| Inner frame, screws + inserts, wiring | 120 + 25 + 50 | battery cage, Jetson mount, brackets, door hinge, ribs, harness | estimates |
| Jetson stack (carrier, module, active heatsink, SSD, Wi-Fi) | 148 | high on the back wall, standing up, fan out through the exhaust grille | maker numbers + ~6 g estimate |
| Battery, 3 sections of 2 cells | 465 | low in the middle, stacked like drawers, cells lying across his body | Molicel datasheet + ~15 g holder each |
| Depth camera (Arducam ToF) | 50 | high in the front, facing forward | shipping weight, an upper bound |
| Power board (ours) | 40 | flat on top of the battery | kept space, estimate |
| Spinal-cord board (ours) | 30 | middle, standing up in front of the Jetson | kept space, estimate |
| Amp board (not picked) | 15 | low front, behind the door | kept space, estimate |
| LTE modem (not picked) | 10 | top left, under the neck | kept space, estimate |
| USB hub (not picked) | 10 | right side wall | kept space, estimate |
| **Chest total** | **1191** | (+ woofer 100 g, handles, shoulder and neck servos, counted as before) | |

Whole robot: **4.25 kg** (was 4.40 with the scaled lump). Balance point over the feet: 295 mm high (was 299), on
the centre line. Same tests, same code, old body vs new (.run/results/mind/body-v2/retest-1025/):

| Test | Old body | New body |
|---|---|---|
| Biggest 0.2 s shove he stays up after: side / back / front | 12 / 6 / 10 N | 12 / 6 / 9.5 N |
| Servo power standing / talking with his hands | 12.7 / 23.9 W | 12.8 / 23.7 W |
| 15 N fall, head's hardest stop: back / side / front | 115 / 171 / 125 g | 104 / 162 / 155 g |
| 15 N fall forward, neck nod twist | 5.9 Nm | 1.6 Nm |

The shell is perfectly rigid in the model, so fall peaks compare bodies; they are not real breaking loads.
Still estimates: frame, screws, wiring, and every kept space until its part is picked or its board is laid out.

## 8. Our own circuit boards (PCBs)

Immy asked (2026-10-25): "Do we have to make any PCBs?" Yes: V2 was planned with our own boards from the start
(2026-10-24, "PCB too"). Rule: make a board only where nothing bought fits the space or does the job; buy the rest.
Bring-up order (from the Aug planner's progressive rule): first wire bought breakout boards together on the bench and
prove each job, then fold the proven circuit into our own board. Sizes below are first estimates for placing them in
the sim (step 3); the real outlines come from the CAD shells (step 5). Design in KiCad, made and assembled by a PCB
house (JLCPCB/PCBWay class); prices checked live when ordering.

| Board | Status | Where | Job | First size estimate |
|---|---|---|---|---|
| Power board | [ours: A0.2 schematic draft](pcb/power/README.md); layout/test pending | chest, lying flat on top of the battery stack, under the Jetson (fit 2026-10-25; gap or spacer so it doesn't warm the cells) | battery in; separately switched leg and upper buses (11.28 V nominal draft to leave margin below STS3250 12 V max); hand supplies (5.695 V nominal draft); clean Jetson feed; 5 V peripherals; hardware cut; USB-C PD + 3S charger; wireless input reserved; current/energy meter and three-segment temperature interlock. BMS, wireless receiver and transient protection remain open | ~60 x 80 mm, ~40 g; unverified |
| Spinal-cord board | [ours: A0.2 schematic draft](pcb/spinal/README.md); layout/test pending | chest, middle height, standing up in front of the Jetson (fit 2026-10-25) | socketed Teensy 4.1 selected for capture; four half-duplex TTL interfaces; LSM6DSOX; four MPR121s; sensor mux and foot/room/edge branches; independent watchdog; USB to Jetson. Firmware, socket/header part selection and physical fit remain open | ~50 x 70 mm, ~30 g; unverified |
| Head board | [ours: A0.2 schematic draft](pcb/head/README.md); layout/test pending | the diamond on top of the head | XVF3800 + four IM69D130 PDM mics, QSPI flash, sequenced supplies and USB. Head-A acoustic targets and channel order captured; board perimeter/mounting, package review, firmware and acoustic tuning remain open; GNSS antenna absent from this draft | fits the diamond, ~10 g; unverified |
| Foot boards | [ours: A0.2 schematic draft](pcb/foot/README.md); layout/test pending | one in each foot; identical PCB twice | two ADS122C04 ADCs multiplex four full bridges; separate toe VL53L4CD supply; six-conductor cable. Exact cells, conversion performance and sole mechanics remain open | ~25 x 40 mm, ~5 g each; unverified |
| BMS (cell protection) | buy if one fits the spine, else ours | on the battery spine | over/under-voltage, over-current, cell balance, temperature cut | off-the-shelf 3S boards ~50 x 20 mm |
| Jetson carrier | buy | chest | Forecr DSBOARD-ORNXS (section 3) | 61 x 91 mm, 43 g |
| Screen driver, amp, LTE modem, cameras, depth camera | buy | head / chest | modules on the list above | per part |

## Open questions for Immy, one at a time

Asked in order as each section gets researched; nothing here needs an answer yet.
