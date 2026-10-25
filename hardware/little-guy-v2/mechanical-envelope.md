# Preliminary mechanical envelope — 2026-10-21

Status: manufacturer references inspected; assumed load cases calculated. No robot geometry selected, mount fit verified, print released or loaded motion tested. The approximately 55 cm overall height and clothing size do not determine shoulder spacing or limb lengths.

## Servo reference geometry

Waveshare provides [2D and STEP archives](https://docs.waveshare.com/ST3215_Servo/Resources-And-Documents), retained under `reference/` with source URLs and SHA-256 digests in [mechanical-sources.json](reference/mechanical-sources.json). The PDF was rendered and visually inspected. Its title is **SCS215**, although the archive and files are named ST3215. Preserve that discrepancy; do not treat the downloaded drawing as confirmed fit for a purchased servo.

| Dimension read from the supplied drawing | mm | Use now |
|---|---:|---|
| Long body envelope | 45.22 | Preliminary packaging |
| Body width | 24.72 | Preliminary packaging |
| Bare main housing span in side view | 29 | Reference only |
| Side-view span across outer case features | 35 | Preliminary packaging |
| Overall span including illustrated round output fittings | 37.25 | Preliminary clearance; actual fittings must match |
| Round fitting diameter | 19.2 | Reference only |
| Opposite illustrated fitting-hole centres | 14 | Reference only; hole diameter/thread unresolved |
| Output-axis distance from nearby short end | 10.11 | Preliminary axis placement |

The [FEETECH product listing](https://www.feetechrc.com/products.html?keyword=STS3215) identifies the 12 V version as ST-3215-C018, gives dimensions 45.2 × 24.7 × 35 mm and mass **55 ± 1 g**. This supports a preliminary mass model and broadly agrees with the main package envelope; it does not resolve the drawing title, accessory fit, cable exits, screw engagement or tolerances. The STEP has now been imported and compared with the newer model-specific C018 drawing. The full nominal reference envelope is 45.23 × 24.73 × 37.8 mm; this includes the older model's accessories and is not a fit tolerance. One of eight solids fails CAD validity. See [servo-interface.md](servo-interface.md) for the actual output-axis datum, accessory differences and screw callouts. Do not use model volume as a measured mass.

## Arm load calculation

[arm-scenarios.json](analysis/arm-scenarios.json) holds editable assumptions; [arm_loads.py](analysis/arm_loads.py) calculates [arm-results.json](analysis/arm-results.json). Reproduce from the repository root:

```sh
python3 hardware/little-guy-v2/analysis/arm_loads.py
```

Each arm has two shoulder servos, an elbow servo and a wrist servo. In this model the proximal shoulder servo is fixed to the torso. Its load includes the second shoulder servo, both distal servos, all moving structure and any hand payload. The elbow servo body is mounted upstream of the elbow axis; its own body is therefore not included in the elbow's moving load. Recalculate if the actual bracket arrangement reverses that mounting. Both shoulder axes still need separate analysis once their geometry exists.

The arm is straight and horizontal, with gravity acting at each component's centre of mass. Structure masses include printed pieces, brackets, horns, fasteners, cables and clothing; they exclude the separately counted servo bodies. Structure is approximated by uniform slender segments and servos by point masses. Each row is a separate acceleration case: the joint under evaluation accelerates at 2 rad/s², its upstream joint remains fixed and its downstream joints remain locked. It is added in the direction that increases required torque. The rows do not bound simultaneous motion or give the elbow holding load while the shoulder accelerates the arm.

`gravity torque = Σ(m × g × horizontal lever arm)`

`inertia = Σ[m × (distance² + segment length² / 12)]`

`screening torque = gravity torque + inertia × angular acceleration`

One kgf·cm equals 0.0980665 Nm. Published rated torque is 10 kgf·cm at 12 V; the 30 kgf·cm stall figure is not the working target. A **5 kgf·cm screening allowance** is an analyst-selected half-rating target, not manufacturer derating, a validated safety factor or a continuous-duty guarantee. Keep it explicit so a calculation cannot silently turn it into a tested limit.

| Assumed input per arm | Compact/light | Longer/heavier |
|---|---:|---:|
| Shoulder–elbow / elbow–wrist / hand length | 100 / 100 / 60 mm | 120 / 120 / 70 mm |
| Upper / forearm / hand structure mass | 80 / 60 / 40 g | 115 / 90 / 55 g |
| Carried second shoulder servo offset | 25 mm | 30 mm |
| Each servo body | 55 g | 55 g |
| Total mass carried by proximal shoulder, no payload | 345 g | 425 g |

| Calculated result, no payload | Compact/light | Longer/heavier |
|---|---:|---:|
| Proximal shoulder gravity torque | 4.01 kgf·cm | 5.97 kgf·cm |
| Proximal shoulder example acceleration torque | 0.13 kgf·cm | 0.24 kgf·cm |
| Proximal shoulder combined | **4.14 kgf·cm** | **6.21 kgf·cm** |
| Elbow combined | 1.40 kgf·cm | 2.10 kgf·cm |
| Extra tip mass before reaching the chosen shoulder allowance | about 31 g | Already exceeds allowance empty |

The compact example is a candidate for further layout, not an accepted arm. The 31 g figure is a mathematical residual at the hand tip under these assumptions, not a payload specification. It is easily consumed by an underestimated bracket, heavier clothing or different motion. The longer example demonstrates why scaling the shell without controlling its mass can exhaust the shoulder budget. A negative signed tip margin is preserved in the JSON rather than hidden as zero.

The model omits gearbox friction, servo-body rotational inertia about their centres, cross-axis offsets, coupled joint dynamics, shock, cable drag, garment snagging and contact forces. Neither numerical case establishes thermal duty, bearing capacity, collision clearance or structural strength. Check each mass budget against actual parts and weighed prototypes; rerun with measured centres of mass and the actual axes. Software raw acceleration settings are not yet mapped to physical rad/s² and cannot implement this scenario directly.

Independent review reproduced all four torque totals without calling the calculator. It identified the isolated-acceleration distinction above: with the shoulder accelerating and elbow locked, the elbow also reacts to translation of its axis. In this simplified planar model its dynamic term becomes `alpha × (I_elbow + upper_length × sum(distal_mass × elbow_radius))`, giving elbow totals about 1.43 and 2.16 kgf·cm. This confirms why the separate-row results must not be presented as a coupled robot simulation.

## Mechanical layout requirements before releasing mounts

- Keep the first shoulder motor supported by the torso. Place carried motors and heavy fittings close to the corresponding axes where practical. Reserve the mass of clothes and moving cables from the beginning.
- Provide a structural load path for shoulder and elbow bending. Torque capacity alone does not establish output bearing or horn capacity. An opposed support/bearing is a candidate, but its fit, alignment, shaft support and fasteners need actual design.
- Head pitch needs head/speaker/camera/cable mass and centre of mass. For scale only, 0.5 kg at a 50 mm horizontal offset consumes 2.5 kgf·cm statically. Keep the user-selected speaker dimensions and mass unresolved until supplied.
- Vertical waist yaw has little gravity drive torque in a level body, but still carries the upper body's axial/bending loads and accelerates its inertia. Use a supporting torso bearing/load path rather than assuming the servo output alone can carry the torso.
- A stand must resist the overturning moment from extended arms and moving torso. Stand footprint, ballast, total centre of mass and mounting strength remain unresolved; no free-standing stability claim follows from an arm torque calculation.
- Reserve cable travel, strain relief, connector access and hard-stop clearances at every joint. Clothing can restrict motion. Confirm the wrist axis and actual usable range before converting the proposed joint names into CAD.

The next geometry input is a dimensioned body layout: shoulder spacing, neck axis, torso height, limb lengths, carried electronics/speaker locations and stand attachment. These inputs and actual servo fit control printable brackets; approximate robot height alone is insufficient.
