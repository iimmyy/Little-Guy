# Alpha 1S: first continuity checks

Prepared 20 October 2026. **No measurements have been taken.**

These seven readings check the probe contact, one connection already traced in the photographs, and five possible ground connections. The results will be reviewed before adding connections to the schematic.

Disconnect the battery, charger, USB and external cables before probing. Use continuity/resistance mode, never current mode. If a probe cannot touch one marked terminal without bridging another, skip that reading. Do not remove components.

Match the board orientation to the pictures. The small ring marks the metal contact; the letter sits beside it. Record the displayed resistance, including units, and whether the meter beeps. Also note OL, drifting readings, or readings that change when the probes are reversed. A beep alone does not establish a direct copper connection.

## Probe locations

![U14 and nearby capacitor probe locations A through E](continuity-u14.png)

| Point | Metal contact |
|---|---|
| A | U14 pin 2: middle lead on the three-lead side |
| B | U14 pin 5: upper lead on the two-lead side |
| C | C83 upper terminal: schematic pin 1 |
| D | C82 upper terminal: schematic pin 2 |
| E | C83 lower terminal: schematic pin 2 |

![U13 probe location F](continuity-u13.png)

**F:** U13 pin 2, middle lead on the three-lead side.

![U12 probe location G](continuity-u12.png)

**G:** U12 pin 8, uppermost right lead in this photograph.

![U4 probe location H](continuity-u4.png)

**H:** U4 pin 4, lowermost left lead in this photograph.

## Record the readings

Start with C00 and C01. If C01 does not read close to the probe baseline C00, stop and report those two readings before continuing. For C02 through C06, keep one probe on A and move the other probe to the indicated point.

| ID | Probe points | Purpose | Resistance | Beep / other observation |
|---|---|---|---|---|
| C00 | Touch the two probe tips together | Probe/contact baseline | | |
| C01 | B to C | Check the short connection already traced in the photographs | | |
| C02 | A to D | Test whether C82's opposite terminal joins U14 ground | | |
| C03 | A to E | Test whether C83's opposite terminal joins U14 ground | | |
| C04 | A to F | Test whether the two regulator ground pins connect | | |
| C05 | A to G | Test amplifier signal ground against regulator ground | | |
| C06 | A to H | Test DAC ground against regulator ground | | |

You can return the readings as seven lines of text or fill in the [blank CSV](continuity-first-pass.csv). Keep the resistance units and observations with each reading.

Pin functions come from the identified components' datasheets. Calling a pin ground does not establish that it connects to the other ground pins on this donor. The photographs locate probe targets; they are not physical dimension measurements.

Source photograph: [back-scaled.png](../reference/back-scaled.png). The eight target locations were visually reviewed on the annotated PNGs. The optional [HTML worksheet](continuity-first-pass.html) has not been browser-tested; the images and table above are the reviewed handoff.
