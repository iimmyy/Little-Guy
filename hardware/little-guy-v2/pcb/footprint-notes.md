# A0.1 package review

2026-10-25. These are candidate land patterns, before PCB layout. No physical fit, assembly trial or fabrication release exists.

| Part | Captured geometry | Evidence and remaining work |
|---|---|---|
| Teensy 4.1 carrier socket | Local `Teensy41_Socket_GPIO_ID`; two 24-way rows, 2.54 mm pitch, 15.24 mm row spacing; 60.96 x 17.78 mm body | [PJRC dimensions](https://www.pjrc.com/teensy/dimensions.html) and [pin card](https://www.pjrc.com/teensy/card11a_rev4_web.pdf). Named pads match the local symbol. Select socket/header parts; check pin/drill fit, module height, USB cable space and underside clearance. |
| XVF3800 QF60B | Local `XMOS_QF60B_7x7mm_P0.4mm_5Paddle`; 60 perimeter pins plus five separate supply paddles | [XMOS datasheet v3.2.1, Fig. 8.1](https://www.xmos.com/documentation/XM-014888-PC/pdf/xvf3800_datasheet_v3.2.1.pdf). Drawing mirrored from package bottom to PCB top view; core paddles 61 left, 62 bottom, 63 right, 64 top; ground 65 center. Copper and paste geometry remain assembly candidates. |
| MPR121QR2 | Standard `Package_DFN_QFN:UQFN-20_3x3mm_P0.4mm` | [Manufacturer package drawing, pp. 23-24](https://resurgentsemi.com/wp-content/uploads/2018/09/MPR121_rev5-Resurgent.pdf). Exactly 20 perimeter terminals, no central exposed pad. A0.1 removes the invented schematic pin 21. Confirm production sourcing and stencil rules. |
| CDBC540-G, D301 | Standard `Diode_SMD:D_SMC`; cathode pad 1 on `AON_SW`, anode pad 2 on `GND` | [Comchip package/polarity drawing](https://www.comchiptech.com/admin/files/product/QW-BB012%20CDBC520-G~5100-G%20RevB522561.pdf). A0.1 fixes the earlier symbol/footprint polarity mismatch. Electrical and thermal sizing remain unmeasured. |

The XMOS central copper paddle is 3.60 mm square with the package's 0.35 mm pin-1-corner chamfer. Four core
paddles are 3.60 x 0.60 mm with 0.40 mm nominal separation from the center. Perimeter lands use initial
0.60 x 0.20 mm geometry at 0.40 mm pitch. Paste is split into windows, approximately 65% center and 55% core-bar
coverage. These paste choices are designer candidates, not a manufacturer stencil prescription. Review mask
webs, stencil thickness and tolerances with the assembler. Connect each paddle to its supply plane during layout;
agree filled/capped via-in-pad processing or a compatible alternate via/paste strategy before release.

`footprint-audit-A0.json` checks all assigned numbered pad sets against their symbols, the D301 polarity anchor,
MPR121 terminal count, nominal Teensy row/pitch/selected GPIO locations, and XMOS paddle orientation and supply
assignments. KiCad CLI also accepted and rendered both custom footprint files. These checks do not independently
validate every pin function, package dimension, component rating, stencil, thermal path or mechanical fit.

IM69D130 microphone pads remain unassigned: the current direct manufacturer PDF download returned an empty
file, so its land/stencil drawing has not been visually checked locally. The published 0.8 mm acoustic opening is
recorded in the schematic; it is not a substitute for completing that package review. Other missing land patterns
are explicitly listed in the audit and draft BOM. Head-A acoustic target locations are now captured in `head/microphone-geometry.md`; package origins remain open. No robot-simulation placement was added.
