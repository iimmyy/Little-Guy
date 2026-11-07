# Recovery notes - 2026-10-17

The previous work installed the tools and created an empty project. This pass
adds native schematic content and non-plotting photographic references. The
identical-board objective remains incomplete.

## Sources and findings

- The [UBTECH product presentation](https://www.farnell.com/datasheets/2068318.pdf),
  page 19, specifies STM32-F103RDT6 and a 7.4 V / 2200 mAh battery for Alpha 1S.
  These are model-level specifications. The donor MCU was subsequently confirmed
  as STM32F103RDT6 by direct user reading on 2026-10-18. The battery label still
  needs checking.
- The donor label identifies BM77SPP03MC2. The manufacturer-hosted
  [2015 BM77 datasheet](https://ww1.microchip.com/downloads/en/DeviceDoc/BM77%20Data%20Sheet%20v2.1.1%202015-Jun16.pdf)
  was saved in `reference/bm77-datasheet-2015.pdf`. Figure 1-3 and Table 1-1 give
  the 30-pin variant's assignments. RXD is pin 20 and TXD is pin 21. This is a
  module pin map, not evidence of which STM32 pins the donor connects to.
  P12/SCL, P13/SDA, and P36 have Do Not Connect instructions in the table.
- The donor U12 marking reads PAM8003. The
  [Diodes DS36421 datasheet](https://www.diodes.com/datasheet/download/PAM8003.pdf),
  pages 1-2, provides the 16-pin map used in the custom library symbol. Board
  channel use, surrounding passives, and land pattern remain unverified.
- Y1 reads SCTF 8.000. The package/pin configuration has not been established.
- The [Alpha 1S FCC record](https://fccid.io/2AFJH-ALPHA1S),
  FCC ID 2AFJH-ALPHA1S, lists a schematic exhibit as metadata only and records
  long-term confidentiality. It also lists internal photo exhibit 2712710.
  Attempts to retrieve that photo PDF from the public mirrors and FCC endpoint
  failed; its photographs were not inspected. No original schematic was obtained.

## Native model

- Thirteen schematic symbols: U8 MCU; U10 module; U12 amplifier; U3 charger;
  U4 DAC; Q1/Q2 MOSFETs; J2 and J7-J10 connectors; unpopulated J6. The symbol pin
  maps contain 184 numbered pins total, including the charger's exposed ground pad.
- Custom module and amplifier symbols are in `alpha1s.kicad_sym`, registered by
  `sym-lib-table`. Their datasheet URLs are embedded as properties.
- All inter-component wires remain absent pending trace/continuity evidence.
  Connector symbol numbers are provisional, not established donor pin numbers.
- Two locked reference images are embedded in the PCB editor on User.Drawings.
  They are non-plotting references. The initial uncalibrated images were replaced
  with the newer user-confirmed border crops, as described below.
- No placed footprint, copper, or manufacturing output is claimed. A draft
  photo-derived Edge.Cuts outline was subsequently added as described below.
- User caliper readings now establish 68.45 mm overall length, 72.08 mm maximum
  width, and 56.15 mm top width. The switch/USB/charging end is at the top; the
  board widens approximately 15 mm down from that edge, as clarified by the user
  and checked against the front photograph. `geometry.json` records provenance.
- Eight locked segments and four dimensions form a review sketch on
  User.Drawings, separate from Edge.Cuts. The combined shoulder width is
  15.93 mm. A 7.965 mm offset on each side is a centering assumption only;
  individual offsets and corner radii are unresolved. Sharp corners are
  placeholders.

## New border-cropped photographs

The user supplied `IMG_4510.heic` (front) and `IMG_4511 2.heic` (back), then
confirmed they were cropped to the PCB border, straightened and photographed
top-down. Original files and SHA-256 hashes are preserved in `reference/`.
These statements supply the basis for mapping the complete image extents to
the user-reported 72.08 mm width and 68.45 mm length; no additional scale object
is required for this mapping.

Derived images use 2800 x 2659 pixels with independent axis resampling to the
measured board aspect ratio. The live KiCad bounding boxes were read back and
the image scale adjusted to 72.080000 x 68.450257 mm (rounding to the supplied
dimensions). The manifest records the source sizes, transformation, image
hashes and native CAD scale. The images retain ports-at-top orientation. These
are calibrated overall extents; feature coordinates inferred from photographs
are still estimates until checked against the donor.

The back photograph now supplies a draft perimeter on Edge.Cuts at KiCad datum
(340, 220) mm, mirrored into front-view coordinates. Segmentation and boundary
simplification (0.08 mm algorithm tolerance, not an accuracy guarantee) produce
36 connected segments. Below 30 mm from the top, a convex body envelope bridges
non-green mounting-hole rims that otherwise merge with the background mask;
the photographs show a continuous outer edge there. The result retains the
photographed shoulder and corner shape and is normalized to the overall caliper
extents. It is a tracing draft, not a verified milling profile.

H1-H4 are drawing crosses, not hole footprints. Their centers average estimates
from the front and mirrored back photographs, rounded to 0.1 mm. The two views
disagree by 0.506, 0.389, 0.347 and 0.258 mm respectively. These differences are
recorded in `photo-geometry.json`; no drill diameter or plating is inferred.
The earlier dimensioned sketch is preserved separately. KiCad's saved-board
DRC reports zero violations, including no invalid-outline finding. With zero
placed pads and copper, this provides no electrical verification. Native visual
review confirmed the closed perimeter and all four reference markers.

## Checks and next evidence

The project-local `Alpha1S:BM77SPP03MC2_Manufacturer_Recommended` footprint was
created from the saved 2015 BM77 datasheet, Figure 1-10 (host lands) and Figure
1-3 (numbering). This is a manufacturer reference, not a claim that the donor
uses that exact land pattern. It remains unassigned and unplaced.

The footprint origin is the center of the 12 x 15 mm module body, viewed from
the top, with pin 1 at upper left. Host lands are 1.5 x 0.7 mm, extending 0.5 mm
outside the body. Left pins 1-12 have 1.1 mm pitch; bottom pins 13-19 run left to
right; right pins 20-30 run bottom to top. The vacant right-side position between
pins 29 and 30 is deliberate: pin 30 is at the same height as pin 1. The F.Fab
chamfer is a pin-1 drawing convention, and the courtyard is a library clearance
choice. Neither is a measured donor feature. No custom solder-mask or paste
allowances, RF routing, keepouts, or 3D model have been inferred from the photo.
KiCad's footprint SVG export parsed the library successfully. The rendered pad
layout was visually compared with the source figures, and all 30 unique pad
numbers, pad sizes, and coordinate anchors were checked. This validates the
entered manufacturer geometry, not the donor match or board routing.

The project-local `Alpha1S:PAM8003_SO16_Manufacturer_Recommended` footprint uses
Diodes DS36421 Rev. 2-2, May 2017, pages 10-11. The manufacturer PDF was retrieved
from `https://www.diodes.com/assets/Datasheets/PAM8003.pdf` and saved as
`reference/pam8003-datasheet-2017.pdf`; its SHA-256 is in the reference manifest.
The package drawing and suggested land pattern show 16 leads and no exposed
thermal pad. This source postdates the donor and is a package reference; the
donor land pattern still needs measurement.

For U12, the source specifies 1.27 mm pitch and 1.45 x 0.67 mm lands. In the
library's vertical orientation, their outer extents are 6.40 x 9.56 mm, with
pad columns centered at x = +/-2.475 mm and end pad centers at y = +/-4.445 mm.
The body drawing uses nominal 3.9 x 9.9 mm dimensions. Courtyard and pin-1
chamfer are library conventions. The footprint remains unassigned and unplaced.
KiCad parsed and exported it; visual review and coordinate checks matched the
source. All sixteen existing schematic pin names and numbers agree with the
datasheet and footprint. This check does not establish any donor connections.

KiCad opened the schematic and exported SVG and XML netlist. Visual review found
overlapping MCU identification fields; their spacing was corrected. Symbol
anchors were placed on the 1.27 mm connection grid. ERC still reports disconnected
and undriven pins because connectivity is not reconstructed; those findings are
retained rather than hidden with no-connect flags or relaxed rules.

The user has obtained a caliper and supplied the dimensions above. Exact shoulder
offsets, shoulder depth, corner radii, mounting geometry, thickness, connector
pitch and host land patterns still need measurement. A readable macro photo of
each unresolved IC and an unpowered continuity map will be needed alongside those
dimensions. The user explicitly reported no multimeter yet on 2026-10-17;
continuity mapping and voltage measurements are pending tool availability.

## U3 charger identification

The new back photograph has a faint marking consistent with BQ24133. Its visible
arrangement of ten pads on each long edge and two on each short edge agrees
with the TI RGY 24-pin package. This initially supported a candidate identification.
On 2026-10-18 the user directly read `BQ24133 / TI` on the physical U3, confirming
the part identity. The remaining spoken characters are preserved verbatim in
`chip-readings-2026-10-18.json` without decoding a date or ordering suffix.
The unmodified rotated detail is retained in
`reference/U3-marking-crop.png`; its source is `IMG_4511 2.heic`.

TI's official BQ24133 PDF is saved as `reference/bq24133-datasheet-rev-d.pdf`
and hashed in the manifest. The technical revision is SLUSAF7D, September 2020;
the downloaded packaging addendum is dated November 9, 2025. These postdate the
donor. Pages 5-6 provide the pin assignments; the package addendum lists BQ24133
as the top marking. No reference application circuit is treated as donor wiring.

The installed KiCad `Battery_Management:BQ24133RGY` symbol was copied into the
project library and every pin name/number checked against those pages. Library
pin 25 represents AGND, the exposed thermal pad; TI calls this Thermal Pad
rather than assigning it a numeric lead. U3 is now visibly labelled `BQ24133`
and has no footprint assignment or wires. Its CELL configuration,
current-setting resistors, thermal-pad connections and donor orientation remain
unverified. No donor charging current or voltage has been inferred from the identity.

At the initial candidate stage, the nine-symbol draft exported successfully to
SVG and XML netlist. ERC reported
187 findings: 144 unconnected-pin, 21 undriven-input, and 22 undriven-power
findings. There are no off-grid findings. These are expected unresolved circuit
connections, not a passing electrical check. Visual review keeps the candidate
and its evidence note separate from the existing symbols.

## Initial physical component map

`component-placement.json` records eleven manually selected package bodies in
the calibrated photographs. Source pixel bounds are retained so selections can
be reviewed. Back-side x coordinates are mirrored about the 72.08 mm board
width to use the same front-view datum as Edge.Cuts. Centers are rounded to
0.1 mm; extra digits in bounding boxes support drawing, not measured accuracy.

U10 is on the front. U8, U3, U12, U1, U4, U9, U13, U14, Q1 and Q2 are on the
back. These include unresolved parts whose positions are visible even though
their identities are unknown. Native KiCad contains 44 locked body-box segments,
11 F/B reference labels and one explanatory legend, all on drawing/comment
layers. No pads, drill holes, tracks or electrical nets were created.

The module's asymmetric pad gap appears on the lower row near its right end.
Compared with the manufacturer's top-view numbering, this is consistent with a
90-degree clockwise rotation and pin 30 at the lower right in the front photo.
This is orientation evidence only. Solder visible outside the module cannot
establish the full host-pad centers or dimensions hidden underneath it. The
manufacturer-reference footprint remains unassigned and unplaced.

The three-panel review image shows each selected body over the source photo,
then both sides in a shared front-view layout. Native review confirms the saved
markers inside the outline. These are placement references to guide subsequent
footprints; donor land patterns, physical accuracy and pin-1 mappings are still
pending verification.

## Resistor markings and unresolved IC review

Twenty-two resistor observations are recorded in `passive-observations.csv`
and incorporated into the inventory. Fourteen provide readable nominal-value
markings: R9/R10 are R020 (0.02 ohm); R12/R15 are 3R9 (3.9 ohm); R28 is 472
(4.7 kohm); R62, R107, R111, R112, R116, R117, R134, R144 and R145 are 103
(10 kohm). These values come from the photographs, not resistance measurements.
The marking conventions were checked against ROHM's manufacturer guide:
https://fscdn.rohm.com/en/techdata_basic/resistor/marking/MCR-E.pdf . This does
not identify the donor resistor manufacturer or series. Package size, tolerance,
power rating, temperature coefficient and electrical assignments remain unknown.

R115, R23 and R110 have ambiguous reading orientation. R31/R32, R35 and
R24/R26 have letter/number markings that need clearer reading before decoding.
Their nominal-value fields remain empty. No resistor values were inferred from
the charger reference circuit, and no donor nets were added from those examples.

At that inspection stage the photographs did not support exact identifications for U1, U4,
U9, U13, U14, Q1 and Q2. The U8 suffix appears more legible in the new crop but
remained a candidate pending an unambiguous reading. Later direct readings resolve
U4, Q1, Q2 and U8, as recorded below. Inspection details are saved
under `reference/parts/` without generated detail or sharpening. That directory's
manifest records hashes, crop provenance and rotation. These observations avoid
repeated inspection of the same unresolved pixels while preserving the evidence
for comparison with future physical readings.

## J6 keyed header geometry

Both photographs show nine plated holes in a two-row, five-column pattern with
one corner absent. The front silkscreen end labels are interpreted as 1, 2 and
9, and the back view has a square pad at the corresponding pin-1 corner.
Intermediate odd/even numbers are inferred from that convention, not individually
printed or electrically verified. In the front view with ports at the top, the
upper row is 1/3/5/7/9; the lower row is 2/4/6/8 with position 10 absent.

Manual hole-center selections yield mean column pitches of 2.019 mm (front)
and 2.026 mm (back); row pitches are 2.040 mm and 1.956 mm. This supports a
2.0 mm nominal-pitch candidate. It is not a caliper measurement. Corresponding
centers differ by up to 0.531 mm between views. The saved draft centers average
the two views after mirroring the back into the front-view coordinate system.
Source crops, selected pixels and coordinate calculations are retained in
`j6-header-geometry.json` and `reference/connectors/`.

The native board contains nine center crosses with numbers and a J6 label on
drawing/comment layers. They create no drilled holes or copper pads. The new
project-local schematic symbol has pins 1-9 only, is marked DNP and excluded
from the BOM to represent the donor's unpopulated site. Pin 10 is absent, not
hidden with a no-connect flag. No debug function, signal assignment or connector
part number is inferred. Drill diameter, land size, exact pitch and connections
still need verification. The ten-symbol draft exports to SVG; ERC now has 196
unconnected/undriven findings, including the nine newly documented contacts.

## Connector and switch site map

Twelve visible housing envelopes were selected on the front photograph: J2,
J7-J10, J3, J13, SW1, SW2, and the SPK, LED and barrel-power sites. The last
three names are observation labels where the reference designator is unreadable;
DC_IN is not an invented donor reference. `connector-sites.json` retains source
pixels, clipping status and bounds. Seven housings touch the photo border, so
their full extents and package centers cannot be recovered from those crops.
Housing faces may sit above the PCB, making them unsuitable for precision pad
position or pitch measurement due to parallax.

The board contains 48 locked envelope segments, twelve labels and one legend on
User.Eco1/User.Comments. This extends the physical review map without creating
footprints or inferred copper. Three new inventory entries cover SW1, SW2 and
the barrel-power site, bringing the partial inventory to 47 entries. Connector
family, exact variants, full dimensions and electrical assignments remain open.
The review image compares all selected envelopes directly with the front photo.

Requested direct text readings for U4, Q1 and Q2 because their existing image
markings remain unreadable. This request does not require another board-scale
reference or repetition of the top-down photo setup.

## Five marked front-side capacitors

C23, C64, C65, C66 and C67 each show `107 C`. The three-digit pF code
corresponds to 100 uF nominal capacitance. The AVX TAJ datasheet, pages 1-2,
documents that coding scheme, voltage code C = 16 V, and an anode polarity
band: https://datasheets.kyocera-avx.com/TAJ.pdf. It is a marking reference,
not proof that these donor parts belong to the TAJ series. Voltage and polarity
interpretations remain conditional until manufacturer/series identification.

With the front facing the viewer and ports at the top, the visible band is
on the left for C23/C64/C65 and on the right for C66/C67. This is an observed
orientation, not a recovered pad number or net assignment. Secondary marking
characters are not used to infer a date, tolerance or series. No ESR, capacitance,
height or electrical connection was measured. All five entries retain unknown
series, footprint and nets; the partial inventory now contains 52 entries.

`capacitor-observations.json` retains pixel crop bounds and hashes of the source
image and five unchanged-resolution crops. Crop boundaries include surrounding
silkscreen and solder and must not be used as package dimensions. The large
central yellow two-lead part has not been classified from its appearance.

The user's request to show U4/Q1/Q2 was answered with board locations and crops.
Actual text readings from those three devices remain pending.

## Central yellow protection-device candidate

The central yellow two-lead component has a `16` line and a `GF1000`-like
line. The first character and logo remain faint. Tyco/Raychem specification
SCD 25199 Rev B (2005-06-29) shows this older marking for RGEF1000, a resettable
PPTC protection device. This is a strong photo-supported candidate, not a
physically confirmed part. The 2022 Littelfuse datasheet uses the newer `G10`
marking; its manufacturer-hosted UL record associates both markings with
RGEF1000. Sources and hashes are in `power-protection-observation.json`.

The candidate specification lists 16 V maximum, 10 A hold and 17 A trip at
25 C. These describe the component, not the allowable board load or spare
capacity for the Q6A. Neither terminal has a recovered net. The donor reference
designator is unreadable, so `CENTRAL_YELLOW` is only an inventory site label.
Exact ordering suffix, mounting geometry and electrical path remain unresolved.
The partial inventory now contains 53 entries. No schematic net, footprint,
copper or replacement part was assigned from this candidate alone.

## Direct chip readings, 2026-10-18

The user read both Q1 and Q2 as IR-logo devices with `F9310`, explicitly stating
they match. This supports International Rectifier IRF9310 P-channel MOSFETs in
SO-8, cross-checked against the manufacturer's Infineon-hosted datasheet.
The ambiguous P524H-like strings are preserved as spoken and are not used as
part numbers or decoded production dates. Exact ordering suffix remains open.

U4 was read as `MOSA / MS6324G / 1306490`. MOSA's MS6324/26 Rev 2 marking table
identifies MS6324G as the SOP8 version of its 16-bit stereo I2S DAC. Tube and
tape/reel variants share this marking. Its pin table is recorded in
`chip-readings-2026-10-18.json`; a manufacturer-authored PDF is saved with hash
and mirror provenance. No board nets follow from the datasheet alone.

Updated the inventory and physical component map identities. Inventory has
54 entries after separating the former combined Q1/Q2 row. These readings
resolve the previous identification request. Native schematic placement of
these three parts and donor connectivity remained pending at that stage.

Q1, Q2 and U4 are now added to the native schematic and project-local symbol
library. All eight package pins are explicit on each recovery symbol. The
IRF9310 source pins are 1-3, gate is 4, and drains are 5-8, visually checked
against International Rectifier PD-97437A page 1. Page 7 also documents the
logo, part marking and separate traceability fields. The 2010 manufacturer PDF
is saved as `reference/irf9310-pd97437a-2010.pdf` with mirror provenance and hash
in the chip-readings record. No manufacturing date was inferred from the user's
uncertain secondary text.

The native editor was reopened and the three new symbols visually checked.
SVG export succeeds. The thirteen-symbol page has 184 numbered package pins
and zero intercomponent wires. ERC reports 227 unconnected/undriven findings;
all findings are in those categories, with no off-grid or library-consistency
findings. This is an incomplete circuit, not an electrical validation pass.
New symbols have no assigned footprints or inferred donor pin-1 orientation.

## SO8 package references

Added an unassigned IRF9310 reference footprint from PD-97437A page 7:
eight 1.78 x 0.72 mm rectangular lands, 1.27 mm pitch, 6.46 mm overall
span across rows. Pad-column centers are therefore 4.68 mm apart. The fab
body is the 3.9 x 4.9 mm midpoint of the package limits; the courtyard is
a project allowance. KiCad successfully parsed and exported the footprint.

MOSA Rev 2 page 10 gives the same nominal body envelope and lead pitch for
MS6324G, but no recommended land pattern. Its package limits are recorded in
`so8-package-references.json`; no MOSA land recommendation was invented.
Neither reference establishes the original board's copper pad dimensions.
These footprints remain unassigned and unplaced pending donor comparison.

Requested the next direct reading for U3 with whole-board landmarks: the wide
chip immediately left of the large central gray square, ports/narrow end up.
The user supplied `B Q two four one three three T I four A I C four V nine G four`.
This confirms BQ24133; the secondary characters remain an uncertain transcription.
Updated the native schematic label and evidence note, inventory, placement
record and chip-readings record. U3, Q1, Q2 and U4 reading requests are resolved.

After U3 confirmation, the native editor was reopened and its updated value and
evidence note visually verified. SVG export succeeds. ERC remains at 227
findings (177 unconnected pins, 26 undriven inputs, 24 undriven power pins),
with no new categories. Symbol pin data and all electrical connections are
unchanged; thirteen symbols and zero wires remain.

## U8 direct marking confirmation — 2026-10-18

The user read STM32F103 RDT6 and an ST logo on U8. The full spoken secondary
marking is preserved in chip-readings-2026-10-18.json without date-code decoding.
ST DS5792 Rev 13 page 135 decodes R as 64 pins, D as 384 Kbytes Flash, T as
LQFP and 6 as -40 to +85 C. The official web PDF was inspected; direct local
download failed, so no local PDF copy is claimed.

Updated the native schematic value, evidence note and datasheet link, inventory,
placement identity and measurement M12. The cached MCU pin map is unchanged.
Reopened KiCad and visually verified the label. SVG export succeeds; ERC stays
at 227 unconnected/undriven findings (177/26/24). There are still 13 symbols and
zero wires. This confirms the part identity, not donor connectivity or firmware.

## U1 area direct readings — 2026-10-18

The user read U1 as `04 equals 9 2 Y`, explicitly confirmed that the first
character is zero, and counted three leads on each side. The literal reading
is recorded as `04=92Y`, with all spoken variants retained. Richtek RT8057AGJ5
uses an `04=` marking but has five leads, so it is excluded for this six-lead
donor. U1 remains unidentified; no pin map or footprint has been assigned.

The user also confirmed L1 is the adjacent part marked `H dot`. Its inductance
is unknown. R3 `103` decodes to 10 kohm, R4 `563` to 56 kohm, and R5 `85B`
to 7.5 kohm using EIA-96 (85 = 750; B = x10). The Bourns CRP0603 marking
table verifies the coding convention, without identifying the donor manufacturer.
These are nominal values from markings, not measured resistances.

Saved the direct readings and document hashes in
`u1-area-readings-2026-10-18.json`, updated U1's chip and placement records,
and added R3/R4/R5/L1 to the inventory. The inventory now has 58 entries;
passive observations have 25 entries, with 17 decoded and eight ambiguous.
No schematic pins, wires or PCB geometry changed from these readings.


## IMG_4512.MOV evidence recovery — 2026-10-18

Preserved the original 81.89-second video and twelve full-resolution frames at
2, 6, 12, 16, 22, 26, 30, 34, 42, 54, 66 and 74 seconds. Hashes and provenance
are in video-readings-2026-10-18.json. The video is rotated 180 degrees from the
existing back photograph; the native PCB datum was not changed.

U9 reads C25R with eight leads. TI's SN74LVC2G125 marking table and package
information support the dual buffer identification and DCU VSSOP-8 package;
exact ordering/finish suffix remains unassigned. U13 and U14 both read YJ33,
which SGMICRO's SGM2019 table maps to the 3.3 V regulator. Their photo sizes
support SOT-23-5, but SC70-5 shares that marking: verify dimensions before
ordering. Both package variants use the same five-pin map. Manufacturer PDFs
are saved locally; page 4 pin diagrams were visually checked for both devices.
U9's dot and the SGM2019 text orientation convention now establish their
photo-relative pin-1 locations. No donor nets are inferred from this.

U1's exact 04=92Y marking and six leads are both clear at 22 seconds, confirming
the user's prior reading without resolving its identity. U12 is visibly
PAM8003 / THC51674. Q12, Q7 and Q11 read J3; short-code identification remains
unresolved. No capacitance was assigned to unmarked ceramic parts.

Recorded 45 resistor observations, including the user's spoken batch and
additional clear video labels. R60 is the unnamed 103 resistor beside Q7.
Resolved previous R31/R32 01D readings as 100 kohm and R35 18C as 15 kohm using
EIA-96. R56's spoken 201 is retained; its 180-degree 102 ambiguity remains
unassigned. Inventory now has 103 entries; passive register has 67, including
61 decoded nominal values and six unresolved readings. All are marking-based,
not electrically measured values.

Added U9/U13/U14 to the native schematic with manufacturer pin functions.
Sixteen symbols, 202 numbered package pins, zero intercomponent wires. No
footprints assigned to the new symbols. Native editor reopened and visually
verified; SVG export succeeded. ERC has 257 findings: 195 unconnected pins,
32 undriven inputs and 30 undriven power inputs, with no other categories.
These expected incomplete-circuit findings are not an electrical validation.


## Full-duration video review, 2026-10-18

Reviewed 164 samples at 2 fps across the entire 81.89-second recording using nine
chronological overview sheets and full-resolution detail views. The initial
twelve-frame pass was incomplete as an extraction of readable evidence.
`video-fullscan-2026-10-18.json` records the follow-up and hashed supporting frames.

Added thirteen reference-resolved resistor readings: ten new entries and three
previously ambiguous readings resolved (R24, R26, R23). R24 reads 68B (4.99 kohm),
R26 65B (4.64 kohm), R13 68D (499 kohm), R8 30E (2 Mohm), and R20 59B
(4.02 kohm). Values follow the saved manufacturer marking table; no resistance
measurements or donor manufacturer assignments are implied.

Withdrew the unique R58 value because a second R58-like label appears by Y1.
The header-site 270 reading is preserved by location. The three adjacent
markings near Y1 are 103/243/513, but reference mapping is unresolved.
Register now has 77 entries: 73 decoded and 4 unresolved. Inventory has 128 entries.

Q10 reads 1YD with an m-like trailing character. Q3/Q4 show 1733 and secondary
characters, with six leads. These are marking observations, not identified parts.
Twelve empty sites around U6 are recorded as visibly unpopulated without assuming
why. U6 has 28 perimeter lands and a central land.

Five short surface connections at U13/U14 are now explicitly recorded between
known IC pins and photograph-relative capacitor terminals. They are supported
visually, not meter-verified, and do not establish capacitor values, opposite
terminal assignments, or complete nets. Native schematic wiring and PCB routing
were not changed in this evidence-registration step.


## Q3/Q4 video identification, 2026-10-18

The 1733 markings and six-contact WSON bodies match TI CSD17313Q2. Saved the
official SLPS260E datasheet and checked the package marking addendum and page8
pinout drawing visually. Pin1/2/5/6/8 are drain, pin3 gate, pin4/7 source.
Pads7/8 are underside lands despite the ordering table describing six pins.
The local symbol includes all eight numbered lands. Added Q3/Q4 instances with
no wiring or assigned footprints; previous sixteen instances preserved. Exact
reel/order suffix, donor pin1 orientation and nets remain unresolved.

Q10 1YD has a discovery-only L8550Q-family match from an aggregator. No primary
manufacturer marking diagram obtained; no identity, pin map or schematic symbol
assigned on that basis. Evidence is in mosfet-video-identifications-2026-10-18.json.


## First native connection pairs, 2026-10-18

Entered five previously recorded video-visible surface connections into the
native schematic: U13.5-C79.1, U13.4-C80.1, U14.5-C83.1, U14.4-C81.1 and
U14.1-C82.1. Capacitor pin1 is an explicitly recorded physical-terminal
convention, not a polarity claim. Values remain Unknown and all pin2 terminals
remain open. No ground or supply assignments inferred.

Native export verifies exactly these five inter-component pairs and verifies
the opposite capacitor terminals remain isolated. Existing U3/U8 stacked pins
produce internal single-component groups in KiCad and are not additional
recovered board nets. Schematic now23symbols228pins14wire segments. SVG export
succeeded and the reopened native view was visually checked. ERC remains275:
211pin_not_connected,34pin_not_driven,30power_pin_not_driven; no other categories.
PCB placement and routing remain unrecovered. Full evidence and verification
are in regulator-connections-2026-10-18.json.


## Power-region follow-up, 2026-10-18

Frame142 shows the Q1 package index at lower left and a short vertical trace
from its third bottom contact (source pin3) to C68 top terminal. Added C68
Unknown and this pair to native schematic; pin2 is opposite terminal, unrecovered.
Native export verifies Q1.3-C68.1 plus all five previous inter-component pairs.
Schematic24symbols230pins17wire segments; GUI reviewed and SVG export passed.
ERC remains275 in the same three unconnected/undriven categories.

Frames138/140/142 show R49 connected across Q10's two same-side leads: top
resistor terminal to upper-left lead, bottom terminal to lower-left lead. These
are recorded physically without assigning Q10 functions or native pin numbers.
An LRC manufacturer-authored Rev.O datasheet hosted by distributor X-ON now
supports an L8550Q-family candidate for 1YD (PNP, B1/E2/C3). The donor maker
and trailing m-like glyph remain unresolved; this does not confirm an exact
orderable donor part. PDF preserved and hashed.

R1 513 confirms the existing51kohm entry; resistor counts unchanged. No new
Q3/Q4 connection was verified through the obscured copper/package regions.


## Caliper measurements, 2026-10-18

User reported bare PCB thickness 1.64 mm and one mounting-hole diameter
3.20 mm. Recorded in measurements.csv (M02, M03D) and geometry.json.
The measured hole location was not specified; do not assign the diameter
to every hole or infer center coordinates. The native board still has its
previous 1.6 mm draft setting; applying measured geometry in KiCad is pending.

User clarification: all four mounting holes are the same. Applied the reported
3.20 mm diameter to all four holes in measurements.csv and geometry.json,
superseding the one-hole restriction above. Hole center coordinates remain
pending; no need to identify an individual hole for diameter assignment.


## Measured geometry applied to native board, 2026-10-18

Saved and closed the PCB editor before updating the native file. Board thickness
is now 1.64 mm. Added four 3.20 mm diameter drawing circles on Dwgs.User at the
existing photo-derived H1-H4 centers. These drawing references do not introduce
unverified plating or drilling. Updated photo-geometry.json diameter fields and
provenance. Center disagreements between the photographs remain 0.258-0.506 mm.
Backup preserved in .run/alpha-measured-mechanics/.

Validation: KiCad SVG export succeeded, board reopened and all four circles
visually checked. Exact content comparison confirms only the thickness field
and six appended drawing objects changed. Each circle diameter is 3.20 mm.
Verification record: mechanical-measurements-2026-10-18.json.


## Circuit priority and input-switch sheet, 2026-10-18

User deferred mounting-hole measurements and restated the objective: recover
the original board and actual circuit to support our V2. Mechanical refinement
is deferred. Added input-switch-region.kicad_sch to the root hierarchy, with
Q10 represented by physical terminals A/B/C (upper-left/lower-left/right in
frame138), without assigning B/E/C or manufacturer pin numbers. R49 10k is wired
across A/B from the existing visually supported connections. Added nearby
R50 1k, R65 10k and R61 1k; their connections remain unrecovered.

Native project now has 29 symbols, 241 numbered pins and 21 wire segments over
two sheets. Exported netlist verifies both Q10/R49 pairs plus all six previous
pairs; unresolved new pins remain isolated. SVG exports for both sheets passed;
GUI reviewed. ERC282: 218 pin_not_connected, 34 pin_not_driven,
30 power_pin_not_driven; no other categories. Backup and script are under
.run/alpha-input-circuit; add.py must not be rerun.


## First linked PCB footprint: Q1, 2026-10-18

Assigned the existing IRF9310 manufacturer-recommended footprint to Q1 and
placed it on B.Cu at (365.8, 240.4) mm, corresponding to photo-derived datum
(25.8, 20.4) mm. Pin1 is upper-right when viewing the back of the donor,
matching frame142 and mirrored front-view placement. Native angle -90 degrees.
Eight pads inherit their schematic net names; pad3 carries Net-(C68-Pad1).
The C68 footprint is still absent, so this connection is not routed.

The native board now has one footprint, eight copper pads, zero tracks and
zero vias. No new donor connections inferred. Manufacturer lands are not proof
of exact donor pad geometry; photo center remains approximate. All eight net
names, symbol UUID path and assigned library ID verified. Native board reopened
and visually checked. Final DRC has zero geometric/library violations and
28 missing-footprint parity findings, which explicitly reflect the incomplete
layout. Evidence in q1-pcb-placement-2026-10-18.json; backups in
.run/alpha-q1-placement/. add.py must not be rerun.

## 2026-10-18 - U9/U13/U14 placement and resistor entry

Placed three linked back-side footprints: U9 at (369,258.7), U13 at (357.2,268.4), U14 at (400.3,274.5) mm. Pin-1 orientation comes from recorded video evidence. TI DCU0008A page25 defines 0.85 x 0.30 mm lands, 0.5 mm pitch and 3.1 mm column spacing; SGM2019 page15 defines 0.99 x 0.69 mm lands, 0.95 mm pitch and 2.59 mm column spacing. These are manufacturer lands, not verified donor land dimensions. U13/U14 package selection follows photo body scale; direct physical package measurement remains unresolved. All18 new pad net names and world coordinates were checked through the running KiCad API. Final DRC reports zero geometry/library violations and97 missing footprints in the now101-symbol project. No routed tracks or vias were added.

Imported72 observed resistors into two native schematic sheets:69 decoded values plus R56/R110/R115 as Unknown. R49/R50/R61/R65 remain on their existing input-switch sheet; R58 remains excluded because of reference conflict. All144 added resistor terminals are isolated and their physical pad convention remains provisional. Netlist comparison confirms the same eight existing intercomponent connections. ERC has362 unconnected-pin,34 undriven-pin and30 undriven-power findings; no other categories. Rendered and checked the four-sheet PDF, then moved bottom notes clear of the new hierarchy links. This adds recovered components and placement; it does not establish additional donor wiring.

## 2026-10-18 - Amplifier connections and three more placements

Video frame007 and adjacent frame008 show the PAM8003 pin-1 dot and three short local connections: U12.5 to R143 right terminal, U12.6 to C69 right terminal, U12.7 to R126 right terminal. Each passive right terminal is schematic pin1; opposite terminals remain isolated. Moved existing U12/R143/R126 into a dedicated amplifier sheet and added C69 with Unknown value. This gives102 symbols387 pins across5 sheets and11 intercomponent nets. Netlist comparison preserved the8 prior pairs and verified3 new pairs. ERC420:358 unconnected,32 undriven,30 undriven-power; no other categories. The image evidence supports local surface traces, not complete branch membership or electrical verification.

Placed Q2 B.Cu(371.4,240.3), angle-90, using the clear package index dot in frame142; U12 B.Cu(359.6,279.5),angle0, using frame007; U10 F.Cu(397,271.2),angle-90, using the unique30-pin BM77 pad pattern and gap between pins29/30. Manufacturer land libraries reused. All54 new pad coordinates and canonical net names checked through live KiCad API; U10 slashes inside pin names require {slash} in PCB canonical nets. DRC0 geometry/library problems,95 missing footprints and0 other parity issues. Board7 footprints80 pads0 tracks0 vias. Exact donor lands and photo center accuracy remain unverified. Updated5-page schematic PDF rendered and reviewed.

## 2026-10-18 — U8 MCU orientation and placement

ST TN1433 Rev7 section4.4 explicitly locates pin1 at the lower-left with upright text, resolving the two corner mould marks on U8. The donor back reference is upside down, putting its pin1 corner upper-right; mirrored front datum upper-left. Added U8 at395,261.6mm B.Cu,-90degrees using a project-local copy of KiCad LQFP64 reference copper geometry. These are not measured donor lands. All64 pad coordinates and net assignments were checked through the live board API. Native board now8 footprints144pads,0tracks0vias. All11 schematic intercomponent nets unchanged. DRC0 geometry/library issues;94 missing-footprint parity findings only. Board reopened and selected U8 visually checked. Sources and hashes are in mcu-pcb-placement-2026-10-18.json.

## 2026-10-18 — Amplifier supply path

Traced the visible path from U12 pin9 through C76 left terminal to the existing U13 pin5/C79 left-terminal net, using frame06s with fullscan007/008. C76 pin1 is its left terminal; capacitance and pin2 connection remain unknown. U13, C79 and C80 moved together into the audio sheet; their two existing local connections were preserved. One new component, not three. Native103symbols389pins5sheets,11intercomponent nets with one expanded four-terminal net. Exact net comparison preserves all other memberships. U12 pad9 and U13 pad5 now both have Net-(U12-VDD), and U13 PCB path points to the audio sheet. Board8footprints144pads0tracks0vias. DRC0geometry/libraryissues95missingfootprints only; ERC419 solely unresolved-pin categories. Updated audio/root PDF pages rendered and reviewed.

## FCC internal photographs retrieved - 2026-10-20

The earlier retrieval failure is resolved. The eight-page manufacturer photo
exhibit is saved at `reference/fcc-2015/internal-photos-2712698.pdf`; its SHA-256
matches the mirror metadata. Exhibits 2712698 and 2712710 identify the same file.
All eight pages were inspected. Pages 1-2 contain the mainboard views; later
pages show servos, battery and adapters. Photo 4 reads ALPHA 1S V1.1 / 2015-01-04,
matching the donor revision/date. It also shows rework near the microSD socket
that is absent from the donor front photograph, so exact population and net
identity are not assumed. Embedded original JPEGs from pages 1-2 are preserved.

The complete share366/RobotControl tree was inspected at
`f78a1836d2cc601394d08369bb540d7717100f35`. It is a fork of
Super169/RobotControl with firmware, protocol documents and linked PCB images;
no editable PCB design or Gerber files were found in that tree. The upstream
servo implementation sends ten-byte commands. This is useful V2 interface
reference, not the donor mainboard schematic. The quoted WhyCan board-variant
claim remains unverified. See `reference/fcc-2015/evidence.json`.

No schematic connectivity, footprint placement, or PCB copper changed in this pass.

## Chinese source chain recovered - 2026-10-20

The exact WhyCan thread https://whycan.com/t_2539.html links alicedodo's
Gitee servo reverse-engineering project. Its partial servo circuit and four-type
comparison image were downloaded and visually inspected, saved under
`reference/chinese-research/`. The comparison includes ATmega8-based designs
and an ATSAMD10/DRV8838 design, so a blanket AVR statement is unsupported.
The author's type labels are local to that comparison. Exact donor servo type
remains unverified. This source supports interface research for V2, not missing
mainboard nets. The original post itself says Alpha 1S or Qrobot tentatively.

Located copies of the 2017 teardown text report BM77SPP03MC2 and JDT A16-1000.
The latter is recorded as an external lead in `power-protection-observation.json`,
without replacing the donor's RGEF1000 candidate based on its visible marking.
No donor schematic or PCB changes were made. See source manifest for evidence.

## 2026-10-20 — Bluetooth local connections

The independent front photograph and the newer front-scaled photograph both show short copper necks from U10 RXD pin20 to the left terminal of R101 and U10 TXD pin21 to the right terminal of R88, with the module label upright. Passive pin1=left and pin2=right in that view. Added R101=27 ohm, R88=0 ohm, R100=27 ohm and R85=1k from markings270/0/270/102; none measured. R100 and R85 remain unwired. U10 moved into bluetooth-region.kicad_sch and its existing PCB hierarchy path and two pad nets were updated. All previous multi-terminal net memberships are preserved exactly. Native107symbols397pins6sheets13intercomponent nets; board8footprints144pads0tracks0vias. ERC422 solely unresolved-pin categories; DRC0geometry/library violations and99missing footprints only. Root and Bluetooth review PDF pages rendered and checked. No MCU endpoint or additional resistor branch inferred; full circuit remains incomplete. Sources and validation are in bluetooth-connections-2026-10-20.json.

## 2026-10-20 — U4 DAC footprint

Video022/026 resolve U4 pin1 by its lower-right package dot. The back-scaled photo is rotated180, putting the dot upper-left, then upper-right in the mirrored front board datum. Placed U4 at383.6,280.8mm B.Cu180degrees using KiCad SOIC8 3.9x4.9mm reference copper geometry, not measured donor lands. All8 pad coordinates and nets verified through live IPC; GUI placement checked. Nine footprints152pads0tracks0vias. All13 existing intercomponent nets unchanged; DRC0geometry/library violations98missing footprints only. Adjacent DAC signal branches remain ambiguous; no connections added.
