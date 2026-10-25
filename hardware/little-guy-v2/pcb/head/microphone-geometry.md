# Head-A microphone locations — frozen snapshot

Generated from `microphone-targets.json` by `tools/head_geometry.py`. Do not edit dimensional values here;
explicitly recapture the owner-frozen points with `tools/capture_head_targets.py`, then rebuild.

The front and back targets each moved 2 mm inward in the owner's frozen revision. A0.2 geometry is superseded.
The current source targets match the frozen table in `PCB-BRIEF.md` within its 0.01 mm rounding. Source hashes
now identify the new CURRENT scene, head generator, brief and existing mechanical fit study. No model or fit-study
source was changed. Source mic points are CAD markers, not measurements of fabricated hardware.

The horizontal front/back span is **58.000000 mm**, lateral span
**100.000000 mm**, and front-to-back height difference **11.651600 mm**.
The orthogonal least-squares plane tilts **11.361876 degrees** front-up. Maximum normal
departure is **0.335687 mm**, within the frozen 0.34 mm allowance.
The front/back distance on this plane is **59.158768 mm**.

These are **acoustic opening targets**, not package centers, pin 1 or PCB copper-surface coordinates.
The raw shell targets remain frozen in `microphone-targets.json`; the table below contains their projections
onto the shared acoustic plane. The JSON includes source points, projected points, residuals and the head transform.

| Ref | Required logical index | Location | PDM net | XVF3800 pin | Array X mm | Array Y mm | Array Z mm |
|---|---:|---|---|---:|---:|---:|---:|
| MK501 | 0 | front | PDM_DATA0 | 46 | 0.000000 | 29.646533 | 0.000000 |
| MK502 | 1 | back | PDM_DATA1 | 50 | 0.000000 | -29.512235 | 0.000000 |
| MK503 | 2 | left | PDM_DATA2 | 48 | -50.000000 | -0.067149 | 0.000000 |
| MK504 | 3 | right | PDM_DATA3 | 51 | 50.000000 | -0.067149 | 0.000000 |

Array frame: right-handed, X toward robot right, Y toward robot front/up along the plane, Z outward normal.
Origin: centroid of the four source targets. KiCad target X equals array X; target Y equals minus array Y.
The small fore/aft asymmetry follows the source points and their projection. Do not replace them with a stock array.

## Mechanical fit reference

`microphone-fit-reference.json` preserves the existing CAD study associated with the frozen points. Its trial
diamond tips extend 3 mm past the mic centers and its sampled slab check reports no clashes. Its wall gaps are
package-to-wall values, before deducting the extra PCB edge. This is CAD evidence under that study's envelope
assumptions, not physical validation. Its fitted plane minimizes vertical error; this file minimizes normal error.

The projected acoustic plane is a datum, not a PCB top-surface height or released perimeter. Board thickness,
exact package height and sound-port offsets/rotation, shell wall, acoustic passages/gaskets, mounting and full
component clearance remain design work. Keep the shell targets fixed and accommodate the residual with the
acoustic interface; do not bend the board against four non-coplanar shell points.

## Firmware connection

`mic_geometries.fragment.yaml` contains only the custom `SQUARECULAR_GEOMETRY` block in metres. Merge it into
the custom product's settings and retain its other required parameters. Match logical MIC0..3 to the physical
PDM input order above; do not inherit the development kit's channel mapping. No firmware has been built or flashed.
`AEC_MIC_ARRAY_GEO` is read-only; read its 12 XYZ metre values back from the built image.
See [XMOS user guide sections 3.5, 4.3 and 10.1](https://www.xmos.com/documentation/XM-014888-PC/pdf/xvf3800_user_guide_v3.2.1.pdf).

Verify raw mic channels individually, then test known front/back/left/right sources and direction conventions.
Repeat in the final shell with actual speakers and echo reference. The geometry calculation establishes no
directional, beamforming or echo-cancellation performance. No robot-simulation electronics placement was added.
