# ST3215 interface reference — 2026-10-21

Status: source drawings visually compared and the supplied STEP inspected. These are nominal reference dimensions, not a verified fit for purchased hardware. No bracket or horn has been released for manufacture.

![Supplied servo model and output axis](reference/servo-step-review.png)

## Sources and configurations

The [FEETECH ST-3215-C018 specification A/0](reference/ST-3215-C018-spec-A0.pdf), dated 2023-07-20, is the closest model-specific source for the planned 12 V servo. Its page 6 drawing shows the bare servo; page 7 specifies no accessories. The [Waveshare drawing](reference/ST3215-2D/ST3215.pdf) and [STEP](reference/ST3215-3D/ST3215.step) include two round fittings and identify SCS215 internally, despite their ST3215 filenames. Keep these configurations distinct until the supplier confirms the actual SKU and accessory set.

| Feature | Evidence | Design use |
|---|---|---|
| Case length × width | C018 drawing 45.23 × 24.73 mm; older drawing 45.22 × 24.72 mm; STEP about 45.223 × 24.723 mm | Nominal envelope; mating tolerances unknown |
| Axis location along case | C018 middle datum to output axis 12.5 mm; half-length minus offset gives 10.115 mm from the near end, consistent with older 10.11 mm | Locate the joint at the output axis, not the case centre |
| Output spline | C018 25T / Ø5.9 mm | Matching purchased horn required; insufficient mating-spline geometry for manufacture |
| Driven-side centre screw | C018 M3 × 6 machine screw | Screw identity; not 6 mm guaranteed thread engagement |
| Opposite support | C018 Ø6 feature and PA3.0 × 5 self-tapping screw | Passive-support reference; no second driven output or M3 threaded interface established |
| Small case screws | C018 B-PA2.0 self-tapping screws | Case hardware; not verified bracket-mounting holes |
| Round fittings in older model | Ø19.2 mm; four modeled Ø2.5 mm holes on a Ø14 mm pitch circle | Accessory reference only; hole cylinders do not establish threads or load rating |

The axial spans use different endpoints. The C018 bare drawing shows 36.5 mm overall, alongside 35, 32 and 29 mm dimensions between particular surfaces. In the older STEP, the round fittings' outer faces are 37.25 mm apart, matching the older drawing. A housing feature extends another 0.55 mm, making that model's full bound **37.8 mm**. The layout now uses **45.23 × 24.73 × 37.8 mm** to enclose the inspected nominal references. This includes no manufacturing allowance, connector or cable space.

## STEP inspection

[servo-step-inspection.json](reference/servo-step-inspection.json) records the unchanged source hash, kernel version, bounds, validity and fitting circles. In the source coordinates, the output/support axis runs along Y through **X = −25.5, Z = 0**; the case's long-axis midpoint is X = −13. Its 12.5 mm offset independently agrees with the drawing. The full upper-body envelope uses centred boxes. The separate [right-shoulder motion study](layout/shoulder/README.md) now places this offset in a connected two-axis mechanism; its boxes still exclude actual brackets and cables.

Eight solids imported. Seven pass the kernel's validity check; solid index 2 reports `BRepCheck_SubshapeNotInShape`, and the assembly fails validity. This is a CAD-topology result, not evidence of a physical defect. The source was not healed or overwritten. The preview is a tessellated visualization; do not use the source unchecked for manufacturing booleans or infer mass from its volume.

Reproduce using an environment with [the pinned CAD dependency](analysis/requirements-cad.txt):

```sh
python hardware/little-guy-v2/analysis/inspect_servo_step.py --mesh-json .run/step-review/mesh.json
python hardware/little-guy-v2/analysis/render_step_preview.py .run/step-review/mesh.json
```

The renderer additionally uses Matplotlib and NumPy. Independent review reproduced the reference dimensions and axis offset, and confirmed the accessory distinction and invalid solid.

## What remains before fitted mounts

The [SO-101 reuse assessment](reference/so101/reuse-assessment.md) inspects an existing two-sided attachment arrangement and records a 0.40 mm horn-spacing discrepancy between its assembly model and the older Waveshare STEP. It is a reference, not a fitted mount.

Confirm the actual servo and included horns, the opposite-side support, screw engagement, connector access and cable motion, mating allowances, and the structural load path through the shoulder. Neither the Ø6 feature nor a torque rating gives an allowable bending, radial or axial load. Use measured hardware or an agreed supplier drawing for release-critical interfaces.

The C018 sheet lists 4–14 V and 180 mA no-load current; the selected Waveshare documentation lists 6–12.6 V and 200 mA. Both support 12 V nominal. Retain the narrower supplier range until the delivered variant is confirmed. The C018 sheet's older neutral/resolution wording also does not replace firmware readback and calibration.

Primary online sources: [FEETECH exact-model page](https://www.feetechrc.com/en/525603), [C018 specification](https://www.feetechrc.com/Data/feetechrc/upload/file/20240507/6385067068652648096680943.pdf), [Waveshare resource page](https://docs.waveshare.com/ST3215_Servo/Resources-And-Documents), and [Waveshare screw guidance](https://docs.waveshare.com/ST3215_Servo/FAQ). Source files and hashes are retained in [mechanical-sources.json](reference/mechanical-sources.json).
