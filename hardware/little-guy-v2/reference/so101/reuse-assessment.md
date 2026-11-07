# SO-101 joint geometry reuse assessment

Reviewed 2026-10-21. These are unchanged third-party reference parts, not Little Guy production mounts.

![Original reference parts](parts-review.png)

The [TheRobotStudio SO-ARM100 repository](https://github.com/TheRobotStudio/SO-ARM100/tree/a758567c3978dfeefe282ede0500085a48fe8f78) provides an existing STS3215 mechanism with attachments on both sides of the shoulder and elbow joints. Its [SO-101 assembly guide](https://huggingface.co/docs/lerobot/main/en/assemble_so101) describes the horn attachments and fastening sequence. This provides a useful example of carrying loads through two side plates instead of hanging the arm from a single plate. It does not establish allowable side loads or fatigue life for Little Guy.

The original [README.md](README.md), [LICENSE](LICENSE) and four STEP files are retained at commit `a758567c3978dfeefe282ede0500085a48fe8f78`. [provenance.json](provenance.json) records their URLs and SHA-256 hashes. The source is Apache-2.0 licensed; no NOTICE file was found in the pinned repository tree. This assessment and the inspection/rendering scripts are separate from the original files.

## Geometry actually checked

Three individual parts each import as one valid solid. Their source-coordinate bounds are:

| Original part | X × Y × Z bounds, mm |
|---|---|
| Motor_holder_SO101_Base | 28.000 × 38.097 × 55.080 |
| Rotation_Pitch_SO101 | 59.800 × 84.200 × 46.000 |
| Upper_arm_SO101 | 142.170 × 24.500 × 67.300 |

These coordinates are not Little Guy's layout coordinates. [inspection.json](inspection.json) contains the individual bounds, validity and circular-edge observations. [assembly-inspection.json](assembly-inspection.json) separately retains the named assembly components and their placements; the assembly's slightly rotated poses mean its world-aligned bounds cannot be substituted for a part's own dimensions.

For five servo components, four coaxial radius-9.6 mm circles define the two round fittings. The greatest centre separation is **36.85 mm**. The fifth servo component has only one fitting represented: its two circles are **2.1 mm** apart, which measures that fitting's thickness, not a two-sided spacing. The separately supplied Waveshare servo STEP has a **37.25 mm** outer-face separation. The **0.40 mm discrepancy** is large enough that these reference interfaces must not be mixed without resolving the actual horns and support hardware.

The embedded servo has SCS215 naming and modified horn geometry. Neither this assembly nor the older Waveshare model proves fit for the planned C018 servo and purchased accessory kit. Whole-assembly validity, interference, fastener engagement and printed strength were not checked. Kernel validity of the three individual solids is not a mechanical strength result.

## Use for Little Guy

Retain the two-sided attachment arrangement as a design reference. Do not yet copy its mating width, assume its opposite-side fitting is a standard bearing, or substitute its screws for those in the C018 drawing. The [servo interface comparison](../../servo-interface.md) records the different driven and passive-side screws.

Before adapting the shoulder, confirm the supplier's exact servo/horn kit, passive support, axial stack and screw engagement. Then locate the parts around the [screened shoulder axes](../../layout/shoulder/README.md), provide the torso opening and cable path, and recalculate moving mass and static/inertial loads with the actual bracket geometry. The current shoulder spacing sample has only about 0.1 mm beyond its chosen combined padding at rest; it does not contain a real bracket or a manufacturing tolerance allocation. No added bracket mass has been assumed away by this reference review.

Reproduce the inspection and preview using the existing CAD environment and plotting dependencies:

```sh
python hardware/little-guy-v2/analysis/inspect_so101.py --mesh-json .run/so101-review/meshes.json
python hardware/little-guy-v2/analysis/render_so101.py .run/so101-review/meshes.json
```
