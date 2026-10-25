"""Rigid microphone-plane targets from a captured, read-only head-model snapshot.

Only standard Python is needed. These targets are acoustic openings, not package
origins or PCB edges. No robot simulation file is written or loaded for motion.
"""
from pathlib import Path
import csv,json,math

HEAD=Path(__file__).resolve().parents[1]/'head'
TUNING='https://www.xmos.com/documentation/XM-014888-PC/html/modules/fwk_xvf/doc/user_guide/04_tuning_the_application.html'


def calculate(spec):
    mics=spec['mic_order'];p=[m['target_head_mm'] for m in mics]
    front,back,left,right=p
    close=lambda a,b:math.isclose(a,b,abs_tol=1e-6)
    # The accepted head has left/right symmetry. Refuse a changed geometry
    # rather than silently applying this reduced two-dimensional plane fit.
    assert [m['location'] for m in mics]==['front','back','left','right']
    assert close(front[0],0) and close(back[0],0)
    assert close(left[0],-right[0]) and close(left[1],right[1]) and close(left[2],right[2])
    origin=[sum(v[i] for v in p)/4 for i in range(3)]
    yy=sum((v[1]-origin[1])**2 for v in p)
    yz=sum((v[1]-origin[1])*(v[2]-origin[2]) for v in p)
    zz=sum((v[2]-origin[2])**2 for v in p)
    # Minor principal component of the symmetric Y/Z scatter matrix is the
    # least-squares normal. +Y in the array is front/up, +X is robot right.
    tilt=.5*math.atan2(-2*yz,yy-zz)
    sin,cos=math.sin(tilt),math.cos(tilt)
    axes=[[-1.,0.,0.],[0.,-cos,sin],[0.,sin,cos]]
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    rows=[]
    for m,v in zip(mics,p):
        relative=[v[i]-origin[i] for i in range(3)]
        x,y,residual=[dot(relative,a) for a in axes]
        projected=[origin[i]+x*axes[0][i]+y*axes[1][i] for i in range(3)]
        row=dict(m)
        row.update({'array_plane_mm':[x,y,0.], 'kicad_acoustic_target_mm':[x,-y],
                    'signed_normal_residual_mm':residual,'projected_head_mm':projected})
        rows.append(row)
    distance=lambda a,b:math.dist(rows[a]['array_plane_mm'],rows[b]['array_plane_mm'])
    maximum=max(abs(m['signed_normal_residual_mm']) for m in rows)
    assert maximum<=spec['owner_dimensions']['max_plane_residual_mm']
    return {'geometry_id':spec['geometry_id'],'owner_frozen':spec.get('owner_frozen',False),'release':False,'hardware_validated':False,
            'source':spec['source'],'plane_origin_head_mm':origin,
            'array_axes_in_head_frame':dict(zip(['x_right','y_front','z_normal'],axes)),
            'tilt_front_up_deg':math.degrees(tilt),'maximum_normal_residual_mm':maximum,
            'front_back_head_horizontal_mm':math.dist(front[:2],back[:2]),
            'front_above_back_mm':front[2]-back[2],
            'front_back_board_plane_mm':distance(0,1),'left_right_board_plane_mm':distance(2,3),
            'array_frame':'right-handed: +X robot right, +Y robot front/up along plane, +Z outward normal; origin at target centroid',
            'kicad_frame':'acoustic target X=array X, Y=-array Y; origin local and arbitrary until layout; no PCB plane height specified',
            'firmware_status':'generated geometry fragment only; logical MIC0..3 must be mapped to stated physical PDM inputs; no firmware built or flashed',
            'microphones':rows,'release_limits':spec['release_limits']}


def generate(directory=HEAD):
    spec=json.loads((directory/'microphone-targets.json').read_text())
    geo=calculate(spec)
    (directory/'microphone-geometry.json').write_text(json.dumps(geo,indent=2)+'\n')
    rows=[]
    for m in geo['microphones']:
        x,y,z=m['array_plane_mm']
        rows.append({'reference':m['reference'],'location':m['location'],'logical_mic':m['logical_index'],
                     'pdm_net':m['pdm_net'],'xvf3800_pin':m['xvf3800_pin'],
                     'acoustic_x_mm':x,'acoustic_y_mm':y,'acoustic_z_mm':z,
                     'kicad_target_x_mm':x,'kicad_target_y_mm':-y})
    with (directory/'microphone-coordinates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# GENERATED from microphone-targets.json. Geometry fragment, NOT a complete product configuration.',
           '# Merge this block into the custom product mic_geometries.yaml; preserve all other required settings.',
           '# Units: metres. Board-plane acoustic centers, not unprojected shell markers.',
           '# Build logical MIC0..3 to match the physical input order below. Verify raw-channel mapping on hardware.',
           '# Documentation: '+TUNING,'SQUARECULAR_GEOMETRY:']
    for m in geo['microphones']:
        x,y,z=[v/1000 for v in m['array_plane_mm']]
        lines.append('  - MIC'+str(m['logical_index'])+': ( '+', '.join(f'{v:.9f}f' for v in [x,y,z])+' ) # '+m['reference']+' '+m['location']+' '+m['pdm_net'])
    (directory/'mic_geometries.fragment.yaml').write_text('\n'.join(lines)+'\n')
    # All dimensional prose is generated from the same snapshot as the CAD
    # labels and firmware fragment, so a recapture cannot leave old numbers.
    rows=[]
    for m in geo['microphones']:
        x,y,z=m['array_plane_mm']
        rows.append(f"| {m['reference']} | {m['logical_index']} | {m['location']} | {m['pdm_net']} | {m['xvf3800_pin']} | {x:.6f} | {y:.6f} | {z:.6f} |")
    notes=f'''# Head-A microphone locations — frozen snapshot

Generated from `microphone-targets.json` by `tools/head_geometry.py`. Do not edit dimensional values here;
explicitly recapture the owner-frozen points with `tools/capture_head_targets.py`, then rebuild.

The front and back targets each moved 2 mm inward in the owner's frozen revision. A0.2 geometry is superseded.
The current source targets match the frozen table in `PCB-BRIEF.md` within its 0.01 mm rounding. Source hashes
now identify the new CURRENT scene, head generator, brief and existing mechanical fit study. No model or fit-study
source was changed. Source mic points are CAD markers, not measurements of fabricated hardware.

The horizontal front/back span is **{geo['front_back_head_horizontal_mm']:.6f} mm**, lateral span
**{geo['left_right_board_plane_mm']:.6f} mm**, and front-to-back height difference **{geo['front_above_back_mm']:.6f} mm**.
The orthogonal least-squares plane tilts **{geo['tilt_front_up_deg']:.6f} degrees** front-up. Maximum normal
departure is **{geo['maximum_normal_residual_mm']:.6f} mm**, within the frozen {spec['owner_dimensions']['max_plane_residual_mm']:.2f} mm allowance.
The front/back distance on this plane is **{geo['front_back_board_plane_mm']:.6f} mm**.

These are **acoustic opening targets**, not package centers, pin 1 or PCB copper-surface coordinates.
The raw shell targets remain frozen in `microphone-targets.json`; the table below contains their projections
onto the shared acoustic plane. The JSON includes source points, projected points, residuals and the head transform.

| Ref | Required logical index | Location | PDM net | XVF3800 pin | Array X mm | Array Y mm | Array Z mm |
|---|---:|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

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
'''
    (directory/'microphone-geometry.md').write_text(notes)
    return geo


if __name__=='__main__':
    g=generate()
    print(f"Head geometry: {g['tilt_front_up_deg']:.6f} deg tilt; {g['maximum_normal_residual_mm']:.6f} mm maximum residual; {g['front_back_board_plane_mm']:.6f} mm front/back on plane. No hardware validation.")
