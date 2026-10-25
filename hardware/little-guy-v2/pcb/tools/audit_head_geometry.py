"""Check head-plane targets, firmware units and physical-to-logical microphone order."""
from pathlib import Path
import json,math,re,xml.etree.ElementTree as ET
from schematic import REVISION

ROOT=Path(__file__).resolve().parents[1]
head=ROOT/'head'
g=json.loads((head/'microphone-geometry.json').read_text())
spec=json.loads((head/'microphone-targets.json').read_text())
checks=[]
def require(name,ok):
    checks.append({'requirement':name,'passed':bool(ok)})
    if not ok:raise AssertionError(name)
dot=lambda a,b:sum(x*y for x,y in zip(a,b))
axes=list(g['array_axes_in_head_frame'].values())
require('Array axes are orthonormal',all(abs(dot(a,b)-(1 if i==j else 0))<1e-10 for i,a in enumerate(axes) for j,b in enumerate(axes)))
x,y,z=axes
cross=[x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]]
require('Array frame is right-handed with outward normal',math.dist(cross,z)<1e-10 and z[2]>0)
require('Front/back horizontal and left/right spans match owner input',
        abs(g['front_back_head_horizontal_mm']-spec['owner_dimensions']['front_back_horizontal_mm'])<1e-6 and abs(g['left_right_board_plane_mm']-spec['owner_dimensions']['left_right_mm'])<1e-6)
require('Tilt and plane error agree with supplied approximate dimensions',
        abs(g['tilt_front_up_deg']-spec['owner_dimensions']['plane_tilt_approx_deg'])<.5 and g['maximum_normal_residual_mm']<=spec['owner_dimensions']['max_plane_residual_mm'])
require('Acoustic target snapshot records owner-frozen points',spec.get('owner_frozen') is True)
require('Front/back along the tilted plane preserves the 3D baseline',
        abs(g['front_back_board_plane_mm']-math.dist(spec['mic_order'][0]['target_head_mm'],spec['mic_order'][1]['target_head_mm']))<.001)
for m in g['microphones']:
    r=m['reference'];source=m['target_head_mm'];projected=m['projected_head_mm'];error=m['signed_normal_residual_mm']
    require(r+' projection and residual reconstruct the CAD target',
            math.dist([projected[i]+z[i]*error for i in range(3)],source)<1e-9)
    require(r+' acoustic target lies on the common plane and uses KiCad Y flip',
            m['array_plane_mm'][2]==0 and m['kicad_acoustic_target_mm']==[m['array_plane_mm'][0],-m['array_plane_mm'][1]])
    require(r+' captured CAD point agrees with the frozen brief rounding',
            all(abs(a-b)<=spec['frozen_rounding_tolerance_mm'] for a,b in zip(source,spec['frozen_reference_mm'][m['location']])))
text=(head/'mic_geometries.fragment.yaml').read_text()
values={int(i):[float(x.strip().removesuffix('f')) for x in triple.split(',')]
        for i,triple in re.findall(r'MIC(\d):\s*\(([^)]*)\)',text)}
require('XMOS fragment contains exactly four XYZ metre entries',len(values)==4 and all(len(v)==3 for v in values.values()))
require('XMOS fragment preserves planned geometry to sub-micron rounding',
        all(math.dist(values[m['logical_index']],[v/1000 for v in m['array_plane_mm']])<1e-9 for m in g['microphones']))
tree=ET.parse(head/'head-netlist.xml')
pn={(n.get('ref'),n.get('pin')):net.get('name') for net in tree.findall('./nets/net') for n in net.findall('node')}
for m in g['microphones']:
    require(m['reference']+' physical data net matches declared logical mic and XMOS pin',
            pn.get((m['reference'],'1'))==pn.get(('U101',str(m['xvf3800_pin'])))==m['pdm_net']==f"PDM_DATA{m['logical_index']}")
report={'revision':REVISION,'geometry_id':g['geometry_id'],'hardware_validated':False,'firmware_built':False,
        'fabrication_released':False,'checks':checks,'maximum_normal_residual_mm':g['maximum_normal_residual_mm'],
        'tilt_front_up_deg':g['tilt_front_up_deg'],'front_back_board_plane_mm':g['front_back_board_plane_mm'],
        'limits':'Checks target math, units and captured nets; does not validate acoustic performance, package origins, enclosure fit, mounts or firmware channel mapping on hardware.'}
(head/'geometry-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(checks)} head-geometry / channel-map requirements passed. No hardware or firmware validation.')
