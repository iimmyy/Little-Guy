"""Explicit read-only recapture from CURRENT.xml, checked against the frozen brief.

Run this deliberately when the owner changes the frozen mic locations. Normal
schematic builds use the saved snapshot and never follow a changing robot model.
"""
from pathlib import Path
import argparse,hashlib,json,math,re,xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[4]


def capture(repo,scene_path,fit_path):
    head=repo/'hardware/little-guy-v2/pcb/head'
    brief=repo/'hardware/little-guy-v2/PCB-BRIEF.md'
    block=brief.read_text().split('Frozen mic locations',1)[1]
    table={name.removeprefix('robot '):[float(x),float(y),float(z)] for name,x,y,z in
           re.findall(r'^\s*\|\s*(front|back|robot left|robot right)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|',block,re.M)}
    if set(table)!={'front','back','left','right'}:raise ValueError('Frozen brief table must contain all four named mic points')
    scene=repo/scene_path;data=scene.read_bytes();tree=ET.fromstring(data)
    groups=[]
    for body in tree.iter('body'):
        markers=[g for g in body.findall('geom') if g.get('rgba','').split()==['0.25','0.25','0.27','1']]
        if markers:groups.append((body,markers))
    if len(groups)!=1 or len(groups[0][1])!=4:raise ValueError('Expected four markers in one head-local body')
    body,markers=groups[0]
    if not any(g.get('mesh')=='i1_shellI1' for g in body.findall('geom')):raise ValueError('Mic body does not contain expected head shell')
    points=[[float(v)*1000 for v in g.get('pos').split()] for g in markers]
    by_location={'front':min(points,key=lambda p:p[1]),'back':max(points,key=lambda p:p[1]),
                 'left':max(points,key=lambda p:p[0]),'right':min(points,key=lambda p:p[0])}
    for name,point in by_location.items():
        if any(abs(a-b)>.0050001 for a,b in zip(point,table[name])):
            raise ValueError(name+' CURRENT marker disagrees with frozen brief table beyond its 0.01 mm rounding')
    fit=repo/fit_path;fit_data=json.loads(fit.read_text())
    if fit_data['mics']!=table:raise ValueError('Fit report mic points do not match frozen brief')
    current=json.loads((head/'microphone-targets.json').read_text())
    old_id=current['geometry_id']
    current.update({'geometry_id':'head-A-diamond-frozen-2026-10-25-v1',
                    'status':'owner frozen locations; recaptured from CURRENT markers; no hardware validation',
                    'owner_frozen':True,'frozen_reference_mm':table,'frozen_rounding_tolerance_mm':.0050001})
    if old_id!=current['geometry_id']:current['supersedes_geometry_id']=old_id
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    current['source'].update({'scene':str(scene_path),'scene_sha256':hashlib.sha256(data).hexdigest(),
         'generator_sha256':digest(repo/current['source']['generator']),
         'brief':str(brief.relative_to(repo)),'brief_sha256':digest(brief),
         'fit_check_script':'tools/body/mic_diamond.py','fit_check_script_sha256':digest(repo/'tools/body/mic_diamond.py'),
         'fit_report':str(fit_path),'fit_report_sha256':digest(fit),
         'selector':'four same-body mic marker centers with rgba 0.25 0.25 0.27 1, in i1_shellI1 mesh frame',
         'read_only':True})
    current['owner_dimensions']={'front_back_horizontal_mm':math.dist(table['front'][:2],table['back'][:2]),
         'left_right_mm':math.dist(table['left'][:2],table['right'][:2]),
         'front_above_back_approx_mm':table['front'][2]-table['back'][2],
         'front_back_plane_approx_mm':float(re.search(r'along the plane\s+([\d.]+)\s*mm',block).group(1)),
         'plane_tilt_approx_deg':float(re.search(r'([\d.]+)\s*degrees front-up',block).group(1)),
         'max_plane_residual_mm':float(re.search(r'every mic within\s+([\d.]+)\s*mm',block).group(1))}
    for m in current['mic_order']:m['target_head_mm']=by_location[m['location']]
    (head/'microphone-targets.json').write_text(json.dumps(current,indent=2)+'\n')
    reference={'status':'existing CAD fit study reference; not physical validation or fabrication release',
               'source':str(fit_path),'source_sha256':digest(fit),'reported_result':fit_data,
               'limits':'Sampled clearance study with assumed wall, gasket, PCB and mic envelope. Its trial outline is not a released PCB perimeter.'}
    (head/'microphone-fit-reference.json').write_text(json.dumps(reference,indent=2)+'\n')
    print('Recaptured frozen mic points, current source hashes and fit-study reference. No robot or fit-study file modified.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,default=REPO)
    p.add_argument('--scene',type=Path,default=Path('.run/results/mind/body-v2/CURRENT.xml'))
    p.add_argument('--fit-report',type=Path,default=Path('.run/results/mind/body-v2/mic-diamond-1025b/mic-diamond.json'))
    args=p.parse_args();capture(args.repo_root,args.scene,args.fit_report)
