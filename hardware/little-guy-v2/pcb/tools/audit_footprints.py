"""Check numbered pads against schematic pins; does not qualify package dimensions or hardware."""
from pathlib import Path
import argparse,json,xml.etree.ElementTree as ET
import sexpdata as sx
from schematic import REVISION

ROOT=Path(__file__).resolve().parents[1]
S=sx.Symbol
def children(node,tag):return [v for v in node if isinstance(v,list) and v and v[0]==S(tag)]
parser=argparse.ArgumentParser()
parser.add_argument('--kicad-footprints',type=Path,required=True,help='Installed KiCad .pretty library parent; never written into reports')
args=parser.parse_args()
cache={};checks=[];pending=[];assigned=0;failed=[]
def require(name,condition):
    checks.append({'requirement':name,'passed':bool(condition)})
    if not condition:failed.append(name)
for board in ['power','spinal','foot','head']:
    manifest=json.loads((ROOT/board/'circuit.json').read_text())
    for part in manifest['parts']:
        if part.get('annotation'):continue
        fp=part['footprint']
        if not fp:
            pending.append({'board':board,'ref':part['ref'],'mpn':part['mpn']});continue
        lib,name=fp.split(':',1)
        path=((ROOT/board) if lib=='LGV2_Footprints' else args.kicad_footprints)/(lib+'.pretty')/(name+'.kicad_mod')
        if path not in cache:
            if path.exists():
                raw=sx.loads(path.read_text());pads=children(raw,'pad')
                cache[path]=(raw,{str(p[1]) for p in pads if str(p[1])})
            else:cache[path]=(None,set())
        raw,numbers=cache[path];assigned+=1
        require(board+'/'+part['ref']+' footprint exists and numbered pads exactly match schematic pins',raw is not None and numbers==set(part['pins']))
    if board=='spinal':
        for ref in ['U601','U701','U801','U901']:
            p=next(p for p in manifest['parts'] if p['ref']==ref)
            require(ref+' MPR121 has exactly pins 1-20 and no invented exposed pad',set(p['pins'])=={str(i) for i in range(1,21)})

# Independent polarity anchor: D_SMC pad 1 is on its cathode-marked side.
tree=ET.parse(ROOT/'power/power-netlist.xml')
pin_net={(n.get('ref'),n.get('pin')):net.get('name') for net in tree.findall('./nets/net') for n in net.findall('node')}
require('Always-on catch diode cathode pad 1 is on switch node and anode pad 2 is on ground',
        pin_net.get(('D301','1'))=='AON_SW' and pin_net.get(('D301','2'))=='GND')
fpath=ROOT/'spinal/LGV2_Footprints.pretty/Teensy41_Socket_GPIO_ID.kicad_mod'
fp=sx.loads(fpath.read_text());pads=children(fp,'pad')
coords={str(p[1]):tuple(children(p,'at')[0][1:3]) for p in pads}
anchors={'VIN':(-29.21,-7.62),'GPIO0':(-26.67,7.62),'GPIO13':(3.81,-7.62),
         'GPIO23':(-21.59,-7.62),'GPIO33':(29.21,-7.62),'GPIO32':(29.21,7.62)}
require('Teensy socket has 48 distinct through-hole pads',len(pads)==len(coords)==48 and all(p[2]==S('thru_hole') for p in pads))
require('Teensy socket key GPIO locations match PJRC top view with USB at negative X',all(coords.get(k)==v for k,v in anchors.items()))
require('Teensy rows are 15.24 mm apart with 2.54 mm pitch',
        sorted({y for x,y in coords.values()})==[-7.62,7.62] and
        sorted({round(x,4) for x,y in coords.values()})==[round(-29.21+2.54*i,4) for i in range(24)])
fp=sx.loads((ROOT/'head/LGV2_Footprints.pretty/XMOS_QF60B_7x7mm_P0.4mm_5Paddle.kicad_mod').read_text())
pads={str(p[1]):p for p in children(fp,'pad') if str(p[1])}
xy=lambda pin:tuple(children(pads[str(pin)],'at')[0][1:3])
size=lambda pin:tuple(children(pads[str(pin)],'size')[0][1:3])
require('XMOS top-view perimeter numbering matches Fig8.1 bottom-view mirror',
        all(xy(p)==v for p,v in {1:(-3.35,-2.8),15:(-3.35,2.8),16:(-2.8,3.35),30:(2.8,3.35),31:(3.35,2.8),45:(3.35,-2.8),46:(2.8,-3.35),60:(-2.8,-3.35)}.items()))
require('XMOS core paddles are left 61, bottom 62, right 63, top 64 in PCB top view',
        all(xy(p)==v for p,v in {61:(-2.5,0),62:(0,2.5),63:(2.5,0),64:(0,-2.5),65:(0,0)}.items()))
require('XMOS four core paddles match nominal 3.60 by 0.60 mm geometry',
        all(size(p)==v for p,v in {61:(.6,3.6),62:(3.6,.6),63:(.6,3.6),64:(3.6,.6)}.items()))
tree=ET.parse(ROOT/'head/head-netlist.xml')
pn={(n.get('ref'),n.get('pin')):net.get('name') for net in tree.findall('./nets/net') for n in net.findall('node')}
require('XMOS footprint paddle nets preserve four 0.9 V paddles and one ground paddle',
        all(pn.get(('U101',str(p)))=='HEAD_0V9' for p in range(61,65)) and pn.get(('U101','65'))=='GND')
report={'revision':REVISION,'hardware_validated':False,'fabrication_released':False,'scope':'Pad identifiers, one diode polarity anchor, Teensy nominal 2D geometry and XMOS paddle geometry/orientation. Other package dimensions, ratings, assembly, stencil and physical fit remain open.',
        'assigned_components_checked':assigned,'unique_footprint_files_read':len(cache),
        'unassigned_components':pending,'checks':checks,'failed_requirements':failed}
(ROOT/'footprint-audit-A0.json').write_text(json.dumps(report,indent=2)+'\n')
print(str(assigned)+' assigned footprints checked; '+str(len(pending))+' assignments still pending; '+str(len(failed))+' failed requirements.')
if failed:raise SystemExit('\n'.join(failed))
