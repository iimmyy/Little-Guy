"""Requirement checks on exported KiCad netlists, including inter-board pin orders."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
from schematic import REVISION

ROOT=Path(__file__).resolve().parents[1]
trees={b:ET.parse(ROOT/b/(b+'-netlist.xml')) for b in ['power','spinal','foot','head']}
pins={b:{(node.get('ref'),node.get('pin')):net.get('name') for net in t.findall('./nets/net') for node in net.findall('node')}
      for b,t in trees.items()}
checks=[]
def require(name,condition):
    checks.append({'requirement':name,'passed':bool(condition)})
    if not condition:raise AssertionError(name)
def net(b,ref,pin):return pins[b].get((ref,str(pin)))
def nc(b,ref,pin):
    for n in trees[b].findall('./nets/net'):
        nodes=n.findall('node')
        if any(x.get('ref')==ref and x.get('pin')==str(pin) for x in nodes):
            return len(nodes)==1 and nodes[0].get('pintype')=='no_connect'
    return False
def port(b,ref):
    p={int(pin):value for (r,pin),value in pins[b].items() if r==ref}
    return [p[i] for i in range(1,len(p)+1)]
require('Charging harness matches power and spinal pin for pin',port('power','J103')==port('spinal','J101'))
power_control=port('power','J1002');spinal_control=port('spinal','J102')
spinal_control=[{'WATCHDOG_FAULT_N':'MOTOR_CLEAR_N'}.get(n,n) for n in spinal_control]
require('Motor and logic control harness matches all 12 pins',power_control==spinal_control)
require('External watchdog can clear power motor latch',net('spinal','U502',7)=='WATCHDOG_FAULT_N' and net('power','J1002',9)==net('power','U1001',1))
require('Watchdog is enabled without a firmware enable signal',net('spinal','U502',3)==net('spinal','U502',5)=='MCU_3V3')
for name,tx,rx,gpios in [('LEG',201,202,[0,1]),('UPPER',241,242,[7,8]),('LEFT',301,302,[15,14]),('RIGHT',341,342,[16,17])]:
    require(name+' servo transmits through a 5 V buffer',net('spinal','U'+str(tx),5)=='AON_5V')
    require(name+' servo receives through a 3.3 V buffer',net('spinal','U'+str(rx),5)=='MCU_3V3')
    require(name+' UART GPIO mapping matches PJRC Serial pin table',
            net('spinal','U101','GPIO'+str(gpios[0]))==name+'_RX' and net('spinal','U101','GPIO'+str(gpios[1]))==name+'_TX')
for k in range(4):
    ref='U'+str(601+k*100)
    require('Touch '+str(k+1)+' has twelve separate electrode nets',len({net('spinal',ref,p) for p in range(8,20)})==12)
for name,tx,a,b,y in [('LEG',201,1,2,3),('UPPER',241,4,5,6),('LEFT',301,9,10,8),('RIGHT',341,12,13,11)]:
    require(name+' transmit enable requires hardware motor arm',
            net('spinal','U1101',a)==name+'_DE' and net('spinal','U1101',b)=='MOTOR_ARMED' and net('spinal','U1101',y)==net('spinal','U'+str(tx),1)==name+'_TX_ENABLE')
require('Four touch addresses are distinct and strapped per NXP table',
        [net('spinal','U'+str(601+k*100),4) for k in range(4)]==['GND','MCU_3V3','HOST_SDA','HOST_SCL'])
require('Foot ADC 0 is strapped to 0x40',net('foot','U201',1)==net('foot','U201',2)=='GND')
require('Foot ADC 1 is strapped to 0x41',net('foot','U301',1)=='BRIDGE_3V3' and net('foot','U301',2)=='GND')
require('Both foot references observe bridge excitation',
        all(net('foot',r,9)=='BRIDGE_3V3' and net('foot',r,8)=='GND' for r in ['U201','U301']))
require('Four cells map to four differential pairs',
        len({net('foot',r,p) for r in ['U201','U301'] for p in [6,7,10,11]})==8)
require('Toe do-not-connect pad is not wired',nc('foot','U103',8))
for k,side in enumerate(['LEFT','RIGHT']):
    physical=port('spinal','J'+str(401+k))
    mapping={side+'_FOOT_5V':'FOOT_5V','SENSOR'+str(k)+'_SCL':'FOOT_SCL','SENSOR'+str(k)+'_SDA':'FOOT_SDA',
             side+'_READY_N':'FOOT_READY_N',side+'_XSHUT':'TOE_XSHUT'}
    require(side+' foot six-wire harness matches the foot PCB', [mapping.get(n,n) for n in physical]==port('foot','J101'))
require('XVF3800 four core paddles are on 0.9 V',all(net('head','U101',p)=='HEAD_0V9' for p in range(61,65)))
require('XVF3800 only ground paddle is pin 65',net('head','U101',65)=='GND')
require('XVF3800 NC pin 25 is not wired',nc('head','U101',25))
require('XVF3800 MCLK output is connected to its clock input',net('head','U101',14)==net('head','U101',41)=='AUDIO_MCLK')
require('Head core enable requires both I/O supervisors',net('head','U304',1)=='HEAD_IO18_OK' and net('head','U304',2)=='HEAD_IO33_OK' and net('head','U304',3)==net('head','U203',1)=='HEAD_CORE_ENABLE')
require('Head reset also requires core supervisor',net('head','U304',4)=='HEAD_CORE_ENABLE' and net('head','U304',5)=='HEAD_CORE_OK' and net('head','U304',6)==net('head','U101',21)=='HEAD_RESET_N')
require('All supervisors assert reset independently of core firmware',all(net('head','U'+str(301+k),6)=='HEAD_USB_5V' for k in range(3)))
require('Four PDM data lines connect to their assigned processor inputs',all(net('head','MK'+str(501+k),1)==net('head','U101',p) for k,p in enumerate([46,50,48,51])))

summary={'revision':REVISION,'hardware_validated':False,'fabrication_released':False,'checks':checks,'boards':{}}
for b,t in trees.items():
    manifest=json.loads((ROOT/b/'circuit.json').read_text())
    physical=[p for p in manifest['parts'] if not p.get('annotation')]
    mismatches=[]
    for part in physical:
        for pin,expected in part['pins'].items():
            actual=net(b,part['ref'],pin)
            correct=(actual is None or actual.startswith('unconnected-')) if expected is None else actual==expected
            if not correct:mismatches.append((part['ref'],pin,expected,actual))
    require(b+' every captured pin matches the exported schematic connectivity',not mismatches)
    erc=json.loads((ROOT/b/'erc-A0.json').read_text())
    violations=[v for s in erc['sheets'] for v in s['violations']]
    errors=[v for v in violations if v['severity']=='error']
    require(b+' exports without ERC errors',not errors)
    summary['boards'][b]={'circuit_sheets':len({p['sheet'] for p in physical}),'physical_components':len(physical),
                         'quantity_in_robot':2 if b=='foot' else 1,'mpn_pending':sum(not p['mpn'] for p in physical),
                         'footprint_pending':sum(not p['footprint'] for p in physical),
                         'erc_errors':len(errors),'erc_warnings':len(violations)-len(errors)}
    # Keep exported evidence public-repository friendly.
    for element in t.iter():
        if element.text and str(ROOT/b) in element.text:element.text=element.text.replace(str(ROOT/b)+'/','')
    t.write(ROOT/b/(b+'-netlist.xml'),encoding='utf-8',xml_declaration=True)
    erc['source']=Path(erc['source']).name
    (ROOT/b/'erc-A0.json').write_text(json.dumps(erc,indent=2)+'\n')
(ROOT/'verification-A0.json').write_text(json.dumps(summary,indent=2)+'\n')
print(str(len(checks))+' connectivity / ERC requirements passed. Physical validation and fabrication release remain false.')
