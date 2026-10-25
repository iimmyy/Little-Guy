"""Check selected electrical requirements against KiCad's exported XML netlist.

These checks establish connectivity and nominal arithmetic, not electrical safety,
loop stability, current capacity, thermal performance or physical validation.
"""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]/'power'
tree=ET.parse(root/'power-netlist.xml')
net_nodes={net.attrib['name']:{(node.attrib['ref'],node.attrib['pin']) for node in net.findall('node')}
           for net in tree.findall('./nets/net')}
pin_net={node:name for name,nodes in net_nodes.items() for node in nodes}
checks=[]

def require(name,condition):
    checks.append({'requirement':name,'passed':bool(condition)})
    if not condition:
        raise AssertionError(name)

components=tree.findall('./components/comp')
refs=[comp.attrib['ref'] for comp in components]
require('Component references are unique',len(refs)==len(set(refs)))
require('Motor battery feed is separate from charger SYS',pin_net['U201','22']!=pin_net['U201','25'])
require('Motor bridge input draws from protected pack',pin_net['Q501','5']=='PACK_PROTECTED')
require('Jetson regulator draws from charger logic SYS',pin_net['U701','7']=='LOGIC_SYS')
require('Hands draw from protected pack, bypassing internal charger BATFET',
        pin_net['U801','2']=='PACK_PROTECTED' and pin_net['U901','2']=='PACK_PROTECTED')
require('Peripheral regulator draws from protected pack',pin_net['U1401','2']=='PACK_PROTECTED')
require('Charge-enable gate drives charger CE pin',pin_net['U303','4']==pin_net['U201','13']=='CHG_CE_N')
require('Independent temperature permission reaches charge gate',pin_net['U303','2']=='PACK_TEMP_OK')
require('Manual cut and supply supervisor clear motor latch',
        pin_net['U1001','1']==pin_net['U1003','2']==pin_net['J1001','2']=='MOTOR_CLEAR_N')
require('Motor arm is edge-clocked rather than tied on',pin_net['U1001','3']=='MOTOR_ARM')
require('Jetson feed is outside motor-arm gate outputs',
        pin_net['U701','1']=='JETSON_EN' and all(pin_net['U1002',p]!='JETSON_EN' for p in ['3','6','8','11']))
require('All six temperature comparators can inhibit charging',
        all(pin_net[ref,p]=='PACK_TEMP_OK' for ref,p in [('U1201','1'),('U1201','2'),('U1201','13'),('U1201','14'),('U1202','1'),('U1202','2')]))
require('Pack shunt separates pack source from every protected load',
        pin_net['J203','1']==pin_net['R1101','1']=='PACK_FROM_BMS' and pin_net['R1101','2']=='PACK_PROTECTED')
require('INA228 polarity observes discharge as positive',pin_net['R1102','1']=='PACK_FROM_BMS' and pin_net['R1103','1']=='PACK_PROTECTED')
require('EEPROM and charger share only the PD controller bus',
        pin_net['U102','5']==pin_net['U201','15']==pin_net['U101','16']=='PD_SDA' and pin_net['J103','4']=='HOST_SDA')

# Values from Semitec AT manufacturer resistance table, units ohms.
rt={0:27280,10:17960,20:12090,25:10000,30:8313,40:5827,50:4160,60:3020}
hot_ratio=10000/(19100+10000)
cold_ratio=26100/(10000+26100)
def temp_ok(resistance):
    ratio=1 if math.isinf(resistance) else resistance/(10000+resistance)
    return hot_ratio<ratio<cold_ratio
for name,resistance,expected in [('cold at 0 C',rt[0],False),('room at 25 C',rt[25],True),
                                 ('hot at 50 C',rt[50],False),('sensor open',math.inf,False),('sensor short',0,False)]:
    require('Nominal temperature window: '+name,temp_ok(resistance)==expected)

summary={'electrically_validated':False,'checks':checks,'component_count':len(refs),
         'net_count':len(net_nodes),'nominal_motor_v':.8*(1+262000/20000),
         'nominal_hand_v':1*(1+100000/21300),'nominal_jetson_v':1.2*(1+100000/11100),
         'nominal_peripheral_v':1*(1+100000/24900),
         'remaining':['exact branch fuses and protection coordination','regeneration protection','pack BMS and cells',
                      'exact MPNs and land patterns','PD configuration and controller firmware',
                      'physical bench tests','layout','manufacturing checks']}
(root/'connectivity-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
# CLI reports contain local absolute paths; retain project-relative source names only.
for element in tree.iter():
    if element.text and str(root) in element.text:
        element.text=element.text.replace(str(root)+'/','')
tree.write(root/'power-netlist.xml',encoding='utf-8',xml_declaration=True)
ercpath=root/'erc-A0.json'
erc=json.loads(ercpath.read_text())
erc['source']=Path(erc['source']).name
ercpath.write_text(json.dumps(erc,indent=2)+'\n')
print(f'{len(checks)} requirements passed on exported netlist and nominal temperature arithmetic.')
print('Physical/electrical validation remains false.')
