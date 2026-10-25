"""Build the power-board charging and housekeeping circuit draft.

Run with Python and sexpdata from this directory. No installed-library paths or
personal machine paths are embedded in generated design files.
"""
from pathlib import Path
from schematic import Sheet, ic, save_project
from power_regulators import add_regulators
from power_hands import add_hands
from power_control import add_control
from power_temperature import add_temperature
from power_branches import add_branches

ROOT = Path(__file__).resolve().parents[1] / 'power'
R = 'Resistor_SMD:R_0603_1608Metric'
C = 'Capacitor_SMD:C_0603_1608Metric'
CB = 'Capacitor_SMD:C_1206_3216Metric'

def resistor(sheet, ref, value, a, b, x, y):
    sheet.passive('R', ref, value, a, b, x, y, R)

def capacitor(sheet, ref, value, a, x, y, b='GND', bulk=False):
    sheet.passive('C', ref, value, a, b, x, y, CB if bulk else C)

p = Sheet('01-usb-pd', 'USB-C PD / configuration', 2)
p.text('USB-C SINK ONLY / 20 V request, input current follows negotiated contract', 15, 15, 1.8)
p.text('Pin map: TPS25751D datasheet SLVSH93A Table 5-1. Configure all grounded GPIO as unused.\n64 KB EEPROM required; reference 32 KB device is too small for current patch bundles.\nFirmware image and USB-C protection component selection remain release tasks.', 15, 24)
p.connector('J101', 'Front-door USB-C power/CC interface', ['USB_VBUS', 'GND', 'USB_CC1', 'USB_CC2'], 45, 68)
p.text('Port connector, CC overvoltage/ESD protection and cable/harness are not released.\nNo external Rd resistors: PD controller implements the Type-C terminations.', 15, 97)
left = [(1, 'LDO_3V3', 'power_out'), (4, 'LDO_1V5', 'power_out'), (38, 'VIN_3V3', 'power_in'),
        (2, 'ADCIN1', 'input'), (3, 'ADCIN2', 'input'), (28, 'CC1', 'bidirectional'), (29, 'CC2', 'bidirectional'),
        (8, 'I2Ct_SDA', 'bidirectional'), (9, 'I2Ct_SCL', 'input'), (10, '~{I2Ct_IRQ}', 'open_collector'),
        (16, 'I2Cc_SDA', 'bidirectional'), (17, 'I2Cc_SCL', 'open_collector'), (18, '~{I2Cc_IRQ}', 'input')]
right = [(23, 'VBUS_IN', 'power_in'), (24, 'VBUS_IN', 'passive'), (25, 'VBUS_IN', 'passive'),
         (20, 'PPHV', 'power_out'), (21, 'PPHV', 'passive'), (22, 'PPHV', 'passive'),
         (32, 'VBUS', 'power_out'), (33, 'VBUS', 'passive'), (34, 'PP5V', 'power_in'), (35, 'PP5V', 'passive'),
         (15, 'DRAIN', 'passive'), (30, 'DRAIN', 'passive'), (40, 'DRAIN_EP', 'passive')]
right += [(i, name, 'bidirectional') for i, name in [(5,'GPIO0'),(6,'GPIO1'),(7,'GPIO2'),(19,'GPIO3'),(26,'GPIO4'),(27,'GPIO5'),(37,'GPIO6'),(36,'GPIO7'),(13,'GPIO11')]]
left += [(i, 'GND' if i != 39 else 'GND_EP', 'power_in' if i == 11 else 'passive') for i in [11,12,14,31,39]]
pdnets = {str(i): 'GND' for i, *_ in left + right}
pdnets.update({'1':'PD_3V3','4':'PD_1V5','38':'AON_3V3','2':'PD_1V5','3':'PD_1V5',
               '28':'USB_CC1','29':'USB_CC2','8':'HOST_SDA','9':'HOST_SCL','10':'PD_IRQ_N',
               '16':'PD_SDA','17':'PD_SCL','18':'CHG_INT_N','23':'USB_VBUS','24':'USB_VBUS','25':'USB_VBUS',
               '20':'USB_PD_IN','21':'USB_PD_IN','22':'USB_PD_IN','32':'PD_VBUS_5V','33':'PD_VBUS_5V',
               '34':'AON_5V','35':'AON_5V','15':'PD_DRAIN','30':'PD_DRAIN','40':'PD_DRAIN'})
p.part(ic('TPS25751D', left, right, 15.24), 'U101', 'TPS25751DREFR', 145, 93, pdnets,
       mpn='TPS25751DREFR', datasheet='https://www.ti.com/lit/ds/symlink/tps25751.pdf')
for j,(net,val) in enumerate([('AON_3V3','10u 10V'),('PD_3V3','10u 10V'),('PD_1V5','10u 10V'),
                              ('PD_VBUS_5V','4u7 10V'),('AON_5V','10u 10V'),('USB_PD_IN','22u 35V'),('USB_PD_IN','22u 35V')]):
    capacitor(p,'C'+str(101+j),val,net,235+(j%4)*40,55+(j//4)*35,bulk=True)
for j,net in enumerate(['USB_CC1','USB_CC2']):
    capacitor(p,'C'+str(108+j),'330p C0G 50V',net,245+j*45,125)
ee = ic('24LC512', [(1,'A0','input'),(2,'A1','input'),(3,'A2','input'),(4,'GND','power_in')],
        [(8,'VCC','power_in'),(7,'WP','input'),(6,'SCL','input'),(5,'SDA','bidirectional')], 10.16)
p.part(ee,'U102','24LC512-I/SN (64 KB)',85,180,
       {'1':'GND','2':'GND','3':'GND','4':'GND','5':'PD_SDA','6':'PD_SCL','7':'PD_EE_WP','8':'PD_3V3'},
       'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm','24LC512-I/SN','https://www.microchip.com/en-us/product/24lc512')
capacitor(p,'C110','100n 16V','PD_3V3',40,180)
for j,net in enumerate(['PD_SDA','PD_SCL']):
    resistor(p,'R'+str(101+j),'2k2 1%','PD_3V3',net,155+j*35,180)
resistor(p,'R103','10k 1%','PD_EE_WP','GND',235,180)
resistor(p,'R104','10k 1%','PD_3V3','CHG_INT_N',270,180)
for j,net in enumerate(['HOST_SDA','HOST_SCL','PD_IRQ_N']):
    resistor(p,'R'+str(105+j),'4k7 1%','AON_3V3',net,320+j*30,180)
p.connector('J102','PD EEPROM programmer / no second controller during use',
            ['PD_3V3','GND','PD_SCL','PD_SDA','PD_EE_WP'],45,245)
p.connector('J103','Spinal-cord host / charging supervision',
            ['AON_5V','GND','HOST_SCL','HOST_SDA','PD_IRQ_N','CHG_ALLOW','PACK_TEMP_STATUS'],180,245)
resistor(p,'R108','10k 1% / isolate status from fault node','PACK_TEMP_OK','PACK_TEMP_STATUS',235,265)
p.text('ADCIN1=ADCIN2=LDO_1V5: decoded 5/5. Verify dead-battery startup/address with the final TI configuration.\nEEPROM is 0x50. I2Cc is private to the PD controller, EEPROM and charger.\nThe spinal cord uses the I2Ct interface and TI bridge commands, not a second controller on I2Cc.',235,230)

b = Sheet('02-charger', '3S charger / dual input', 3)
b.text('3S / 1.5 MHz / charger and logic power path; motor power must bypass BATFET',15,15,1.8)
b.text('BQ25798 Rev C pin functions and TI BMS034 / TIDA-050047 reference connections.\nMaximum 5 A charge; SYS battery discharge limited to 6 A RMS. Not a whole-robot motor feed.\nNo battery connection until pack protection, temperature interlock and configuration are verified.',15,24)
bl=[(2,'VBUS','power_in'),(3,'VBUS','passive'),(9,'VAC1','input'),(8,'VAC2','input'),
    (11,'ACDRV1','output'),(10,'ACDRV2','output'),(29,'PMID','passive'),(5,'REGN','power_out'),
    (6,'D+','bidirectional'),(7,'D-','bidirectional'),(14,'SCL','input'),(15,'SDA','bidirectional'),
    (20,'PROG','input'),(17,'ILIM_HIZ','input'),(13,'~{CE}','input'),(12,'~{QON}','input'),(27,'GND','power_in')]
br=[(4,'BTST1','passive'),(28,'SW1','output'),(19,'BTST2','passive'),(26,'SW2','output'),
    (25,'SYS','power_out'),(18,'BATP','input'),(22,'BAT','power_in'),(23,'BAT','passive'),
    (24,'SDRV','output'),(1,'STAT','open_collector'),(21,'~{INT}','open_collector'),(16,'TS','input')]
b.part(ic('BQ25798',bl,br,15.24),'U201','BQ25798RQMR',145,100,
       {'2':'CHG_VBUS','3':'CHG_VBUS','9':'USB_PD_IN','8':'WIRELESS_DC','11':'USB_ACDRV','10':'WLC_ACDRV',
        '29':'CHG_PMID','5':'CHG_REGN','6':'GND','7':'GND','14':'PD_SCL','15':'PD_SDA','20':'CHG_PROG',
        '17':'CHG_REGN','13':'CHG_CE_N','12':'CHG_QON_N','27':'GND','4':'CHG_BOOT1','28':'CHG_SW1',
        '19':'CHG_BOOT2','26':'CHG_SW2','25':'LOGIC_SYS','18':'CHG_BATP','22':'PACK_PROTECTED',
        '23':'PACK_PROTECTED','24':'CHG_SDRV','1':'CHG_STAT_N','21':'CHG_INT_N','16':'PACK_NTC_MAIN'},
       mpn='BQ25798RQMR',datasheet='https://www.ti.com/lit/ds/symlink/bq25798.pdf')
# External back-to-back NMOS input selectors. Pin maps are logical until exact FET/land selection.
fet=ic('NMOS_INPUT_SELECTOR',[(1,'G','input'),(2,'S','passive')],[(3,'D','passive')],5.08)
for ref,src,gate,mid,x in [('Q201','USB_PD_IN','USB_ACDRV','USB_FET_MID',35),
                          ('Q203','WIRELESS_DC','WLC_ACDRV','WLC_FET_MID',75)]:
    b.part(fet,ref,'Input NMOS / select rating',x,180,{'1':gate,'2':mid,'3':src})
    b.part(fet,'Q'+str(int(ref[1:])+1),'Reverse-blocking NMOS / select rating',x,220,
           {'1':gate,'2':mid,'3':'CHG_VBUS'})
for j,(net,val) in enumerate([('CHG_VBUS','10u 35V'),('CHG_VBUS','10u 35V'),('CHG_PMID','10u 35V'),
                              ('CHG_PMID','10u 35V'),('LOGIC_SYS','10u 25V'),('LOGIC_SYS','10u 25V'),
                              ('LOGIC_SYS','10u 25V'),('LOGIC_SYS','10u 25V'),('LOGIC_SYS','10u 25V'),
                              ('PACK_PROTECTED','10u 25V'),('PACK_PROTECTED','10u 25V'),('CHG_REGN','4u7 10V')]):
    capacitor(b,'C'+str(201+j),val,net,235+(j%5)*33,60+(j//5)*42,bulk=True)
for j,net in enumerate(['CHG_VBUS','CHG_PMID','LOGIC_SYS','PACK_PROTECTED']):
    capacitor(b,'C'+str(213+j),'100n 50V',net,235+j*38,190)
capacitor(b,'C217','47n 16V','CHG_BOOT1',220,235,'CHG_SW1')
capacitor(b,'C218','47n 16V','CHG_BOOT2',265,235,'CHG_SW2')
capacitor(b,'C219','1n 50V','CHG_SDRV',310,235)
b.passive('L','L201','1uH / Isat and DCR selection pending','CHG_SW1','CHG_SW2',365,235)
resistor(b,'R201','100 1%','PACK_PROTECTED','CHG_BATP',35,260)
resistor(b,'R202','10k5 1%','CHG_PROG','GND',80,260)
resistor(b,'R203','100k 1%','AON_3V3','CHG_QON_N',125,260)
resistor(b,'R204','5k23 1%','CHG_REGN','PACK_NTC_MAIN',170,260)
resistor(b,'R205','30k1 1%','PACK_NTC_MAIN','GND',215,260)
b.connector('J201','Pack primary 103AT-2 NTC (must touch cells)', ['PACK_NTC_MAIN','GND'],155,180)
b.connector('J202','Wireless receiver DC; select matched receiver/coil', ['WIRELESS_DC','GND'],40,140)
b.connector('J203','Fused/protected 3S pack / NOT bare cells', ['PACK_FROM_BMS','GND'],340,25)
resistor(b,'R206','2k2 1%','AON_3V3','CHG_LED_A',330,155)
led=ic('LED_AK',[(1,'K','passive')],[(2,'A','passive')],3.81)
b.part(led,'D201','Charge status LED',250,25,{'1':'CHG_STAT_N','2':'CHG_LED_A'})
b.text('WIRELESS_DC: 3.6-24 V chip operating range; system target 20 V. Receiver not selected.\nACDRV selectors require exact FET ratings and gate-connection audit before release.\nAC input selector and BATFET paths must be checked for source handover/backfeed.',15,280)

v = Sheet('03-housekeeping', 'Always-on 5 V / charge permission', 4)
v.text('Always-on supply remains alive while Jetson and motor rails are disabled',15,15,1.8)
v.text('5 V buck network follows TI TIDA-050047; adapt compensation only after load/inductor selection.\nThe spinal cord remains powered to supervise charging and handle button wake.\nCharging permission defaults OFF; external temperature-OK also defaults OFF.',15,24)
vl=[(2,'VIN','power_in'),(3,'EN','input'),(4,'SS','passive'),(6,'COMP','passive'),(7,'GND','power_in'),(9,'PAD','passive')]
vr=[(1,'BOOT','passive'),(8,'PH','output'),(5,'VSENSE','input')]
v.part(ic('TPS54531',vl,vr,10.16),'U301','TPS54531DDAR',75,80,
       {'2':'LOGIC_SYS','3':'AON_BUCK_EN','4':'AON_SS','6':'AON_COMP','7':'GND','9':'GND',
        '1':'AON_BOOT','8':'AON_SW','5':'AON_FB'},
       'Package_SO:SOIC-8-1EP_3.9x4.9mm_P1.27mm_EP2.29x3mm','TPS54531DDAR',
       'https://www.ti.com/lit/ds/symlink/tps54531.pdf')
v.passive('L','L301','4u7 / >=5 A thermal, saturation selection pending','AON_SW','AON_5V',145,70)
diode=ic('Schottky_KA',[(1,'K','passive')],[(2,'A','passive')],3.81)
v.part(diode,'D301','CDBC540-G',145,110,{'1':'AON_SW','2':'GND'},'Diode_SMD:D_SMC','CDBC540-G',
       'https://www.comchiptech.com/admin/files/product/QW-BB012%20CDBC520-G~5100-G%20RevB522561.pdf')
for ref,val,a,bn,x,y in [('R301','665k 1%','LOGIC_SYS','AON_BUCK_EN',30,130),('R302','130k 1%','AON_BUCK_EN','GND',70,130),
                        ('R303','10k2 1%','AON_5V','AON_FB',210,85),('R304','1k96 1%','AON_FB','GND',250,85),
                        ('R305','37k4 1%','AON_COMP','AON_COMP_RC',110,175)]:
    resistor(v,ref,val,a,bn,x,y)
for ref,val,net,x,y,bn,bulk in [('C301','4u7 50V','LOGIC_SYS',25,80,'GND',True),('C302','4u7 50V','LOGIC_SYS',25,105,'GND',True),
                              ('C303','100n 16V','AON_BOOT',190,130,'AON_SW',False),('C304','10n 50V','AON_SS',25,175,'GND',False),
                              ('C305','2n2 50V','AON_COMP_RC',70,175,'GND',False),('C306','22p C0G','AON_COMP',145,175,'GND',False),
                              ('C307','100u 10V','AON_5V',210,130,'GND',True),('C308','47u 16V','AON_5V',250,130,'GND',True),
                              ('C309','47u 16V','AON_5V',290,130,'GND',True)]:
    capacitor(v,ref,val,net,x,y,bn,bulk)
ldo=ic('TLV75533P',[(1,'IN','power_in'),(3,'EN','input'),(2,'GND','power_in')],[(5,'OUT','power_out'),(4,'NC','no_connect')],7.62)
v.part(ldo,'U302','TLV75533PDBVR',335,80,{'1':'AON_5V','3':'AON_5V','2':'GND','5':'AON_3V3','4':None},
       'Package_TO_SOT_SMD:SOT-23-5','TLV75533PDBVR','https://www.ti.com/lit/ds/symlink/tlv755p.pdf')
capacitor(v,'C310','1u 10V','AON_5V',300,40)
capacitor(v,'C311','1u 10V','AON_3V3',370,40)
nand=ic('SN74LVC1G00',[(1,'A','input'),(2,'B','input'),(3,'GND','power_in')],[(4,'Y','output'),(5,'VCC','power_in')],7.62)
v.part(nand,'U303','SN74LVC1G00DBVR',255,230,
       {'1':'CHG_ALLOW','2':'PACK_TEMP_OK','3':'GND','4':'CHG_CE_N','5':'AON_3V3'},
       'Package_TO_SOT_SMD:SOT-23-5','SN74LVC1G00DBVR','https://www.ti.com/lit/ds/symlink/sn74lvc1g00.pdf')
resistor(v,'R306','100k 1%','CHG_ALLOW','GND',150,230)
resistor(v,'R307','100k 1%','PACK_TEMP_OK','GND',190,230)
resistor(v,'R308','100k 1%','CHG_REGN','CHG_CE_N',335,230)
capacitor(v,'C312','100n 16V','AON_3V3',375,230)
v.text('Charge-enable truth table: CE_N = NOT(CHG_ALLOW AND PACK_TEMP_OK).\nREGN pull-up inhibits charging if the gate loses its 3.3 V supply.\nThree-segment temperature window is on sheet 12; firmware remains open.\nA logic gate is not a replacement for the independent pack BMS.',20,260)

# Explicit source declarations, not an assertion of electrical ratings.
for net,x,y in [('GND',40,40),('USB_VBUS',100,40)]:
    p.power_flag(net,x,y)
for net,x,y in [('CHG_VBUS',45,45),('PACK_FROM_BMS',110,45),('PACK_PROTECTED',185,45)]:
    b.power_flag(net,x,y)
v.power_flag('AON_5V',380,150)
sheets=[p,b,v]
add_regulators(sheets,resistor,capacitor)
add_hands(sheets,resistor,capacitor)
add_control(sheets,resistor,capacitor)
add_temperature(sheets,resistor,capacitor)
add_branches(sheets,resistor,capacitor)
save_project(ROOT,sheets)
