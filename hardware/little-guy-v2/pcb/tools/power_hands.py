"""Independent hand converters. PCB and load-transient verification still required."""
from schematic import Sheet, ic

def add_hands(sheets, resistor, capacitor):
    for side,base,page in [('left',800,9),('right',900,10),('peripheral',1400,15)]:
        s=Sheet(f'{page-1:02}-hand-{side}','Peripheral 5 V supply' if side=='peripheral' else f'{side.capitalize()} hand supply',page)
        prefix='PERIPHERAL' if side=='peripheral' else 'HAND_'+side.upper()
        peripheral=side=='peripheral'
        output='5V0' if peripheral else '5V7'
        s.text('PERIPHERALS / 5.016 V nominal / independent switched branch' if peripheral else f'{side.upper()} HAND / 5.695 V nominal / 10-A converter class',15,15,1.8)
        s.text('LM61495RPHR / 400 kHz. Source is the protected pack, independent of charger SYS.\nScreen, USB hub and sensor-module output; USB/LTE startup pulses need measurement.\nThis rail switches off for charging sleep; branch fuse and connector ratings remain open.' if peripheral else 'LM61495RPHR adjustable variant, SNVSBZ4A Table 6-1. 400 kHz.\nNominal voltage is below 6 V to reserve margin for the SCS0009 maximum; confirm exact servo specification.\n8.2 A summed stall reference per hand is a pulse/fault test condition, not an intended continuous load.',15,24)
        left=[(1,'PGND2','power_in'),(2,'VIN2','power_in'),(3,'RBOOT','passive'),(4,'CBOOT','passive'),
              (5,'BIAS','power_in'),(6,'VCC','power_out'),(7,'FB','input'),(8,'AGND','power_in')]
        right=[(9,'RT','input'),(10,'RESET','open_collector'),(11,'SPSP','input'),(12,'SYNC/MODE','input'),
               (13,'EN','input'),(14,'VIN1','power_in'),(15,'PGND1','power_in'),(16,'SW','output')]
        net=lambda suffix: prefix+'_'+suffix
        s.part(ic('LM61495',left,right,12.7),'U'+str(base+1),'LM61495RPHR',80,95,
               {'1':'GND','2':'PACK_PROTECTED','3':net('RBOOT'),'4':net('BOOT'),'5':net(output),
                '6':net('VCC'),'7':net('FB'),'8':'GND','9':net('VCC'),'10':net('PG_N'),
                '11':'GND','12':'GND','13':net('ENABLE'),'14':'PACK_PROTECTED','15':'GND','16':net('SW')},
               mpn='LM61495RPHR',datasheet='https://www.ti.com/lit/ds/symlink/lm61495.pdf')
        s.passive('L','L'+str(base+1),'4u7 / saturation >= switch short-circuit limit',net('SW'),net(output),180,70)
        for k,val,a,b,x,y in [(1,'0 / gate slew tuning',net('RBOOT'),net('BOOT'),175,120),
                              (2,'100k 0.1%',net(output),net('FB'),230,70),
                              (3,'24k9 0.1%' if peripheral else '21k3 0.1%',net('FB'),'GND',280,70),
                              (4,'4k99 1%',net(output),net('FF_RC'),330,70),
                              (5,'100k 1%',net('ENABLE'),'GND',230,130),
                              (6,'10k 1%','AON_3V3',net('PG_N'),280,130)]:
            resistor(s,'R'+str(base+k),val,a,b,x,y)
        for k,val,a,b,x,y in [(1,'100n 16V',net('BOOT'),net('SW'),30,180),
                              (2,'1u 10V',net('VCC'),'GND',75,180),
                              (3,'15p C0G / loop check required',net('FF_RC'),net('FB'),330,130),
                              (4,'100n 16V',net(output),'GND',125,180),
                              (5,'470n 50V','PACK_PROTECTED','GND',180,180),
                              (6,'470n 50V','PACK_PROTECTED','GND',230,180)]:
            capacitor(s,'C'+str(base+k),val,a,x,y,b)
        for k in range(4):
            capacitor(s,'C'+str(base+7+k),'10u 50V X7R','PACK_PROTECTED',35+k*70,220,bulk=True)
        for k in range(4):
            capacitor(s,'C'+str(base+11+k),'47u 16V X7R',net(output),35+k*70,265,bulk=True)
        s.connector('J'+str(base+1),'Power output / branch and cable protection pending',[net(output),'GND'],370,190)
        s.connector('J'+str(base+2),'Hardware-gated enable and power-good',[net('ENABLE'),net('PG_N'),'GND'],370,230)
        s.power_flag(net(output),370,45)
        sheets.append(s)
