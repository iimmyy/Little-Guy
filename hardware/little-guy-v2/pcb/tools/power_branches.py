"""Separately switched leg and upper-body branches; protection not released."""
from schematic import Sheet, ic

def add_branches(sheets,resistor,capacitor):
    s=Sheet('13-motor-branches','Leg and upper-body switching',14)
    s.text('SEPARATE LEG / UPPER-BODY SWITCHES / source is regulated motor bus',15,15,1.8)
    s.text('Paired Si7149DP P-channel devices provide a low-resistance high-side switch per group.\nTheir body diodes allow returned energy back toward the common motor rail. That energy still needs an absorber.\nFuse ratings, harness leaf protection, inrush SOA and regeneration are not qualified; do not connect motors yet.',15,24)
    pm=ic('Power_PMOS_8',[(1,'S','passive'),(2,'S','passive'),(3,'S','passive'),(4,'G','input')],
          [(5,'D','passive'),(6,'D','passive'),(7,'D','passive'),(8,'D','passive')],5.08)
    npn=ic('MMBT3904',[(1,'B','input'),(2,'E','passive')],[(3,'C','passive')],5.08)
    for group,base,x,enable in [('LEG',1300,60,'LEG_SWITCH_ENABLE'),('UPPER',1350,245,'UPPER_SWITCH_ENABLE')]:
        for k in range(2):
            s.part(pm,'Q'+str(base+1+k),'Si7149DP-T1-GE3',x+k*70,85,
                   {**{str(i):'MOTOR_11V3' for i in [1,2,3]},'4':group+'_GATE',
                    **{str(i):group+'_PRE_FUSE' for i in [5,6,7,8]}},
                   mpn='SI7149DP-T1-GE3',datasheet='https://www.vishay.com/docs/68934/si7149dp.pdf')
        s.part(npn,'Q'+str(base+3),'MMBT3904-7-F',x,155,
               {'1':group+'_DRIVE_B','2':'GND','3':group+'_DRIVE_C'},
               'Package_TO_SOT_SMD:SOT-23','MMBT3904-7-F','https://www.diodes.com/assets/Datasheets/MMBT3904.pdf')
        resistor(s,'R'+str(base+1),'10k 1%',enable,group+'_DRIVE_B',x+65,150)
        resistor(s,'R'+str(base+2),'100k 1%',group+'_DRIVE_B','GND',x+110,150)
        resistor(s,'R'+str(base+3),'4k7 1%',group+'_GATE',group+'_DRIVE_C',x,215)
        resistor(s,'R'+str(base+4),'100k 1%','MOTOR_11V3',group+'_GATE',x+65,215)
        capacitor(s,'C'+str(base+1),'220n 50V / inrush tuning',group+'_GATE',x+110,215,group+'_PRE_FUSE')
        s.passive('F','F'+str(base+1),'Branch fuse / rating and holder pending',group+'_PRE_FUSE',group+'_11V3',x,265)
        s.connector('J'+str(base+1),group+' power to protected harness',[group+'_11V3','GND'],x+80,185 if group=='UPPER' else 265)
    gate=ic('SN74LVC1G08',[(1,'A','input'),(2,'B','input'),(3,'GND','power_in')],[(4,'Y','output'),(5,'VCC','power_in')],7.62)
    s.part(gate,'U1301','SN74LVC1G08DBVR',340,45,
           {'1':'MOTOR_ARMED','2':'UPPER_REQUEST','3':'GND','4':'UPPER_SWITCH_ENABLE','5':'AON_3V3'},
           'Package_TO_SOT_SMD:SOT-23-5','SN74LVC1G08DBVR','https://www.ti.com/lit/ds/symlink/sn74lvc1g08.pdf')
    resistor(s,'R1391','100k 1%','UPPER_REQUEST','GND',200,50)
    capacitor(s,'C1391','100n 16V','AON_3V3',245,50)
    sheets.append(s)
