"""Independent motor-arm latch and pack metering circuit capture."""
from schematic import Sheet, ic

def add_control(sheets, resistor, capacitor):
    s=Sheet('10-motor-cut','Motor-cut latch / control',11)
    s.text('HARDWARE MOTOR CUT / closing the cut loop does not re-arm the motors',15,15,1.8)
    s.text('Normally-closed external cut loop and voltage-supervisor output clear the D flip-flop.\nThe controller must send a new MOTOR_ARM clock edge after the loop is closed and the supply is good.\nThis is an electrical motor-power cut, not a validated stop/pose strategy.',15,24)
    ff=ic('SN74LVC74A',[(1,'~{1CLR}','input'),(2,'1D','input'),(3,'1CLK','input'),(4,'~{1PRE}','input'),
                       (7,'GND','power_in'),(10,'~{2PRE}','input'),(11,'2CLK','input'),(12,'2D','input'),(13,'~{2CLR}','input')],
          [(5,'1Q','output'),(6,'~{1Q}','output'),(8,'~{2Q}','output'),(9,'2Q','output'),(14,'VCC','power_in')],10.16)
    s.part(ff,'U1001','SN74LVC74APWR',75,90,
           {'1':'MOTOR_CLEAR_N','2':'AON_3V3','3':'MOTOR_ARM','4':'AON_3V3','5':'MOTOR_ARMED',
            '6':None,'7':'GND','8':None,'9':None,'10':'AON_3V3','11':'GND','12':'GND','13':'GND','14':'AON_3V3'},
           'Package_SO:TSSOP-14_4.4x5mm_P0.65mm','SN74LVC74APWR','https://www.ti.com/lit/ds/symlink/sn74lvc74a.pdf')
    and4=ic('SN74LVC08A',[(1,'1A','input'),(2,'1B','input'),(4,'2A','input'),(5,'2B','input'),
                         (9,'3A','input'),(10,'3B','input'),(12,'4A','input'),(13,'4B','input'),(7,'GND','power_in')],
            [(3,'1Y','output'),(6,'2Y','output'),(8,'3Y','output'),(11,'4Y','output'),(14,'VCC','power_in')],10.16)
    s.part(and4,'U1002','SN74LVC08APWR',220,90,
           {'1':'MOTOR_ARMED','2':'MOTOR_REG_REQUEST','3':'MOTOR_REG_EN',
            '4':'MOTOR_ARMED','5':'HAND_LEFT_REQUEST','6':'HAND_LEFT_ENABLE',
            '9':'MOTOR_ARMED','10':'HAND_RIGHT_REQUEST','8':'HAND_RIGHT_ENABLE',
            '12':'MOTOR_ARMED','13':'LEG_REQUEST','11':'LEG_SWITCH_ENABLE','7':'GND','14':'AON_3V3'},
           'Package_SO:TSSOP-14_4.4x5mm_P0.65mm','SN74LVC08APWR','https://www.ti.com/lit/ds/symlink/sn74lvc08a.pdf')
    supervisor=ic('TLV803E_DBZ_default',[(3,'VDD','power_in'),(1,'GND','power_in')],[(2,'~{RESET}','open_collector')],7.62)
    s.part(supervisor,'U1003','TLV803EA30DBZR / verify orderability',345,90,
           {'1':'GND','2':'MOTOR_CLEAR_N','3':'AON_3V3'},'Package_TO_SOT_SMD:SOT-23',
           'TLV803EA30DBZR','https://www.ti.com/lit/ds/symlink/tlv803e.pdf')
    resistor(s,'R1001','3k3 1%','AON_3V3','CUT_LOOP_FEED',40,175)
    resistor(s,'R1002','100k 1%','MOTOR_CLEAR_N','GND',85,175)
    for k,net in enumerate(['MOTOR_ARM','MOTOR_REG_REQUEST','HAND_LEFT_REQUEST','HAND_RIGHT_REQUEST','LEG_REQUEST']):
        resistor(s,'R'+str(1003+k),'100k 1%',net,'GND',140+k*48,175)
    for k,x in enumerate([40,85,140]):
        capacitor(s,'C'+str(1001+k),'100n 16V','AON_3V3',x,225)
    s.connector('J1001','Normally-closed cut loop / opens to clear',['CUT_LOOP_FEED','MOTOR_CLEAR_N'],235,225)
    s.connector('J1002','Spinal-cord arm and rail requests',
                ['GND','MOTOR_ARM','MOTOR_REG_REQUEST','HAND_LEFT_REQUEST','HAND_RIGHT_REQUEST','LEG_REQUEST','MOTOR_ARMED','UPPER_REQUEST','MOTOR_CLEAR_N','JETSON_EN','PERIPHERAL_ENABLE','PACK_ALERT_N'],345,225)
    s.text('Branch fuses, output overvoltage clamps and returned-energy absorption still need exact selection\nand qualification. Do not bridge those protections with wire.\nSupply supervisor is 3.0 V / 200 ms candidate; exact package suffix and timing require verification.',15,275)
    sheets.append(s)

    g=Sheet('11-pack-meter','Pack current / energy meter',12)
    g.text('PACK METER / every charging and load path passes through the shunt',15,15,1.8)
    g.text('INA228 SLYS021A Table 5-1; I2C address 0x40 when A0=A1=GND.\nCounts charge and energy. It does not provide a calibrated state-of-charge model or replace a BMS.\nPack fuse and the 3S protection/balancing assembly remain upstream release items.',15,24)
    monitor=ic('INA228',[(10,'IN+','input'),(9,'IN-','input'),(8,'VBUS','input'),(6,'VS','power_in'),(7,'GND','power_in')],
               [(1,'A1','input'),(2,'A0','input'),(3,'~{ALERT}','open_collector'),(4,'SDA','bidirectional'),(5,'SCL','input')],10.16)
    g.part(monitor,'U1101','INA228AIDGSR',95,100,
           {'1':'GND','2':'GND','3':'PACK_ALERT_N','4':'HOST_SDA','5':'HOST_SCL','6':'AON_3V3','7':'GND',
            '8':'PACK_FROM_BMS','9':'PACK_SENSE_N','10':'PACK_SENSE_P'},
           'Package_SO:MSOP-10_3x3mm_P0.5mm','INA228AIDGSR','https://www.ti.com/lit/ds/symlink/ina228.pdf')
    g.passive('R','R1101','1m 0.5% / >=4 W target, exact part pending','PACK_FROM_BMS','PACK_PROTECTED',235,65)
    resistor(g,'R1102','10 1%','PACK_FROM_BMS','PACK_SENSE_P',190,130)
    resistor(g,'R1103','10 1%','PACK_PROTECTED','PACK_SENSE_N',240,130)
    resistor(g,'R1104','10k 1%','AON_3V3','PACK_ALERT_N',290,130)
    capacitor(g,'C1101','100n 16V','PACK_SENSE_P',190,190,'PACK_SENSE_N')
    capacitor(g,'C1102','100n 16V','AON_3V3',240,190)
    g.connector('J1101','Controller metering interrupt',['PACK_ALERT_N','GND'],320,190)
    g.text('High-side Kelvin connections to R1101. No separate charger or load-positive feed may bypass this shunt.\nDischarge polarity is positive; charge is negative with the shown shunt orientation.\nAt 40 A a 1 mOhm shunt dissipates 1.6 W; thermal placement must keep heat away from the cell spine.',15,250)
    sheets.append(g)
