"""Teensy 4.1 carrier: four TTL buses, IMU, touch, sensor fan-out, watchdog."""
from pathlib import Path
from schematic import Sheet,ic,save_project
from board_common import r,c
from module_footprints import teensy_socket

ROOT=Path(__file__).resolve().parents[1]/'spinal'
PROJECT='little-guy-spinal'
def sheet(name,title,page):return Sheet(name,title,page,PROJECT)
sheets=[]
s=sheet('01-controller','Teensy 4.1 / power-board controls',2)
s.text('SPINAL CORD / socketed Teensy 4.1; GPIO numbers match the PJRC Rev 4 pin card',15,15,1.8)
s.text('VIN uses AON_5V. BEFORE external VIN plus USB: cut the Teensy VUSB-VIN link exactly as PJRC documents.\nUSB is the module device port to the Jetson. No second USB power feed may enter VIN.\n3V3 output budget is 250 mA per PJRC; both feet have their own 5 V-fed regulators.',15,24)
assignment={0:('LEG_RX','input'),1:('LEG_TX','output'),2:('LEG_DE','output'),3:('UPPER_DE','output'),
            4:('LEFT_DE','output'),5:('RIGHT_DE','output'),6:('IMU_INT','input'),7:('UPPER_RX','input'),
            8:('UPPER_TX','output'),9:('WATCHDOG_KICK','output'),10:('IMU_CS_N','output'),
            11:('IMU_MOSI','output'),12:('IMU_MISO','input'),13:('IMU_SCK','output'),
            14:('LEFT_TX','output'),15:('LEFT_RX','input'),16:('RIGHT_RX','input'),17:('RIGHT_TX','output'),
            18:('HOST_SDA','bidirectional'),19:('HOST_SCL','open_collector'),20:('MOTOR_ARM','output'),
            21:('MOTOR_REG_REQUEST','output'),22:('HAND_LEFT_REQUEST','output'),23:('HAND_RIGHT_REQUEST','output'),
            24:('LEG_REQUEST','output'),25:('UPPER_REQUEST','output'),26:('JETSON_EN','output'),
            27:('PERIPHERAL_ENABLE','output'),28:('CHG_ALLOW','output'),29:('PACK_TEMP_STATUS','input'),
            30:('MOTOR_ARMED','input'),31:('PD_IRQ_N','input'),32:('PACK_ALERT_N','input'),
            33:('SENSOR_MUX_RESET_N','output'),34:('LEFT_READY_N','input'),35:('RIGHT_READY_N','input'),
            36:('POWER_BUTTON_N','input'),37:('POWER_LED_DRIVE','output'),38:(None,'bidirectional'),
            39:('TOUCH_IRQ_N','input'),40:('LEFT_XSHUT','output'),41:('RIGHT_XSHUT','output')}
# Pad names deliberately use GPIO identities. A matching custom socket footprint is required.
left=[('VIN','VIN','power_in'),('GND0','GND','power_in'),('3V3A','3V3','power_out')]
right=[('GND1','GND','passive'),('GND2','GND','passive'),('3V3B','3V3','passive')]
for i in range(42):
    (left if i<21 else right).append(('GPIO'+str(i),str(i)+' / '+(assignment[i][0] or 'unused'),assignment[i][1]))
nets={str(num):assignment[int(str(num)[4:])][0] for num,*_ in left+right if str(num).startswith('GPIO')}
nets.update({'VIN':'AON_5V','GND0':'GND','GND1':'GND','GND2':'GND','3V3A':'MCU_3V3','3V3B':'MCU_3V3'})
s.part(ic('Teensy41_Socket_GPIO_ID',left,right,27.94),'U101','Teensy 4.1 / VUSB-VIN link cut',105,120,nets,
       'LGV2_Footprints:Teensy41_Socket_GPIO_ID','TEENSY41','https://www.pjrc.com/store/teensy41.html')
s.connector('J101','To power J103 / charging and always-on power',
            ['AON_5V','GND','HOST_SCL','HOST_SDA','PD_IRQ_N','CHG_ALLOW','PACK_TEMP_STATUS'],285,75)
s.connector('J102','To power J1002 / motor control and watchdog clear',
            ['GND','MOTOR_ARM','MOTOR_REG_REQUEST','HAND_LEFT_REQUEST','HAND_RIGHT_REQUEST',
             'LEG_REQUEST','MOTOR_ARMED','UPPER_REQUEST','WATCHDOG_FAULT_N','JETSON_EN','PERIPHERAL_ENABLE','PACK_ALERT_N'],285,140)
s.connector('J103','Power-board logic enables / connector interface pending',
            ['GND','JETSON_EN','PERIPHERAL_ENABLE','PACK_ALERT_N'],285,205)
s.connector('J104','Front-door button and LED / no motor current here',
            ['GND','POWER_BUTTON_N','AON_5V','LED_CATHODE'],285,250)
r(s,'R101','10k 1%','MCU_3V3','POWER_BUTTON_N',60,235)
r(s,'R102','1k 1% / initial LED current','LED_CATHODE','BUTTON_LED_C',115,235)
npn=ic('MMBT3904',[(1,'B','input'),(2,'E','passive')],[(3,'C','passive')],5.08)
s.part(npn,'Q101','MMBT3904-7-F',185,235,{'1':'BUTTON_LED_B','2':'GND','3':'BUTTON_LED_C'},
       'Package_TO_SOT_SMD:SOT-23','MMBT3904-7-F','https://www.diodes.com/assets/Datasheets/MMBT3904.pdf')
r(s,'R103','10k 1%','POWER_LED_DRIVE','BUTTON_LED_B',55,270)
r(s,'R104','100k 1%','BUTTON_LED_B','GND',125,270)
c(s,'C101','10u 10V','AON_5V',350,75)
c(s,'C102','100n 16V','MCU_3V3',350,135)
s.power_flag('AON_5V',355,195);s.power_flag('GND',355,240)
sheets.append(s)

tx=ic('AHCT1G126_DBV',[(1,'OE','input'),(2,'A','input'),(3,'GND','power_in')],[(4,'Y','tri_state'),(5,'VCC','power_in')],7.62)
rx=ic('LVC1G125_DBV',[(1,'~{OE}','input'),(2,'A_5V_tolerant','input'),(3,'GND','power_in')],[(4,'Y','output'),(5,'VCC','power_in')],10.16)
for bank,groups in enumerate([['LEG','UPPER'],['LEFT','RIGHT']]):
    s=sheet('0'+str(bank+2)+'-servo-buses-'+str(bank+1),'Half-duplex TTL buses '+str(bank+1),bank+3)
    s.text('SERVO DATA ONLY / power harness comes from the power board, never from these connectors',15,15,1.8)
    s.text('AHCT TX buffer: 3.3 V TTL-compatible inputs, 5 V bus output, active-high enable default low.\nLVC RX buffer: powered at 3.3 V with a 5.5 V-tolerant input. Receive echo is visible during transmit.\nDirection timing, first/last byte, powered-off servo behaviour and 1 Mbps edges require the actual harness bench test.',15,24)
    for k,g in enumerate(groups):
        base=200+bank*100+k*40;y=85+k*115
        s.part(tx,'U'+str(base+1),'SN74AHCT1G126DBVR',70,y,
               {'1':g+'_TX_ENABLE','2':g+'_TX','3':'GND','4':g+'_TX_DRV','5':'AON_5V'},
               'Package_TO_SOT_SMD:SOT-23-5','SN74AHCT1G126DBVR','https://www.ti.com/lit/ds/symlink/sn74ahct1g126.pdf')
        r(s,'R'+str(base+1),'47 1% / source damping',g+'_TX_DRV',g+'_DATA',150,y)
        s.part(rx,'U'+str(base+2),'SN74LVC1G125DBVR',245,y,
               {'1':'GND','2':g+'_DATA','3':'GND','4':g+'_RX','5':'MCU_3V3'},
               'Package_TO_SOT_SMD:SOT-23-5','SN74LVC1G125DBVR','https://www.ti.com/lit/ds/symlink/sn74lvc1g125.pdf')
        s.connector('J'+str(base+1),g+' data to powered servo harness',[g+'_DATA','GND'],350,y)
        r(s,'R'+str(base+2),'100k 1%',g+'_DE','GND',45,y+45)
        r(s,'R'+str(base+3),'100k 1%','MCU_3V3',g+'_TX',125,y+45)
        c(s,'C'+str(base+1),'100n 16V','AON_5V',205,y+45)
        c(s,'C'+str(base+2),'100n 16V','MCU_3V3',285,y+45)
    s.text('LEG=Serial1 pins 0/1, UPPER=Serial2 pins 7/8, LEFT=Serial3 pins 15/14, RIGHT=Serial4 pins 16/17.\nSTM/STS and older SCS framing is firmware work; these buffers do not translate protocols.\nExternal ESD device selection and bus pull-up, if the actual servos require one, remain bench decisions.',15,270)
    sheets.append(s)

s=sheet('04-sensor-buses','Sensor mux / IMU / foot cabling',5)
s.text('SENSOR FAN-OUT / TCA9548A 0x70 isolates duplicate addresses and long cable branches',15,15,1.8)
s.text('Select only one branch at a time. Ch0=left foot, Ch1=right foot, Ch2=BME688, Ch3/4=extra edge sensors.\nStart at 100 kHz; pull-ups and maximum cable capacitance must be qualified together.\nIMU uses a separate four-wire SPI port so an I2C cable fault does not occupy that interface.',15,24)
pins=[(1,'A0','input'),(2,'A1','input'),(3,'~{RESET}','input'),(12,'GND','power_in'),
      (21,'A2','input'),(22,'SCL','bidirectional'),(23,'SDA','bidirectional'),(24,'VCC','power_in')]
pairs=[(4,5),(6,7),(8,9),(10,11),(13,14),(15,16),(17,18),(19,20)]
outs=[];nets={'1':'GND','2':'GND','3':'SENSOR_MUX_RESET_N','12':'GND','21':'GND','22':'HOST_SCL','23':'HOST_SDA','24':'MCU_3V3'}
for k,(sd,sc) in enumerate(pairs):
    outs += [(sd,'SD'+str(k),'bidirectional'),(sc,'SC'+str(k),'bidirectional')]
    nets[str(sd)]='SENSOR'+str(k)+'_SDA' if k<5 else None
    nets[str(sc)]='SENSOR'+str(k)+'_SCL' if k<5 else None
s.part(ic('TCA9548A_PW',pins,outs,12.7),'U401','TCA9548APWR',85,95,nets,
       'Package_SO:TSSOP-24_4.4x7.8mm_P0.65mm','TCA9548APWR','https://www.ti.com/lit/ds/symlink/tca9548a.pdf')
r(s,'R401','10k 1%','MCU_3V3','SENSOR_MUX_RESET_N',45,165)
pulls=sheet('10-sensor-pullups','Sensor branch pull-ups / fitting options',11)
pulls.text('I2C BRANCH PULL-UPS / one fitted pair per selected electrical branch',15,15,1.8)
pulls.text('Initial value 4k7. Remove duplicate remote module pull-ups or mark these DNP as appropriate.\nActual values depend on full cable capacitance, sink current and selected bus speed.\nQualify all five branches; this sheet does not establish cable noise immunity.',15,24)
for k in range(5):
    if k<2:
        side=['LEFT','RIGHT'][k]
        s.passive('F','F'+str(401+k),'MF-MSMF020-2 / verify trip coordination','AON_5V',side+'_FOOT_5V',200+k*100,60,
                  'Fuse:Fuse_1812_4532Metric','MF-MSMF020-2')
        nets=[side+'_FOOT_5V','GND','SENSOR'+str(k)+'_SCL','SENSOR'+str(k)+'_SDA',side+'_READY_N',side+'_XSHUT']
        title=side+' foot / mates foot J101 pin for pin'
    else:
        nets=['AON_5V','GND','SENSOR'+str(k)+'_SCL','SENSOR'+str(k)+'_SDA']
        title=['BME688 module','Edge sensor A','Edge sensor B'][k-2]
    s.connector('J'+str(401+k),title,nets,205+(k%2)*125,120+(k//2)*65)
    for j,sig in enumerate(['SCL','SDA']):
        r(pulls,'R'+str(402+k*2+j),'4k7 1% / fitting option','MCU_3V3','SENSOR'+str(k)+'_'+sig,45+k*75,90+j*75)
c(s,'C401','100n 16V','MCU_3V3',365,60)
sheets.append(s)
sheets.append(pulls)

s=sheet('05-imu-watchdog','SPI IMU / independent watchdog',6)
s.text('IMU / LSM6DSOX SPI mode 1; WATCHDOG / loss of falling edges clears power-board motor latch',15,15,1.8)
s.text('Watchdog timeout 170-230 ms (200 ms typical) using factory timing; WDO is open drain.\nA frozen Teensy is detected electrically. Jetson link loss still requires firmware to stop kicking / disarm.\nPose-hold, controlled sitting, fall response and recovery are unimplemented firmware and physical validation tasks.',15,24)
imu=ic('LSM6DSOX',[(1,'SDO','output'),(2,'SDx','input'),(3,'SCx','input'),(4,'INT1','output'),
                     (5,'VDDIO','power_in'),(6,'GND','power_in'),(7,'GND','passive')],
       [(8,'VDD','power_in'),(9,'INT2','output'),(10,'OCS_Aux','no_connect'),(11,'SDO_Aux','output'),
        (12,'~{CS}','input'),(13,'SCL_SPC','input'),(14,'SDA_SDI','input')],10.16)
s.part(imu,'U501','LSM6DSOXTR',80,100,
       {'1':'IMU_MISO','2':'GND','3':'GND','4':'IMU_INT','5':'MCU_3V3','6':'GND','7':'GND','8':'MCU_3V3',
        '9':None,'10':None,'11':None,'12':'IMU_CS_N','13':'IMU_SCK','14':'IMU_MOSI'},
       mpn='LSM6DSOXTR',datasheet='https://www.st.com/resource/en/datasheet/lsm6dsox.pdf')
c(s,'C501','100n 16V','MCU_3V3',45,175);c(s,'C502','100n 16V','MCU_3V3',115,175)
r(s,'R501','10k 1%','MCU_3V3','IMU_CS_N',45,230)
wd=ic('TPS3431_DRB',[(1,'VDD','power_in'),(2,'CWD','passive'),(3,'EN','input'),(4,'GND','power_in'),(9,'EP','passive')],
      [(5,'SET1','input'),(6,'WDI','input'),(7,'~{WDO}','open_collector'),(8,'ENOUT','open_collector')],10.16)
s.part(wd,'U502','TPS3431SDRBR',260,100,
       {'1':'MCU_3V3','2':'WD_CWD','3':'MCU_3V3','4':'GND','5':'MCU_3V3','6':'WATCHDOG_KICK',
        '7':'WATCHDOG_FAULT_N','8':None,'9':'GND'},mpn='TPS3431SDRBR',datasheet='https://www.ti.com/lit/ds/symlink/tps3431.pdf')
r(s,'R502','10k 1%','MCU_3V3','WD_CWD',220,175)
r(s,'R503','10k 1%','MCU_3V3','WATCHDOG_FAULT_N',290,175)
r(s,'R504','100k 1%','WATCHDOG_KICK','GND',355,175)
c(s,'C503','100n 16V','MCU_3V3',220,230)
s.connector('J501','Watchdog test points / fault goes to power J1002.9',
            ['WATCHDOG_KICK','WATCHDOG_FAULT_N','GND'],310,235)
sheets.append(s)

touch=ic('MPR121_QFN20',[(1,'~{IRQ}','open_collector'),(2,'SCL','input'),(3,'SDA','bidirectional'),
                        (4,'ADDR','input'),(5,'VREG','power_out'),(6,'VSS','power_in'),(7,'REXT','passive')],
         [(i+8,'ELE'+str(i),'passive') for i in range(12)]+[(20,'VDD','power_in')],12.7)
for k,addrnet in enumerate(['GND','MCU_3V3','HOST_SDA','HOST_SCL']):
    base=600+k*100
    s=sheet('0'+str(k+6)+'-touch-'+str(k+1),'Touch electrodes '+str(k*12+1)+'-'+str(k*12+12),k+7)
    s.text('TOUCH / MPR121 address '+hex(0x5A+k)+'; 12 electrodes of 48',15,15,1.8)
    s.text('75k 1% REXT and independent 100n VREG bypass per NXP datasheet. IRQ outputs share a wired-OR line.\nRun electrode wires away from motor cables. Pad sizes, shell thickness and wire capacitance need actual-shell tuning.\nNo electrodes or copper over the charging coil, screen or antenna keep-outs.',15,24)
    nets={'1':'TOUCH_IRQ_N','2':'HOST_SCL','3':'HOST_SDA','4':addrnet,'5':'TOUCH'+str(k)+'_VREG',
          '6':'GND','7':'TOUCH'+str(k)+'_REXT','20':'MCU_3V3'}
    nets.update({str(8+j):'TOUCH_PAD_'+str(k*12+j+1).zfill(2) for j in range(12)})
    s.part(touch,'U'+str(base+1),'MPR121QR2 / '+hex(0x5A+k),95,105,nets,
           'Package_DFN_QFN:UQFN-20_3x3mm_P0.4mm','MPR121QR2','https://www.nxp.com/docs/en/data-sheet/MPR121.pdf')
    r(s,'R'+str(base+1),'75k 1%','TOUCH'+str(k)+'_REXT','GND',215,75)
    c(s,'C'+str(base+1),'100n 16V','MCU_3V3',285,75)
    c(s,'C'+str(base+2),'100n 16V','TOUCH'+str(k)+'_VREG',355,75)
    s.connector('J'+str(base+1),'12 shell electrodes plus ground / no shield over sensing pads',
                ['TOUCH_PAD_'+str(k*12+j+1).zfill(2) for j in range(12)]+['GND'],270,165)
    if k==0:r(s,'R'+str(base+2),'10k 1%','MCU_3V3','TOUCH_IRQ_N',95,230)
    s.text('Electrode-to-body map is defined in BOM-full-robot.md. Final harness labels must name the actual body locations.\nParts are procurement candidates; MPR121 lifecycle/assembly availability and alternate sourcing need confirmation.',15,270)
    sheets.append(s)
s=sheet('11-transmit-cut','Hardware-gated servo transmit enables',12)
s.text('SERVO DATA CUT / motor latch clear disables all four transmit drivers',15,15,1.8)
s.text('TX_ENABLE = software DE AND hardware MOTOR_ARMED. A stuck-high DE cannot override a cleared motor latch.\nPull-downs hold AHCT enables low while the 3.3 V gating supply is absent. MOTOR_ARMED has a local pull-down.\nMeasure actual servo data-pin backfeed with power off; this provision is not physical validation.',15,24)
cutgate=ic('SN74LVC08A',[(1,'1A','input'),(2,'1B','input'),(4,'2A','input'),(5,'2B','input'),
                       (9,'3A','input'),(10,'3B','input'),(12,'4A','input'),(13,'4B','input'),(7,'GND','power_in')],
           [(3,'1Y','output'),(6,'2Y','output'),(8,'3Y','output'),(11,'4Y','output'),(14,'VCC','power_in')],10.16)
cutnets={'7':'GND','14':'MCU_3V3'}
for g,(a,b,y) in zip(['LEG','UPPER','LEFT','RIGHT'],[(1,2,3),(4,5,6),(9,10,8),(12,13,11)]):
    cutnets.update({str(a):g+'_DE',str(b):'MOTOR_ARMED',str(y):g+'_TX_ENABLE'})
s.part(cutgate,'U1101','SN74LVC08APWR',100,105,cutnets,
       'Package_SO:TSSOP-14_4.4x5mm_P0.65mm','SN74LVC08APWR','https://www.ti.com/lit/ds/symlink/sn74lvc08a.pdf')
for k,g in enumerate(['LEG','UPPER','LEFT','RIGHT']):
    r(s,'R'+str(1101+k),'100k 1%',g+'_TX_ENABLE','GND',50+k*90,195)
r(s,'R1105','100k 1%','MOTOR_ARMED','GND',230,105)
c(s,'C1101','100n 16V','MCU_3V3',315,105)
sheets.append(s)

teensy_socket(ROOT/'LGV2_Footprints.pretty')
save_project(ROOT,sheets,'Little Guy V2 - spinal-cord board',
             'SPINAL CORD / CIRCUIT CAPTURE IN PROGRESS\nTeensy carrier, four TTL buses, SPI IMU, 48 touch inputs and isolated sensor branches.\nNo controller firmware, cable timing, stop behaviour or hardware has been validated; no fabrication release.')
