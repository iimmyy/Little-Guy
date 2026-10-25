"""Power-stage circuit capture. Values marked pending are not released parts."""
from schematic import Sheet, ic

def add_regulators(sheets, resistor, capacitor):
    m = Sheet('04-motor-controller','Motor buck-boost control',5)
    m.text('MOTOR BUS / 11.28 V nominal / 20 A aggregate current-limit target',15,15,1.8)
    m.text('LM5176PWP pin map: SNVSAI1D Table 5-1. Starting power stage: TI PMP21278.\nFeedback and output-current shunt changed from reference. Loop/transient/thermal tests required.\n11.28 V target provides margin below STS3250 12 V maximum; overshoot is not yet validated.',15,24)
    left=[(1,'EN/UVLO','input'),(2,'VIN','power_in'),(3,'VINSNS','input'),(4,'MODE','input'),
          (5,'DITH','input'),(6,'RT/SYNC','input'),(7,'SLOPE','input'),(8,'SS','input'),(9,'COMP','output'),
          (10,'AGND','power_in'),(11,'FB','input'),(12,'VOSNS','input'),(13,'ISNS-','input'),(14,'ISNS+','input'),
          (15,'CSG','input'),(16,'CS','input')]
    right=[(17,'PGOOD','open_collector'),(18,'SW2','passive'),(19,'HDRV2','output'),(20,'BOOT2','passive'),
           (21,'LDRV2','output'),(22,'PGND','power_in'),(23,'VCC','power_out'),(24,'BIAS','power_in'),
           (25,'LDRV1','output'),(26,'BOOT1','passive'),(27,'HDRV1','output'),(28,'SW1','passive'),(29,'PAD','passive')]
    nets={'1':'MOTOR_REG_EN','2':'MR_VIN_BIAS','3':'MR_VINSNS','4':'MR_MODE','5':'GND','6':'MR_RT',
          '7':'MR_SLOPE','8':'MR_SS','9':'MR_COMP','10':'GND','11':'MR_FB','12':'MOTOR_REG_RAW',
          '13':'MR_ISNS_N','14':'MR_ISNS_P','15':'MR_CSG','16':'MR_CS','17':'MOTOR_REG_PG',
          '18':'MR_SW2','19':'MR_HDRV2','20':'MR_BOOT2','21':'MR_LDRV2','22':'GND','23':'MR_VCC',
          '24':'MOTOR_11V3','25':'MR_LDRV1','26':'MR_BOOT1','27':'MR_HDRV1','28':'MR_SW1','29':'GND'}
    m.part(ic('LM5176PWP',left,right,12.7),'U401','LM5176PWPR',85,95,nets,
           mpn='LM5176PWPR',datasheet='https://www.ti.com/lit/ds/symlink/lm5176.pdf')
    values=[('R401','2k 1%','PACK_PROTECTED','MR_VINSNS'),
            ('R402','10 1%','MR_BIAS_DIODE','MR_VIN_BIAS'),
            ('R403','33k2 1%','MR_RT','GND'),('R404','93k1 1%','MR_MODE','GND'),
            ('R405','6k04 1%','MR_COMP_RC','GND'),('R406','262k 0.1%','MOTOR_11V3','MR_FB'),
            ('R407','20k 0.1%','MR_FB','GND'),('R408','10k 1%','AON_3V3','MOTOR_REG_PG'),
            ('R409','100 1%','MR_SENSE_HIGH','MR_CS'),('R410','100 1%','GND','MR_CSG'),
            ('R411','100 1%','MOTOR_REG_RAW','MR_ISNS_P'),('R412','100 1%','MOTOR_11V3','MR_ISNS_N'),
            ('R413','100k 1%','MOTOR_REG_EN','GND')]
    for j,(ref,val,a,b) in enumerate(values):
        resistor(m,ref,val,a,b,170+(j%6)*36,60+(j//6)*62)
    caps=[('C401','100n 50V','MR_VIN_BIAS','GND'),('C402','4u7 16V','MR_VCC','GND'),
          ('C403','150p C0G','MR_RT','GND'),('C404','33n 50V','MR_COMP','MR_COMP_RC'),
          ('C405','1n2 C0G','MR_COMP','GND'),('C406','22n 50V','MR_SS','GND'),
          ('C407','220p C0G','MR_SLOPE','GND'),('C408','47p C0G','MR_CS','MR_CSG'),
          ('C409','330n 50V','MR_ISNS_P','MR_ISNS_N'),('C410','220n 16V','MR_BOOT1','MR_SW1'),
          ('C411','220n 16V','MR_BOOT2','MR_SW2')]
    for j,(ref,val,a,b) in enumerate(caps):
        x,y=(120,185) if j==10 else (170+(j%6)*36,215+(j//6)*35)
        capacitor(m,ref,val,a,x,y,b)
    diode=ic('Diode_AK',[(1,'A','passive')],[(2,'K','passive')],3.81)
    for ref,a,b,x in [('D401','PACK_PROTECTED','MR_BIAS_DIODE',30),
                      ('D402','MR_VCC','MR_BOOT1',75),('D403','MR_VCC','MR_BOOT2',120)]:
        m.part(diode,ref,'Schottky bootstrap / exact selection pending',x,230,{'1':a,'2':b})
    m.connector('J401','Regulator enable / independent motor-cut gate', ['MOTOR_REG_EN','GND','MOTOR_REG_PG'],40,165)
    m.text('Regulator input: protected pack, not charger SYS.\nEN has a hardware pull-down and comes from the motor-arm gate.\nFinal inductor, current limit and output transient protection are not released.',15,275)
    m.power_flag('MR_VIN_BIAS',300,190)
    m.power_flag('MOTOR_11V3',370,190)

    q=Sheet('05-motor-bridge','Motor buck-boost bridge',6)
    q.text('4-switch bridge / matched parallel FETs / Kelvin current sensing',15,15,1.8)
    q.text('Based on PMP21278; this implementation is not tested. Source reference used external input damping.\nReturned motor energy, branch switching/fusing and emergency-cut circuitry remain incomplete.\nNo motor connection or manufacturing release from these draft sheets.',15,24)
    nf=ic('Power_NMOS_8',[(1,'S','passive'),(2,'S','passive'),(3,'S','passive'),(4,'G','input')],
          [(5,'D','passive'),(6,'D','passive'),(7,'D','passive'),(8,'D','passive')],5.08)
    stages=[('H1','PACK_PROTECTED','MR_SW1','MR_HDRV1','CSD19502Q5B'),
            ('L1','MR_SW1','MR_SENSE_HIGH','MR_LDRV1','CSD19502Q5B'),
            ('H2','MOTOR_REG_RAW','MR_SW2','MR_HDRV2','CSD17573Q5B'),
            ('L2','MR_SW2','MR_SENSE_HIGH','MR_LDRV2','CSD17573Q5B')]
    for group,(tag,drain,source,gate,mpn) in enumerate(stages):
        for k in range(2):
            index=group*2+k
            ref='Q'+str(501+index)
            gn='MR_GATE_'+tag+'_'+str(k)
            q.part(nf,ref,mpn,45+group*95,70+k*60,
                   {**{str(i):source for i in [1,2,3]},'4':gn,**{str(i):drain for i in [5,6,7,8]}},
                   mpn=mpn,datasheet='https://www.ti.com/product/'+mpn)
            resistor(q,'R'+str(501+index),'0 / gate resistor tuning',gate,gn,45+group*95,177+k*35)
    q.passive('L','L501','2uH high-current; exact part from BOM review','MR_SW1','MR_SW2',40,250)
    for k,x in enumerate([100,145]):
        q.passive('R','R'+str(509+k),'5m 1% / pulse and thermal rating pending','MR_SENSE_HIGH','GND',x,250,
                  'Resistor_SMD:R_2512_6332Metric')
    for k,x in enumerate([200,245]):
        q.passive('R','R'+str(511+k),'5m 1% / output shunt, 20 A aggregate target','MOTOR_REG_RAW','MOTOR_11V3',x,250,
                  'Resistor_SMD:R_2512_6332Metric')
    # Capacitor bank gets its own sheet to preserve readable circuit spacing.
    c=Sheet('06-motor-capacitors','Motor reservoir / damping',7)
    c.text('Motor power-stage reservoirs; capacitance and damping require final part data',15,15,1.8)
    for j in range(12):
        capacitor(c,'C'+str(501+j),'4u7 50V X7R','PACK_PROTECTED',35+(j%6)*65,65+(j//6)*55,bulk=True)
    for j in range(6):
        capacitor(c,'C'+str(513+j),'22u 25V X7R','MOTOR_REG_RAW',35+j*65,185,bulk=True)
    for j in range(3):
        capacitor(c,'C'+str(519+j),'330u 25V polymer / exact ESR pending','MOTOR_11V3',35+j*85,245,bulk=True)
    c.connector('J601','External input damping test point', ['PACK_PROTECTED','GND'],350,235)
    c.text('The reference test used three 6.8 mF input capacitors outside the board.\nThat test condition is not silently adopted as a small on-board solution.\nSelect damping for the actual battery leads and measure ringing before layout release.',15,270)
    sheets.extend([m,q,c])

    # A dedicated computer regulator keeps computer power on its own branch.
    j=Sheet('07-jetson-feed','Regulated Jetson supply',8)
    j.text('JETSON / 12 V, 3 A current-limit target / source is charger logic SYS',15,15,1.8)
    j.text('TPS552892 Rev A application 9-36 V to 12 V / 3 A. 8 A is inductor-current capability, not output rating.\nInitial compensation below is calculated for 100 uF effective Cout; actual part bias/ESR must be checked.\nCarrier pinout, hardware shutdown/wake sequencing and transient hold-up remain release tasks.',15,24)
    lp=[(1,'EN/UVLO','input'),(2,'MODE','input'),(3,'PG','open_collector'),(4,'CC','open_collector'),
        (5,'DITH/SYNC','input'),(6,'FSW','input'),(7,'VIN','power_in'),(9,'PGND','power_in'),
        (14,'FB','input'),(15,'COMP','output'),(16,'CDC','output'),(17,'AGND','power_in'),(21,'EXTVCC','input')]
    rp=[(8,'SW1','output'),(10,'SW2','output'),(11,'VOUT','power_out'),(12,'ISP','input'),(13,'ISN','input'),
        (18,'VCC','power_out'),(19,'BOOT2','passive'),(20,'BOOT1','passive')]
    j.part(ic('TPS552892',lp,rp,12.7),'U701','TPS552892RYQR',75,95,
           {'1':'JETSON_EN','2':'GND','3':'JETSON_PG_N','4':'JETSON_CC_N','5':'GND','6':'JREG_FSW',
            '7':'LOGIC_SYS','9':'GND','14':'JREG_FB','15':'JREG_COMP','16':None,'17':'GND','21':None,
            '8':'JREG_SW1','10':'JREG_SW2','11':'JETSON_RAW','12':'JETSON_RAW','13':'JETSON_12V',
            '18':'JREG_VCC','19':'JREG_BOOT2','20':'JREG_BOOT1'},
           mpn='TPS552892RYQR',datasheet='https://www.ti.com/lit/ds/symlink/tps552892.pdf')
    j.passive('L','L701','4u7 / >=10 A saturation target','JREG_SW1','JREG_SW2',175,65)
    j.passive('R','R701','16m5 1% 1W / current limit','JETSON_RAW','JETSON_12V',225,65,'Resistor_SMD:R_2512_6332Metric')
    for ref,val,a,b,x,y in [('R702','100k 0.1%','JETSON_12V','JREG_FB',280,65),
                          ('R703','11k1 0.1%','JREG_FB','GND',325,65),('R704','49k9 1%','JREG_FSW','GND',175,120),
                          ('R705','4k87 1% / initial','JREG_COMP','JREG_COMP_RC',225,120),
                          ('R706','100k 1%','JETSON_EN','GND',280,120),
                          ('R707','10k 1%','AON_3V3','JETSON_PG_N',325,120),
                          ('R708','10k 1%','AON_3V3','JETSON_CC_N',370,120)]:
        resistor(j,ref,val,a,b,x,y)
    for k in range(2):
        capacitor(j,'C'+str(701+k),'22u 25V X7R','LOGIC_SYS',175+k*45,180,bulk=True)
    for k in range(4):
        capacitor(j,'C'+str(703+k),'47u 25V X7R / verify effective capacitance','JETSON_RAW',175+k*55,235,bulk=True)
    for ref,val,a,b,x,y in [('C707','100n 16V','JREG_BOOT1','JREG_SW1',30,180),
                          ('C708','100n 16V','JREG_BOOT2','JREG_SW2',75,180),
                          ('C709','4u7 10V','JREG_VCC','GND',120,180),
                          ('C710','39n 50V / initial','JREG_COMP_RC','GND',30,240),
                          ('C711','220p C0G / initial','JREG_COMP','GND',75,240),
                          ('C712','100n 50V','LOGIC_SYS','GND',120,240)]:
        capacitor(j,ref,val,a,x,y,b)
    j.connector('J701','Jetson carrier power / connector selection pending',['JETSON_12V','GND'],345,180)
    j.connector('J702','Spinal-cord power sequencing',['JETSON_EN','JETSON_PG_N','JETSON_CC_N','GND'],345,210)
    j.text('12 V output from 100k / 11k1 divider is 12.011 V nominal.\n3.03 A nominal output current limit from 50 mV / 16.5 mOhm.\nThis branch does not establish whole-system brownout immunity.',15,270)
    sheets.append(j)
