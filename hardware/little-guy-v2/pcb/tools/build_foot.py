"""Two identical foot assemblies; separate spinal I2C mux branches."""
from pathlib import Path
from schematic import Sheet,ic,save_project
from board_common import r,c,ldo,vl53

ROOT=Path(__file__).resolve().parents[1]/'foot'
PROJECT='little-guy-foot'
def sheet(name,title,page):return Sheet(name,title,page,PROJECT)
sheets=[]
s=sheet('01-foot-interface','Foot power / single cable / toe ranging',2)
s.text('FOOT / identical PCB fitted twice; each foot has its own spinal I2C mux channel',15,15,1.8)
s.text('5 V cable supply; local 3.3 V powers the bridges and both ADCs. ToF uses a separate local 3.3 V regulator.\nCable order is fixed here and matched by the spinal schematic. Connector family and ESD parts remain open.\nADC addresses 0x40 / 0x41, toe 0x29 (7-bit). Start at 100 kHz; qualify the complete leg cable before 400 kHz.',15,24)
s.connector('J101','Spinal to foot / 6 conductors',
            ['FOOT_5V','GND','FOOT_SCL','FOOT_SDA','FOOT_READY_N','TOE_XSHUT'],65,90)
ldo(s,'U101','FOOT_5V','BRIDGE_3V3',190,80)
ldo(s,'U102','FOOT_5V','TOF_3V3',315,80,'33')
for k,(net,x) in enumerate([('FOOT_5V',45),('BRIDGE_3V3',150),('FOOT_5V',255),('TOF_3V3',355)]):
    c(s,'C'+str(101+k),'4u7 10V X7R / verify effective C',net,x,145)
vl53(s,'U103','TOF_3V3','FOOT_SCL','FOOT_SDA','FOOT_READY_N','TOE_XSHUT',105,220)
c(s,'C105','100n 16V','TOF_3V3',210,215)
c(s,'C106','4u7 10V','TOF_3V3',255,215)
r(s,'R101','10k 1%','TOF_3V3','TOE_XSHUT',310,215)
r(s,'R102','10k 1%','BRIDGE_3V3','FOOT_READY_N',360,215)
s.power_flag('FOOT_5V',70,155);s.power_flag('GND',100,155)
s.text('ToF AVDD=3.3 V is within ST 2.6-3.5 V operating range; configure matching I/O mode.\nShared READY_N is wired-OR: firmware must service both ADCs and the ToF interrupt before returning to sleep.\nKeep optical aperture clear of sole material; final window, light barrier and crosstalk calibration are not designed.',15,270)
sheets.append(s)

d=ic('ADS122C04_PW',[(1,'A0','input'),(2,'A1','input'),(3,'~{RESET}','input'),(4,'DGND','power_in'),
                     (5,'AVSS','power_in'),(6,'AIN3','input'),(7,'AIN2','input'),(8,'REFN','input')],
     [(9,'REFP','input'),(10,'AIN1','input'),(11,'AIN0','input'),(12,'AVDD','power_in'),
      (13,'DVDD','power_in'),(14,'~{DRDY}','open_collector'),(15,'SDA','bidirectional'),(16,'SCL','input')],12.7)
for bank in range(2):
    base=200+bank*100
    s=sheet('0'+str(bank+2)+'-bridges-'+str(bank+1),'Bridge pair '+str(bank+1)+' / ratiometric ADC',bank+3)
    s.text('TWO FULL-BRIDGE CELLS / ADS122C04 uses excitation as external voltage reference',15,15,1.8)
    s.text('TSSOP-16 pin map: ADS122C04 SBAS751B Table 5. A1=GND, A0='+('GND' if bank==0 else 'DVDD')+'.\nSelect external reference, PGA gain 128 initially; alternate AIN0-AIN1 and AIN2-AIN3 after single-cycle settling.\nTarget >=100 settled samples/s/cell; resolution, conversion rate and I2C timing must be measured with the actual cells.',15,24)
    pins={'1':'GND' if bank==0 else 'BRIDGE_3V3','2':'GND','3':'BRIDGE_3V3','4':'GND','5':'GND',
          '6':'ADC'+str(bank)+'_AIN3','7':'ADC'+str(bank)+'_AIN2','8':'GND','9':'BRIDGE_3V3',
          '10':'ADC'+str(bank)+'_AIN1','11':'ADC'+str(bank)+'_AIN0','12':'BRIDGE_3V3','13':'BRIDGE_3V3',
          '14':'FOOT_READY_N','15':'FOOT_SDA','16':'FOOT_SCL'}
    s.part(d,'U'+str(base+1),'ADS122C04IPWR / '+hex(0x40+bank),95,95,pins,
           'Package_SO:TSSOP-16_4.4x5mm_P0.65mm','ADS122C04IPWR','https://www.ti.com/lit/ds/symlink/ads122c04.pdf')
    for k in range(2):
        cell=bank*2+k+1
        pref='CELL'+str(cell)
        s.connector('J'+str(base+1+k),'Load cell '+str(cell)+' / full bridge',
                    ['BRIDGE_3V3','GND',pref+'_P',pref+'_N'],240+k*95,80)
        for sign,m in [('P',0),('N',1)]:
            r(s,'R'+str(base+1+k*2+m),'1k 0.1%',pref+'_'+sign,'ADC'+str(bank)+'_AIN'+str(k*2+m),225+k*100+m*40,155)
        c(s,'C'+str(base+1+k),'10n C0G 16V','ADC'+str(bank)+'_AIN'+str(k*2),245+k*100,215,'ADC'+str(bank)+'_AIN'+str(k*2+1))
    for k,x in enumerate([45,95,145]):c(s,'C'+str(base+3+k),'100n 16V','BRIDGE_3V3',x,215)
    s.text('REFP/REFN return to the same local excitation star as the cell connector supplies (Kelvin routing).\nEach ADC multiplexes two cells; readings have separate timestamps. Do not claim simultaneous sampling.\n1k + 1k / 10n differential filters are initial values. Bridge impedance, settling and noise require bench validation.\nExact full-bridge load cell, excitation current, connector and overload mounting stops remain selection tasks.',15,270)
    sheets.append(s)
save_project(ROOT,sheets,'Little Guy V2 - foot board',
             'FOOT BOARD / CIRCUIT CAPTURE IN PROGRESS\nTwo ADCs, four full bridges and toe ToF. No load cell or cable has been physically tested.\nNo layout or fabrication release; footprints, ESD and mechanics remain open.')
