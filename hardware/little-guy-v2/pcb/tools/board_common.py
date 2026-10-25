"""Shared schematic helpers; values remain procurement candidates until reviewed."""
from schematic import ic

RFP='Resistor_SMD:R_0603_1608Metric'
CFP='Capacitor_SMD:C_0603_1608Metric'

def r(s,ref,value,a,b,x,y):
    s.passive('R',ref,value,a,b,x,y,RFP)

def c(s,ref,value,a,x,y,b='GND'):
    s.passive('C',ref,value,a,b,x,y,CFP)

def ldo(s,ref,vin,vout,x,y,voltage='33'):
    d=ic('TLV755P_DBV',[(1,'IN','power_in'),(2,'GND','power_in'),(3,'EN','input')],
         [(4,'NC','no_connect'),(5,'OUT','power_out')],7.62)
    s.part(d,ref,'TLV755'+voltage+'PDBVR',x,y,
           {'1':vin,'2':'GND','3':vin,'4':None,'5':vout},
           'Package_TO_SOT_SMD:SOT-23-5','TLV755'+voltage+'PDBVR',
           'https://www.ti.com/lit/ds/symlink/tlv755p.pdf')

def vl53(s,ref,supply,scl,sda,irq,shutdown,x,y):
    d=ic('VL53L4CD',[(1,'AVDDVCSEL','power_in'),(2,'AVSSVCSEL','power_in'),(3,'GND','passive'),
                      (4,'GND2','passive'),(5,'XSHUT','input'),(6,'GND3','passive')],
         [(7,'GPIO1','open_collector'),(8,'DNC','no_connect'),(9,'SDA','bidirectional'),
          (10,'SCL','input'),(11,'AVDD','power_in'),(12,'GND4','passive')],10.16)
    s.part(d,ref,'VL53L4CDV0DH/1',x,y,
           {'1':supply,'2':'GND','3':'GND','4':'GND','5':shutdown,'6':'GND',
            '7':irq,'8':None,'9':sda,'10':scl,'11':supply,'12':'GND'},
           mpn='VL53L4CDV0DH/1',datasheet='https://www.st.com/resource/en/datasheet/vl53l4cd.pdf')
