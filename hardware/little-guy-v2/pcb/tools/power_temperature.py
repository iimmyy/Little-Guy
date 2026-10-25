"""Three independent NTC windows with wired-AND charge permission."""
from schematic import Sheet, ic

def add_temperature(sheets, resistor, capacitor):
    t=Sheet('12-pack-temperature','Three-segment charge temperature interlock',13)
    t.text('HARDWARE TEMPERATURE PERMISSION / one 103AT-2 on each battery segment',15,15,1.8)
    t.text('Ratiometric windows: NTC=103AT-2; nominal acceptance about 1.1-43.2 C from the published R/T table.\nOpen or shorted sensor inhibits charging. Tolerance and actual switching points require bench verification.\nThe charger has an additional dedicated TS thermistor; this window does not replace pack protection.',15,24)
    # TLV1704 PW package, SBOS589D pin-function table. All six fault outputs share TEMP_OK.
    comp=ic('TLV1704PW',[(4,'1IN-','input'),(5,'1IN+','input'),(6,'2IN-','input'),(7,'2IN+','input'),
                        (8,'3IN-','input'),(9,'3IN+','input'),(10,'4IN-','input'),(11,'4IN+','input')],
            [(1,'2OUT','open_collector'),(2,'1OUT','open_collector'),(3,'V+','power_in'),
             (12,'V-','power_in'),(13,'4OUT','open_collector'),(14,'3OUT','open_collector')],10.16)
    first={'4':'TEMP_HOT_REF','5':'NTC_SEG1','6':'NTC_SEG1','7':'TEMP_COLD_REF',
           '8':'TEMP_HOT_REF','9':'NTC_SEG2','10':'NTC_SEG2','11':'TEMP_COLD_REF',
           '1':'PACK_TEMP_OK','2':'PACK_TEMP_OK','3':'AON_3V3','12':'GND','13':'PACK_TEMP_OK','14':'PACK_TEMP_OK'}
    second={'4':'TEMP_HOT_REF','5':'NTC_SEG3','6':'NTC_SEG3','7':'TEMP_COLD_REF',
            '8':'AON_3V3','9':'GND','10':'AON_3V3','11':'GND',
            '1':'PACK_TEMP_OK','2':'PACK_TEMP_OK','3':'AON_3V3','12':'GND','13':None,'14':None}
    for k,nets in enumerate([first,second]):
        t.part(comp,'U'+str(1201+k),'TLV1704PWR',85+k*150,95,nets,
               'Package_SO:TSSOP-14_4.4x5mm_P0.65mm','TLV1704PWR','https://www.ti.com/lit/ds/symlink/tlv1704.pdf')
    refs=[('R1201','19k1 0.1%','AON_3V3','TEMP_HOT_REF'),('R1202','10k 0.1%','TEMP_HOT_REF','GND'),
          ('R1203','10k 0.1%','AON_3V3','TEMP_COLD_REF'),('R1204','26k1 0.1%','TEMP_COLD_REF','GND'),
          ('R1205','10k 1%','AON_3V3','PACK_TEMP_OK')]
    for k,(ref,val,a,b) in enumerate(refs):
        resistor(t,ref,val,a,b,35+k*70,175)
    for k in range(3):
        net='NTC_SEG'+str(k+1)
        resistor(t,'R'+str(1206+k),'10k 0.1%','AON_3V3',net,35+k*100,230)
        capacitor(t,'C'+str(1201+k),'100n 16V',net,75+k*100,230)
        t.connector('J'+str(1201+k),'103AT-2 segment '+str(k+1),[net,'GND'],45+k*100,265)
    capacitor(t,'C1204','100n 16V','AON_3V3',345,175)
    capacitor(t,'C1205','100n 16V','AON_3V3',345,230)
    t.text('Cold reference = 26.1 / (10 + 26.1) of the same 3.3 V rail as NTC pullups.\nHot reference = 10 / (19.1 + 10). Fault comparators pull TEMP_OK low directly.\nSensor harness opens/shorts and brownout behaviour must be tested before charging a pack.',15,145)
    sheets.append(t)
