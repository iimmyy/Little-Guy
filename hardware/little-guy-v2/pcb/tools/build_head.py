"""XVF3800 circuit capture from XMOS pin/power tables; geometry and firmware open."""
from pathlib import Path
from schematic import Sheet,ic,save_project
from board_common import r,c,ldo
from chip_footprints import xmos_qf60b
from head_geometry import generate as generate_geometry

ROOT=Path(__file__).resolve().parents[1]/'head'
PROJECT='little-guy-head'
GEOMETRY=generate_geometry(ROOT)
DS='https://www.xmos.com/documentation/XM-014888-PC/pdf/xvf3800_datasheet_v3.2.1.pdf'
def sheet(name,title,page):return Sheet(name,title,page,PROJECT)
sheets=[]
s=sheet('01-voice-processor','XVF3800 / QSPI / USB / clock',2)
s.text('HEAD AUDIO / XVF3800 production silicon, USB accessory firmware, four independent PDM data lines',15,15,1.8)
s.text('Pin map: XMOS XVF3800 v3.2.1 Tables 4.1 and 6.1. QF60B has FOUR core-supply paddles plus one ground paddle.\n61-64 must connect to 0.9 V; 65 is ground. The local five-paddle footprint follows Fig.8.1 nominal geometry.\nUSB ESD, exact connector, XTAG4 adapter, land-pattern/stencil review and firmware remain release tasks.',15,24)
signals={1:('QSPI_D1','bidirectional','FLASH_D1'),2:('QSPI_D3','bidirectional','FLASH_D3'),
         3:('QSPI_CS_N','output','FLASH_CS_N'),5:('QSPI_CLK','output','FLASH_CLK'),
         6:('SPI_CS_N','input','SPI_BOOT_CS_N'),7:('SPI_MOSI','input','GND'),
         9:('I2S_DATA0','output',None),10:('I2S_LRCK','output',None),11:('BUTTON0','input','HEAD_BUTTON0'),
         13:('I2S_BCLK','output',None),14:('MCLK_INOUT','output','AUDIO_MCLK'),15:('XOUT','output',None),
         16:('XIN','input','HEAD_XIN'),18:('TDI','input','HEAD_TDI'),20:('TDO','output','HEAD_TDO'),
         21:('RST_N','input','HEAD_RESET_N'),22:('PLL_AVDD','power_in','HEAD_PLL_0V9'),
         23:('TMS','input','HEAD_TMS'),24:('TCK','input','HEAD_TCK'),25:('NC','no_connect',None),
         28:('USB_DM','bidirectional','HEAD_USB_DM'),29:('USB_DP','bidirectional','HEAD_USB_DP'),
         30:('USB_VDD33A','power_in','HEAD_3V3'),31:('USB_VDD18A','power_in','HEAD_1V8'),
         32:('BUTTON1','input','HEAD_BUTTON1'),33:('XL_DN1','input',None),35:('XL_DN0','input',None),
         36:('XL_UP0','output',None),37:('XL_UP1','output',None),39:('I2S_DATA1','input','GND'),
         40:('CODEC_RST_N','output',None),41:('MCLK','input','AUDIO_MCLK'),43:('MIC_CLK','output','PDM_CLK_RAW'),
         44:('I2C_SCL','bidirectional',None),45:('I2C_SDA','bidirectional',None),
         46:('MIC_DATA0','input','PDM_DATA0'),47:('SPI_MISO','output',None),48:('MIC_DATA2','input','PDM_DATA2'),
         50:('MIC_DATA1','input','PDM_DATA1'),51:('MIC_DATA3','input','PDM_DATA3'),
         53:('I2S_DATA2','output',None),54:('SQ_nLIN','output',None),55:('INT_N','output',None),
         56:('LED_R','output',None),58:('LED_G','output',None),59:('QSPI_D0','bidirectional','FLASH_D0'),
         60:('QSPI_D2','bidirectional','FLASH_D2'),65:('GND_PADDLE','power_in','GND')}
for pin in [4,12,19,27,34,42,49,57,61,62,63,64]:signals[pin]=('VDD_CORE','power_in','HEAD_0V9')
for pin in [8,38,52]:signals[pin]=('VDDIO_3V3','power_in','HEAD_3V3')
for pin in [17,26]:signals[pin]=('VDDIOB18','power_in','HEAD_1V8')
assert set(signals)==set(range(1,66))
left=[(i,*signals[i][:2]) for i in range(1,34)]
right=[(i,*signals[i][:2]) for i in range(34,66)]
s.part(ic('XVF3800_QF60B_5PADDLE',left,right,20.32),'U101','XVF3800-QF60B-C',95,105,
       {str(i):v[2] for i,v in signals.items()},'LGV2_Footprints:XMOS_QF60B_7x7mm_P0.4mm_5Paddle',mpn='XVF3800-QF60B-C',datasheet=DS)
flash=ic('W25Q32JV_SOIC8',[(1,'~{CS}','input'),(2,'DO_IO1','bidirectional'),(3,'~{WP}_IO2','bidirectional'),(4,'GND','power_in')],
         [(8,'VCC','power_in'),(7,'~{HOLD}_IO3','bidirectional'),(6,'CLK','input'),(5,'DI_IO0','bidirectional')],10.16)
s.part(flash,'U102','W25Q32JVSSIQ',260,80,
       {'1':'FLASH_CS_N','2':'FLASH_D1','3':'FLASH_D2','4':'GND','5':'FLASH_D0','6':'FLASH_CLK','7':'FLASH_D3','8':'HEAD_3V3'},
       'Package_SO:SOIC-8_5.3x5.3mm_P1.27mm','W25Q32JVSSIQ','https://www.winbond.com/hq/product/code-storage-flash-memory/serial-nor-flash/?partNo=W25Q32JV')
clock=ic('LVCMOS_Oscillator4',[(1,'OE','input'),(2,'GND','power_in')],[(3,'OUT','output'),(4,'VDD','power_in')],10.16)
s.part(clock,'Y101','SiT8008AI-71-18E-24.000000',260,155,
       {'1':'HEAD_1V8','2':'GND','3':'HEAD_CLK24_RAW','4':'HEAD_1V8'},
       mpn='SiT8008AI-71-18E-24.000000',datasheet='https://www.sitime.com/parts/sit8008ai-71-18e-24000000')
r(s,'R101','33 1% / source tuning','HEAD_CLK24_RAW','HEAD_XIN',350,155)
for k,(net,x,y) in enumerate([('FLASH_CS_N',190,210),('SPI_BOOT_CS_N',250,210),('HEAD_BUTTON0',310,210),('HEAD_BUTTON1',370,210)]):
    r(s,'R'+str(102+k),'4k7 1%' if k<2 else '10k 1%','HEAD_3V3',net,x,y)
c(s,'C101','100n 16V','HEAD_3V3',195,100)
c(s,'C102','100n 16V','HEAD_1V8',195,155)
s.connector('J101','Internal USB 2 device / ESD network pending',
            ['HEAD_USB_5V','HEAD_USB_DM','HEAD_USB_DP','GND'],65,215)
s.connector('J102','JTAG breakout / NOT an XSYS2 pin-compatible header',
            ['HEAD_1V8','GND','HEAD_TCK','HEAD_TMS','HEAD_TDI','HEAD_TDO','HEAD_RESET_N'],250,255)
r(s,'R106','10k 1%','HEAD_1V8','HEAD_TMS',45,265)
r(s,'R107','10k 1%','HEAD_1V8','HEAD_TDI',110,265)
r(s,'R108','100k 1%','HEAD_TCK','GND',175,265)
s.text('QSPI_D1 has no boot-select pull-up: boot from local flash. Set and verify nonvolatile Quad Enable before boot.\nUSB playback must carry the actual speaker signal as the echo reference; capture alone does not cancel the robot speakers.\nUnused pin settings and SDR sampling edge must match the custom firmware; no stock Seeed image is assumed compatible.',15,282)
sheets.append(s)

s=sheet('02-supplies','Audio supplies / core buck / PLL filtering',3)
s.text('SUPPLIES / USB 5 V -> 3.3 V, 1.8 V and 0.9 V; core enable is qualified by both I/O supplies',15,15,1.8)
s.text('XMOS power sequencing: core must remain off while 1.8 V is absent and 3.3 V is present.\nThe two I/O supervisors gate CORE_ENABLE; core supervisor then allows processor reset to release.\nThis circuit has not been measured. Rail timing, inrush, USB current and reset waveform are bench gates.',15,24)
ldo(s,'U201','HEAD_USB_5V','HEAD_3V3',80,85)
ldo(s,'U202','HEAD_USB_5V','HEAD_1V8',240,85,'18')
for k,(net,x,y) in enumerate([('HEAD_USB_5V',35,140),('HEAD_3V3',110,140),('HEAD_USB_5V',210,140),('HEAD_1V8',285,140)]):
    c(s,'C'+str(201+k),'10u 10V X7R','HEAD_USB_5V' if k%2==0 else net,x,y)
buck=ic('TPS62825_DMQ',[(1,'EN','input'),(2,'PG','open_collector'),(3,'FB','input')],
        [(4,'GND','power_in'),(5,'SW','output'),(6,'VIN','power_in')],10.16)
s.part(buck,'U203','TPS62825ADMQR',80,210,
       {'1':'HEAD_CORE_ENABLE','2':None,'3':'HEAD_CORE_FB','4':'GND','5':'HEAD_CORE_SW','6':'HEAD_USB_5V'},
       mpn='TPS62825ADMQR',datasheet='https://www.ti.com/lit/ds/symlink/tps62825a.pdf')
s.passive('L','L201','0u47 / XFL4015-471MEB','HEAD_CORE_SW','HEAD_0V9',180,195,mpn='XFL4015-471MEB')
r(s,'R201','49k9 0.1%','HEAD_0V9','HEAD_CORE_FB',250,195)
r(s,'R202','100k 0.1%','HEAD_CORE_FB','GND',315,195)
r(s,'R203','100k 1%','HEAD_CORE_ENABLE','GND',375,195)
c(s,'C205','4u7 10V','HEAD_USB_5V',180,250)
c(s,'C206','10u 6V3 X7R','HEAD_0V9',250,250)
c(s,'C207','10u 6V3 X7R','HEAD_0V9',315,240)
c(s,'C208','120p C0G / loop check','HEAD_0V9',375,240,'HEAD_CORE_FB')
s.power_flag('HEAD_USB_5V',365,65);s.power_flag('GND',365,125)
s.power_flag('HEAD_0V9',80,265)
sheets.append(s)

s=sheet('03-power-supervision','Three-rail supervision / reset logic',4)
s.text('POWER-UP / independent voltage checks; all supervisor outputs are pulled up to the 1.8 V logic rail',15,15,1.8)
s.text('TPS3808G01: 0.405 V threshold, CT floating gives 20 ms delay. Supervisors are supplied from USB 5 V.\nCORE_ENABLE = IO18_OK AND IO33_OK. RESET_N = CORE_ENABLE AND CORE_OK.\nNominal thresholds: 1.687 V / 3.035 V / 0.863 V. Include regulator and supervisor tolerances in the release calculation.',15,24)
sup=ic('TPS3808_DBV',[(1,'~{RESET}','open_collector'),(2,'GND','power_in'),(3,'~{MR}','input')],
       [(4,'CT','input'),(5,'SENSE','input'),(6,'VDD','power_in')],10.16)
for k,(net,pg,top,bottom) in enumerate([('HEAD_1V8','HEAD_IO18_OK','100k 0.1%','31k6 0.1%'),
                                       ('HEAD_3V3','HEAD_IO33_OK','100k 0.1%','15k4 0.1%'),
                                       ('HEAD_0V9','HEAD_CORE_OK','11k3 0.1%','10k 0.1%')]):
    x=65+k*125;sense='HEAD_SENSE'+str(k)
    s.part(sup,'U'+str(301+k),'TPS3808G01DBVR',x,85,
           {'1':pg,'2':'GND','3':'HEAD_USB_5V','4':None,'5':sense,'6':'HEAD_USB_5V'},
           'Package_TO_SOT_SMD:SOT-23-6','TPS3808G01DBVR','https://www.ti.com/lit/ds/symlink/tps3808.pdf')
    r(s,'R'+str(301+k*3),top,net,sense,x-20,140)
    r(s,'R'+str(302+k*3),bottom,sense,'GND',x+25,140)
    r(s,'R'+str(303+k*3),'10k 1%','HEAD_1V8',pg,x-20,195)
    c(s,'C'+str(301+k),'100n 16V','HEAD_USB_5V',x+25,195)
gate=ic('SN74LVC08A',[(1,'1A','input'),(2,'1B','input'),(4,'2A','input'),(5,'2B','input'),
                     (9,'3A','input'),(10,'3B','input'),(12,'4A','input'),(13,'4B','input'),(7,'GND','power_in')],
        [(3,'1Y','output'),(6,'2Y','output'),(8,'3Y','output'),(11,'4Y','output'),(14,'VCC','power_in')],10.16)
s.part(gate,'U304','SN74LVC08APWR / powered at 1.8 V',100,250,
       {'1':'HEAD_IO18_OK','2':'HEAD_IO33_OK','3':'HEAD_CORE_ENABLE','4':'HEAD_CORE_ENABLE',
        '5':'HEAD_CORE_OK','6':'HEAD_RESET_N','7':'GND','8':None,'9':'GND','10':'GND','11':None,
        '12':'GND','13':'GND','14':'HEAD_1V8'},'Package_SO:TSSOP-14_4.4x5mm_P0.65mm',
       'SN74LVC08APWR','https://www.ti.com/lit/ds/symlink/sn74lvc08a.pdf')
c(s,'C304','100n 16V','HEAD_1V8',210,245)
r(s,'R310','100k 1%','HEAD_RESET_N','GND',275,240)
sheets.append(s)

s=sheet('04-decoupling','Processor local decoupling / PLL filter',5)
s.text('LOCAL DECOUPLING / one low-inductance bypass per supply pin, four core paddles on 0.9 V',15,15,1.8)
s.text('XMOS recommends at least eight 100n core bypass capacitors plus >=10u bulk on the core and I/O rails.\nUse direct short returns to the central ground paddle, never route 61-64 to ground.\nPlacement and via geometry still require layout review; schematic capacitance does not establish supply integrity.',15,24)
for k in range(8):c(s,'C'+str(401+k),'100n 10V 0402','HEAD_0V9',40+(k%4)*90,75+(k//4)*55)
for k,net in enumerate(['HEAD_3V3']*4+['HEAD_1V8']*3):
    c(s,'C'+str(409+k),'100n 10V 0402',net,40+(k%4)*90,185+(k//4)*45)
s.passive('L','FB401','BKH1005LM601-T / 600R @100MHz','HEAD_0V9','HEAD_PLL_0V9',50,260,mpn='BKH1005LM601-T')
c(s,'C416','1u 6V3','HEAD_PLL_0V9',180,260)
s.power_flag('HEAD_PLL_0V9',260,260)
sheets.append(s)

s=sheet('05-microphones','Four PDM microphones / head-A geometry targets',6)
s.text('FOUR INDEPENDENT PDM MICROPHONES / IM69D130 / each data line goes to one XVF3800 input',15,15,1.8)
s.text('Shared 3.072 MHz clock; all SELECT pins grounded. Custom firmware must use the channel order below.\n'
       f"Frozen head-A targets: {GEOMETRY['left_right_board_plane_mm']:.3f} mm left/right, {GEOMETRY['front_back_board_plane_mm']:.3f} mm front/back on an {GEOMETRY['tilt_front_up_deg']:.3f} deg plane.\n"
       'See microphone-geometry.json and mic_geometries.fragment.yaml. Outline, package offsets and gaskets remain open.',15,24)
mic=ic('IM69D130',[(1,'DATA','output'),(2,'VDD','power_in'),(3,'CLOCK','input')],
       [(4,'SELECT','input'),(5,'GND','power_in')],10.16)
r(s,'R501','33 1% / scope tuning','PDM_CLK_RAW','PDM_CLK',65,65)
for k in range(4):
    x=65+(k%2)*190;y=120+(k//2)*95
    m=GEOMETRY['microphones'][k]
    ax,ay,az=m['array_plane_mm']
    s.text(f"MIC{k}: {m['location'].upper()} / array X={ax:.3f}, Y={ay:.3f} mm; Z=0",x-30,y-25,1.4)
    s.part(mic,'MK'+str(501+k),'IM69D130V01XTSA1 / '+m['location'].upper(),x,y,
           {'1':'PDM_DATA'+str(k),'2':'HEAD_3V3','3':'PDM_CLK','4':'GND','5':'GND'},
           mpn='IM69D130V01XTSA1',datasheet='https://www.infineon.com/assets/row/public/documents/24/49/infineon-im69d130-datasheet-en.pdf')
    c(s,'C'+str(501+k),'100n 16V','HEAD_3V3',x+75,y)
s.text('Clock and data must remain clear of switching nodes. Keep all four acoustic openings on the same outside surface.\nBeamforming, voice direction and speaker echo cancellation are untested until custom firmware is loaded and tuned.\nGNSS antenna is not on this draft; determine antenna-module ground-plane/keep-out requirements before adding one.',15,275)
sheets.append(s)
xmos_qf60b(ROOT/'LGV2_Footprints.pretty')
save_project(ROOT,sheets,'Little Guy V2 - head microphone board',
             'HEAD MICROPHONE BOARD / CIRCUIT CAPTURE IN PROGRESS\n'
             f"Frozen head-A targets: rigid plane, {GEOMETRY['tilt_front_up_deg']:.3f} deg front-up tilt, explicit PDM channel order.\n"
             'USB protection, package/outline/mounting, firmware and acoustic tests remain open; no fabrication release.')
