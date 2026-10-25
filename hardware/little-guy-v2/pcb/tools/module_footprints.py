"""Package preparation, not PCB placement or a fabrication release."""
from schematic import n,S,pretty,effects

TEENSY_TOP=['VIN','GND1','3V3A']+['GPIO'+str(i) for i in range(23,12,-1)]+['GND2']+['GPIO'+str(i) for i in range(41,32,-1)]
TEENSY_BOTTOM=['GND0']+['GPIO'+str(i) for i in range(13)]+['3V3B']+['GPIO'+str(i) for i in range(24,33)]

def teensy_socket(directory):
    """Two 24-way rows, 2.54 mm pitch, 15.24 mm separation, USB at -X.

    Reference: PJRC dimensions_teensy41.png and card11a_rev4_web.pdf.
    Module body 60.96 x 17.78 mm; socket hardware and physical fit remain open.
    """
    directory.mkdir(parents=True,exist_ok=True)
    assert len(TEENSY_TOP)==len(TEENSY_BOTTOM)==24
    assert len(set(TEENSY_TOP+TEENSY_BOTTOM))==48
    fp=n('footprint','Teensy41_Socket_GPIO_ID',n('version',20260206),n('generator','pcbnew'),
         n('layer','F.Cu'),n('descr','Teensy 4.1 carrier socket pattern; PJRC 60.96x17.78 mm, 2x24, 2.54 mm pitch, 15.24 mm rows; named pads match LGV2 symbol. Candidate, no physical fit test.'),
         n('tags','Teensy 4.1 socket GPIO named pads'),
         n('property','Reference','REF**',n('at',0,-11.0),n('layer','F.SilkS'),effects(1)),
         n('property','Value','Teensy41_Socket_GPIO_ID',n('at',0,11.0),n('layer','F.Fab'),effects(1)),
         n('attr',S('through_hole')))
    for layer,corner_a,corner_b,width in [
        ('F.Fab',(-30.48,-8.89),(30.48,8.89),.1),
        ('F.SilkS',(-30.6,-9.05),(30.6,9.05),.12),
        ('F.CrtYd',(-31.45,-9.2),(30.75,9.2),.05),
        ('Dwgs.User',(-45.48,-5.5),(-31.18,5.5),.15)]:
        fp.append(n('fp_rect',n('start',*corner_a),n('end',*corner_b),n('stroke',n('width',width),n('type',S('default'))),n('fill',S('none')),n('layer',layer)))
    fp.append(n('fp_text',S('user'),'USB / reserve cable space',n('at',-38.5,0,90),n('layer','Dwgs.User'),effects(.8)))
    fp.append(n('fp_text',S('user'),'VIN',n('at',-29.21,-10.4),n('layer','F.SilkS'),effects(.8)))
    fp.append(n('fp_circle',n('center',-29.21,-9.4),n('end',-28.96,-9.4),
                n('stroke',n('width',.12),n('type',S('default'))),n('fill',S('none')),n('layer','F.SilkS')))
    for row,y in [(TEENSY_TOP,-7.62),(TEENSY_BOTTOM,7.62)]:
        for i,pin in enumerate(row):
            x=round(-29.21+i*2.54,4)
            fp.append(n('pad',pin,S('thru_hole'),S('rect' if pin=='VIN' else 'oval'),
                        n('at',x,y),n('size',1.7,1.7),n('drill',1.0),n('layers','*.Cu','*.Mask')))
    path=directory/'Teensy41_Socket_GPIO_ID.kicad_mod'
    path.write_text(pretty(fp)+'\n')
    return path
