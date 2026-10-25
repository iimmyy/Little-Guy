"""Candidate package geometry, before board layout or assembly release."""
from schematic import n,S,pretty,effects


def xmos_qf60b(directory):
    """XVF3800 datasheet v3.2.1 Fig8.1, mirrored from bottom to PCB top view.

    Toe/heel allowance, paste apertures, via-in-pad and assembly tolerances
    require fabricator review. This is not an XMOS reference-CAD import.
    """
    directory.mkdir(parents=True,exist_ok=True)
    name='XMOS_QF60B_7x7mm_P0.4mm_5Paddle'
    fp=n('footprint',name,n('version',20260206),n('generator','pcbnew'),n('layer','F.Cu'),
         n('descr','Candidate XVF3800 QF60B; XMOS datasheet v3.2.1 Fig8.1. Four core paddles 61-64, GND65. Top view. Assembly and stencil review pending.'),
         n('tags','XVF3800 QF60B five paddle 7x7 0.4'),
         n('property','Reference','REF**',n('at',0,-4.7),n('layer','F.SilkS'),effects(1)),
         n('property','Value',name,n('at',0,4.7),n('layer','F.Fab'),effects(.75)),
         n('attr',S('smd')))
    for layer,a,b,width in [('F.Fab',(-3.5,-3.5),(3.5,3.5),.1),
                            ('F.CrtYd',(-3.95,-3.95),(3.95,3.95),.05)]:
        fp.append(n('fp_rect',n('start',*a),n('end',*b),n('stroke',n('width',width),n('type',S('default'))),n('fill',S('none')),n('layer',layer)))
    fp.append(n('fp_circle',n('center',-3.8,-3.8),n('end',-3.6,-3.8),n('stroke',n('width',.12),n('type',S('default'))),n('fill',S('none')),n('layer','F.SilkS')))
    for i in range(15):
        v=round(-2.8+.4*i,4)
        for pin,x,y,sx,sy in [(1+i,-3.35,v,.6,.2),(16+i,v,3.35,.2,.6),
                              (31+i,3.35,-v,.6,.2),(46+i,-v,-3.35,.2,.6)]:
            fp.append(n('pad',str(pin),S('smd'),S('rect'),n('at',x,y),n('size',sx,sy),n('layers','F.Cu','F.Paste','F.Mask')))
    # Four core paddles: D3=3.60, E3=0.60, gap S=0.40.
    for pin,x,y,sx,sy in [(61,-2.5,0,.6,3.6),(62,0,2.5,3.6,.6),
                          (63,2.5,0,.6,3.6),(64,0,-2.5,3.6,.6)]:
        fp.append(n('pad',str(pin),S('smd'),S('rect'),n('at',x,y),n('size',sx,sy),n('layers','F.Cu','F.Mask')))
        for offset in [-1.15,0,1.15]:
            px=x if sx<sy else offset;py=offset if sx<sy else y
            fp.append(n('pad','',S('smd'),S('rect'),n('at',px,py),n('size',.42 if sx<sy else .95,.95 if sx<sy else .42),n('layers','F.Paste')))
    # Ground paddle D2/E2=3.60, pin-1 corner chamfer C0.350 x45deg.
    pts=[(-1.45,-1.8),(1.8,-1.8),(1.8,1.8),(-1.8,1.8),(-1.8,-1.45)]
    fp.append(n('pad','65',S('smd'),S('custom'),n('at',0,0),n('size',.2,.2),
                n('layers','F.Cu','F.Mask'),n('options',n('clearance',S('outline')),n('anchor',S('rect'))),
                n('primitives',n('gr_poly',n('pts',*[n('xy',x,y) for x,y in pts]),n('width',0),n('fill',S('yes'))))))
    for x in [-.85,.85]:
        for y in [-.85,.85]:
            fp.append(n('pad','',S('smd'),S('rect'),n('at',x,y),n('size',1.45,1.45),n('layers','F.Paste')))
    fp.append(n('fp_text',S('user'),'61-64 CORE 0V9 / 65 GND',n('at',0,5.7),n('layer','Dwgs.User'),effects(.7)))
    path=directory/(name+'.kicad_mod');path.write_text(pretty(fp)+'\n')
    return path
