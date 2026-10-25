"""Dimensioned target-location drawing; not a PCB layout or board-edge definition."""
from pathlib import Path
import argparse,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[1]
g=json.loads((root/'head/microphone-geometry.json').read_text())
revision=json.loads((root/'head/circuit.json').read_text())['revision'].removesuffix('-WIP')
fig,(plan,side)=plt.subplots(1,2,figsize=(12,6.3),gridspec_kw={'width_ratios':[1.25,1]})
blue='#24618b';gray='#6a727b'
fig.suptitle('Little Guy — Head-A microphone targets',fontsize=17,x=.07,ha='left')
fig.text(.07,.91,revision+' • frozen head targets • CAD dimensions, physically untested',color=gray,fontsize=10)

points={m['location']:m['array_plane_mm'][:2] for m in g['microphones']}
loop=[points[n] for n in ['front','right','back','left','front']]
plan.plot([p[0] for p in loop],[p[1] for p in loop],color='#aeb5bc',ls='--',lw=1)
for m in g['microphones']:
    x,y,_=m['array_plane_mm']
    plan.scatter([x],[y],s=55,c=blue,zorder=4)
    dx,dy,ha={'front':(0,4,'center'),'back':(0,-10,'center'),'left':(3,-8,'left'),'right':(-3,-8,'right')}[m['location']]
    plan.text(x+dx,y+dy,m['reference']+' / '+m['location']+'\nMIC'+str(m['logical_index']),ha=ha,va='bottom',fontsize=9)
plan.scatter([0],[0],s=20,c=gray,marker='+')
left,right=points['left'][0],points['right'][0]
plan.annotate('',xy=(right,-47),xytext=(left,-47),arrowprops={'arrowstyle':'<->','color':gray,'lw':.8})
for x in [left,right]:plan.plot([x,x],[-2,-49],lw=.6,c='#aeb5bc')
plan.text(0,-45,f"{g['left_right_board_plane_mm']:.3f} mm",ha='center',fontsize=9,color=gray)
front,back=points['front'][1],points['back'][1]
plan.annotate('',xy=(64,front),xytext=(64,back),arrowprops={'arrowstyle':'<->','color':gray,'lw':.8})
for y in [front,back]:plan.plot([2,66],[y,y],lw=.6,c='#aeb5bc')
plan.text(66,0,f"{g['front_back_board_plane_mm']:.3f} mm",rotation=90,va='center',fontsize=9,color=gray)
plan.set(xlim=(-68,75),ylim=(-56,49),xlabel='Array X (mm) — positive toward robot right',ylabel='Array Y (mm) — positive toward robot front')
plan.set_aspect('equal');plan.set_title('Acoustic openings in the board plane',loc='left',fontsize=11,pad=13)
plan.grid(alpha=.12);plan.spines[['top','right']].set_visible(False)

theta=math.radians(g['tilt_front_up_deg']);height=g['plane_origin_head_mm'][2]
side.plot([-36,36],[height-36*math.tan(theta),height+36*math.tan(theta)],c=blue,lw=1.6,label='Rigid acoustic plane')
for m in g['microphones']:
    x,y,z=m['target_head_mm']
    side.scatter([-y],[z],c='#333333',s=28,zorder=4)
source={m['location']:m['target_head_mm'] for m in g['microphones']}
front_z,back_z=source['front'][2],source['back'][2]
side.text(-source['back'][1],back_z-2,'back',ha='center',fontsize=9)
side.text(-source['front'][1],front_z+1.5,'front',ha='center',fontsize=9)
side.annotate('',xy=(43,front_z),xytext=(43,back_z),arrowprops={'arrowstyle':'<->','color':gray,'lw':.8})
side.text(45,(front_z+back_z)/2,f"{g['front_above_back_mm']:.3f} mm",va='center',rotation=90,color=gray,fontsize=9)
side.set(xlim=(-41,55),ylim=(24,51),xlabel='Head fore/aft position (mm) — front to the right',ylabel='Head-frame height (mm)')
side.set_aspect('equal');side.set_title('Side view at the head',loc='left',fontsize=11,pad=15)
side.spines[['top','right']].set_visible(False);side.grid(alpha=.12)
side.legend(loc='upper left',frameon=False,fontsize=8)
fig.text(.59,.28,f"Front-up tilt: {g['tilt_front_up_deg']:.3f}°\nLargest marker-to-plane distance: {g['maximum_normal_residual_mm']:.3f} mm\n{g['front_back_head_horizontal_mm']:.3f} mm in plan view\n= {g['front_back_board_plane_mm']:.3f} mm along the plane.",fontsize=9,va='top',linespacing=1.5)
fig.text(.07,.065,'The dashed diamond joins microphone targets. It is not the PCB edge.',fontsize=10,color='#333333')
fig.text(.07,.035,'Package offsets, gaskets, supports, board height and final outline remain design work.',fontsize=9,color=gray)
fig.subplots_adjust(left=.07,right=.96,top=.83,bottom=.21,wspace=.34)
for suffix in ['png','pdf','svg']:
    fig.savefig(args.output_dir/('head-microphone-geometry-'+revision+'.'+suffix),dpi=180,facecolor='white')
print('Head acoustic-target drawing exported as PNG, PDF and SVG; no PCB layout created.')
