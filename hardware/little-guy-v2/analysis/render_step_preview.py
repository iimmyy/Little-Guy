"""Render the disposable mesh produced by inspect_servo_step.py --mesh-json."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

parser=argparse.ArgumentParser()
parser.add_argument("mesh",type=Path)
args=parser.parse_args()
parts=json.loads(args.mesh.read_text())
triangles=[]; colors=[]
for part in parts:
    i=part["solid_index"]
    color="#c28a36" if i in (6,7) else "#687581" if i in (0,1,2) else "#6e93a5"
    triangles.extend(part["triangles"])
    colors.extend([color]*len(part["triangles"]))
fig=plt.figure(figsize=(12,7),facecolor="#faf9f6")
for j,az in enumerate((-55,125),1):
    ax=fig.add_subplot(1,2,j,projection="3d",computed_zorder=False)
    ax.set_facecolor("#faf9f6")
    poly=Poly3DCollection(triangles,facecolors=colors,linewidth=0,
                         shade=True,lightsource=LightSource(azdeg=315,altdeg=45),zorder=2)
    ax.add_collection3d(poly)
    ax.plot([-25.5,-25.5],[-35,18],[0,0],color="#e6523d",linewidth=1.3,linestyle="--",zorder=3)
    ax.set_xlim(-40,15); ax.set_ylim(-35,18); ax.set_zlim(-15,15)
    ax.set_box_aspect((55,53,30));ax.view_init(elev=24,azim=az)
    ax.set_xlabel("X mm");ax.set_ylabel("Y mm");ax.set_zlabel("Z mm")
    ax.set_title(f"Source model • view {j}",fontsize=12)
fig.suptitle("Supplied servo STEP • geometry inspection",fontsize=19,x=.05,ha="left")
fig.text(.05,.055,"Red overlay: common output axis • Gold: round fittings • Original source geometry retained",fontsize=10)
fig.text(.05,.02,"Source identifies itself as SCS215. One of eight solids fails the CAD validity check; this is a visualization, not a mount release.",fontsize=9,color="#914b29")
plt.subplots_adjust(top=.84,bottom=.16,wspace=.08)
destination=Path(__file__).resolve().parents[1]/"reference/servo-step-review.png"
fig.savefig(destination,dpi=160,facecolor=fig.get_facecolor())
print(f"Rendered {len(triangles)} triangles to {destination}")
