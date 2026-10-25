"""Render original reference parts. Source attribution stays with the image."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("mesh_json",type=Path)
args=parser.parse_args()
parts=json.loads(args.mesh_json.read_text())
fig=plt.figure(figsize=(14,6),facecolor="#faf9f6")
for i,part in enumerate(parts,1):
    ax=fig.add_subplot(1,len(parts),i,projection="3d")
    triangles=np.array(part["triangles"]);points=triangles.reshape(-1,3)
    lower,upper=points.min(0),points.max(0)
    ax.add_collection3d(Poly3DCollection(triangles,facecolors="#8aadc4",linewidths=0,
                                       shade=True,lightsource=LightSource(315,45)))
    for j,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):
        setter(lower[j]-2,upper[j]+2)
    ax.set_box_aspect(upper-lower+4);ax.view_init(elev=27,azim=-55)
    ax.set_title(part["name"].replace("_SO101", "").replace("_", " "),fontsize=12)
    ax.set_xlabel("X mm");ax.set_ylabel("Y mm");ax.set_zlabel("Z mm")
fig.suptitle("SO-101 original parts • reuse inspection",fontsize=20,x=.04,ha="left")
fig.text(.04,.035,"TheRobotStudio / SO-ARM100 • Apache-2.0 • Unmodified reference geometry, not a Little Guy mount release",fontsize=10)
fig.subplots_adjust(top=.86,bottom=.15,wspace=.2)
fig.savefig(Path(__file__).resolve().parents[1]/"reference/so101/parts-review.png",dpi=150)
