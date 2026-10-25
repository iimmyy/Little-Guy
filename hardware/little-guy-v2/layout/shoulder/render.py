"""Render the four named shoulder samples, using exactly the screening geometry."""
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import study

FACES = ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))


def draw_box(ax, part, color, alpha=.5, wire=False):
    vv = study.vertices(part)
    faces = [[vv[i] for i in face] for face in FACES]
    poly = Poly3DCollection(faces,facecolors=[(0,0,0,0)] if wire else color,
                           edgecolors=color, linewidths=.6, alpha=None if wire else alpha)
    ax.add_collection3d(poly)


def main():
    report = json.loads((study.ROOT/"results.json").read_text())
    fig = plt.figure(figsize=(13,11),facecolor="#faf9f6")
    fixed = study.obstacles()
    for i,(key,label) in enumerate((("rest","Rest"),("side_90","Side lift 90°"),
                                   ("forward_90","Forward lift 90°"),("combined_45","Both joints 45°")),1):
        result=report["named_poses"][key]
        moving,child_point,child_axis=study.pose(result["side_lift_deg"],result["forward_lift_deg"])
        ax=fig.add_subplot(2,2,i,projection="3d",computed_zorder=False)
        ax.set_facecolor("#faf9f6")
        for p in fixed:
            if p["name"] == "torso outer reference":
                draw_box(ax,p,"#9ba0a5",wire=True)
            elif p["name"] == "future Jetson bay":
                draw_box(ax,p,"#36a58d",.1)
            elif p["name"] == "right proximal shoulder":
                draw_box(ax,p,"#717880",.7)
        for p in moving:
            draw_box(ax,p,"#4c7ab1" if "reference" not in p["name"] else "#c29954",.65)
        for point,axis,color in ((study.PARENT_AXIS_POINT,np.array([0,1,0]),"#c34b48"),
                                 (child_point,child_axis,"#815da5")):
            ends=np.array([point-axis*35,point+axis*35])
            ax.plot(*ends.T,color=color,lw=2.2)
        ax.set_xlim(-20,365);ax.set_ylim(-75,335);ax.set_zlim(-110,225)
        ax.set_box_aspect((385,410,335));ax.view_init(elev=19,azim=-60)
        ax.set_xlabel("X mm",fontsize=8);ax.set_ylabel("Y mm",fontsize=8);ax.set_zlabel("Z mm",fontsize=8)
        ax.tick_params(labelsize=7)
        torques=result["gravity_torque"]
        ax.set_title(label+f"\nStatic torque: {abs(torques['parent_roll_signed_kgf_cm']):.2f} / {abs(torques['child_pitch_signed_kgf_cm']):.2f} kgf·cm",fontsize=11,pad=6)
    fig.suptitle("Little Guy V2 • right shoulder motion study",fontsize=21,x=.055,ha="left")
    fig.text(.055,.925,"Axis positions from the servo reference; arm masses assumed. Elbow and wrist held straight.",fontsize=11)
    fig.legend(handles=[Line2D([0],[0],color=c,lw=3,label=l) for c,l in
                         (("#c34b48","Parent: side lift"),("#815da5","Child: forward lift"),
                          ("#4c7ab1","Carried servos"),("#c29954","Arm structure"),("#36a58d","Future bay"))],
               loc="lower center",bbox_to_anchor=(.5,.045),ncol=5,frameon=False,fontsize=9)
    fig.text(.055,.025,"Torso outline intersects moving parts. No brackets, cable motion, strength or hardware limits validated.",fontsize=10,color="#984528")
    plt.subplots_adjust(top=.875,bottom=.105,hspace=.14,wspace=.05)
    fig.savefig(study.ROOT/"shoulder-poses.png",dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)



if __name__ == "__main__":
    main()
