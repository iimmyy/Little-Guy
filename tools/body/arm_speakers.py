#!/usr/bin/env python3
"""Stereo speaker pods in Little Guy's upper arms (Immy, 2026-10-25: "in this like biceps, kind of like the tricep
area ... a little concave ... cutout with like a USB-C port, and then I can put a little speaker in there. USB-C wouldn't
be for charging though, it would mainly be for connecting audio").

His sound is a 2.1 system: these two pods carry the stereo (mids and highs, 40 mm full-range drivers), one woofer in
his lower back carries the bass (torso_v2.py). Each pod sits on the outer side of the upper arm, where the triceps'
outer head is: it faces sideways, so the two pods are as far apart as his body allows, nothing blocks them when the arms
hang, and the elbow can still fold all the way (a pod on the front of the arm would stop the forearm).

The upper arm is full of two servos (shoulder yaw, elbow) with only ~6.5 mm to spare on the outer side, so the arm grows
a round housing there, like a muscle: the pod snaps into a cutout in it and plugs into a USB-C socket at the bottom
of the cutout. The socket only carries the speaker signal from the amp in his chest (no charging, no Bluetooth: both
sides stay in step), with a guard so a phone charger plugged in by mistake can't hurt anything.

usage: arm_speakers.py <body scene.xml> <out scene.xml>
"""
import sys

import mujoco
import numpy as np

import parts
import torso_options as T

PUCK_R, FACE = 0.024, 0.0405    # housing radius; pod face, out from the arm's centre line (arm shell radius is 0.024)
BASE = 0.010                    # housing starts this far out from the arm's centre line (inside the arm shell)
POD_R, GAP_R = 0.0205, 0.0225   # pod face (40 mm driver behind a grille) and the cutout around it
GRILLE = [0.2, 0.21, 0.24, 1]
UP = 0.010                      # pod sits this far above the arm shell's middle, toward the shoulder: clear of the
                                # forearm when the elbow folds (arm sweep check)


def frame(R_body, out_world):
    """axes in the upper arm's own frame: along the arm, straight out to the side, and forward"""
    along = R_body.T @ np.array([0, 0, 1.0])                     # arm hangs down: world up, in the arm's frame
    out = R_body.T @ out_world
    out -= along * (out @ along)
    out /= np.linalg.norm(out)
    fwd = np.cross(out, along)
    return along, out, fwd


def quat_from(x, y, z):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).flatten())
    return q


def main():
    src, out = sys.argv[1], sys.argv[2]
    spec = mujoco.MjSpec.from_file(src)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for side, s in (("left", 1), ("right", -1)):
        name = f"{side}_upper_arm"
        b = m.body(name).id
        shell = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b
                 and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)][0]
        along, outd, fwd = frame(d.xmat[b].reshape(3, 3), np.array([0, s, 0.0]))
        centre = m.geom_pos[shell] + along * UP                  # on the arm shell's centre line, in the arm's frame
        body = spec.body(name)
        add = lambda **kw: body.add_geom(density=0, contype=0, conaffinity=0, **kw)
        cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
        q = quat_from(fwd, np.cross(outd, fwd), outd)            # a cylinder's axis = straight out
        mid = (BASE + FACE) / 2
        add(type=cyl, size=[PUCK_R, (FACE - BASE) / 2, 0], pos=centre + outd * mid, quat=q, rgba=T.SHELL)
        add(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[PUCK_R + 0.002, 0.016, PUCK_R + 0.002],     # soft shoulder
            pos=centre + outd * (BASE + 0.012), quat=quat_from(fwd, outd, along), rgba=T.SHELL)  # into the arm
        add(type=cyl, size=[GAP_R, 0.0003, 0], pos=centre + outd * (FACE + 0.0002), quat=q, rgba=T.DARK)   # cutout
        add(type=cyl, size=[POD_R, 0.0005, 0], pos=centre + outd * (FACE + 0.0006), quat=q, rgba=GRILLE)   # pod
        add(type=cyl, size=[0.006, 0.0007, 0], pos=centre + outd * (FACE + 0.0009), quat=q, rgba=T.ACCENT)  # dust cap
        kg = parts.PARTS[f"speaker pod, 40 mm full-range driver, {side}"][0] / 1000
        w = body.add_body(name=f"{side}_pod_weight", pos=centre + outd * (FACE - 0.01))
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.008, 0, 0], mass=kg, group=4, contype=0, conaffinity=0)
    m = spec.compile()
    open(out, "w").write(spec.to_xml())
    print(f"{out}: arm speaker pods on, mass {m.body_subtreemass[0]:.2f} kg, pod face {FACE * 1000:.1f} mm out from "
          f"the arm's centre line ({(FACE - 0.024) * 1000:.1f} mm past the old arm)")


if __name__ == "__main__":
    main()
