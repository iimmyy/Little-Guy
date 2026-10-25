#!/usr/bin/env python3
"""Little Guy V2 neck B: nod AND tilt, from two servos that share both (Immy, 2026-10-25: "I want him to be able to turn
his head, tilt it side to side, up and down. That gives him a lot more expressions"; picked B over a stack of three).

Runs after neck_v2.py, which made the turn (yaw servo upright in the top of the chest) and the nod pin in the base of
the head. This step:
  - turns the nod pin into a ball joint: the head now nods (pitch) AND tilts (roll) about the same point, under its
    balance point, like the joint between the skull and the top vertebra
  - drives both from two ST3215 servos lying side by side in the base of the head, each pushing a short rod down to a
    cross-arm on top of the neck post (a differential, like Reachy's Orbita or Engineered Arts' neck):
        both rods the same way      = nod      left servo = nod + tilt
        rods opposite ways           = tilt     right servo = nod - tilt
    so a nod or a fall onto the face is held by both servos together, each taking half
  - the turn stays where it is (servo in the chest)

In the model the two rods are two tendons on the nod and tilt joints with equal weights (rod offset = horn length, so
one degree at a servo = one degree at the head); the real rod geometry is not exactly linear and gets checked when the
parts are drawn.

usage: neck_b.py <body scene.xml from neck_v2 (or later)> <out scene.xml>
"""
import sys

import mujoco
import numpy as np

from neck_v2 import PIVOT, SERVO, DARK

TILT = 35                                   # tilt range, degrees each way (people: ~45; enough for a curious tilt)
SIDE = 0.0185                               # servos side by side: centres this far left and right of the middle
ROD_X, ROD_DROP = 0.018, 0.011              # rods in front of the pin, and how far below it they meet the post


def main():
    src, out = sys.argv[1], sys.argv[2]
    spec = mujoco.MjSpec.from_file(src)
    old = mujoco.MjModel.from_xml_path(src)
    old_q = {mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, j):
             old.key_qpos[0][old.jnt_qposadr[j]:old.jnt_qposadr[j] + (7 if old.jnt_type[j] == 0 else 1)]
             for j in range(old.njnt)}
    for k in list(spec.keys):
        spec.delete(k)
    head, link = spec.body("head"), spec.body("neck_yaw_link")
    nod = head.joints[0]
    roll = head.add_joint(name="neck_roll", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], pos=PIVOT,
                          range=np.radians([-TILT, TILT]), limited=1)
    roll.armature, roll.damping, roll.frictionloss = nod.armature, nod.damping, nod.frictionloss
    for a in list(spec.actuators):                     # the single nod servo's drive goes; two shared ones come in
        if a.target == nod.name:
            spec.delete(a)
    box = mujoco.mjtGeom.mjGEOM_BOX
    for g in head.geoms:                               # the nod servo box moves left; its twin goes right
        if g.type == box and np.allclose(g.size, [0.0226, 0.0175, 0.0124]):
            g.pos = PIVOT + [0, SIDE, 0.012]
    add = lambda body, **kw: body.add_geom(contype=0, conaffinity=0, density=0, **kw)
    add(head, type=box, size=[0.0226, 0.0175, 0.0124], pos=PIVOT + [0, -SIDE, 0.012], rgba=SERVO)
    hp = np.array(head.pos)                            # head frame -> post frame (the head sits unrotated on the post)
    add(link, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.0055, 0, 0], pos=hp + PIVOT, rgba=DARK)        # the ball
    add(link, type=box, size=[0.004, SIDE + 0.004, 0.0025], pos=hp + PIVOT + [ROD_X / 2, 0, -ROD_DROP],
        rgba=DARK)                                                                                       # cross-arm
    for s in (1, -1):
        top, bot = PIVOT + [ROD_X, s * SIDE, 0.001], PIVOT + [ROD_X, s * SIDE, -ROD_DROP]
        add(head, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*top, *bot], size=[0.0018, 0, 0], rgba=DARK)  # rod
    for side, s in (("left", 1), ("right", -1)):
        t = spec.add_tendon(name=f"neck_servo_{side}")
        t.wrap_joint(nod.name, 1.0)
        t.wrap_joint("neck_roll", float(s))
        a = spec.add_actuator(name=f"neck_servo_{side}", target=t.name, trntype=mujoco.mjtTrn.mjTRN_TENDON)
        a.ctrlrange, a.ctrllimited = [-10, 10], 1
    m = spec.compile()
    q = m.qpos0.copy()
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in old_q:
            n = 7 if m.jnt_type[jid] == 0 else 1
            q[m.jnt_qposadr[jid]:m.jnt_qposadr[jid] + n] = old_q[name]
    spec.add_key(name="home", qpos=q.tolist())
    m = spec.compile()
    open(out, "w").write(spec.to_xml())
    print(f"{out}: neck B (ball joint, nod {np.degrees(m.jnt_range[m.joint(nod.name).id]).round()} deg, tilt +-{TILT} "
          f"deg, two servos share both), head {m.body_subtreemass[m.body('head').id] * 1000:.0f} g, "
          f"mass {m.body_subtreemass[0]:.2f} kg")


if __name__ == "__main__":
    main()
