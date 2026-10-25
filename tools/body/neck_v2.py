#!/usr/bin/env python3
"""Little Guy V2 neck (Immy, 2026-10-25: "his neck needs way better support, dude. One fall and that thing is over").

ToddlerBot nods the head through a small plate and two thin rods and turns it through two small gears; a fall onto the
head goes straight through those. This replaces them with:
  - a short thick neck post (3.4 cm across), turned by an ST3215 standing upright in the top of the chest
  - the nod pivot moved up into the base of the head, under the head's balance point, so the head barely loads it
  - a fork on top of the post holding the pivot pin from both sides (a fall loads the pin and frame, not the gears);
    the nod servo (ST3215) rides in the head with its shaft on the pin
  - a rubber bumper ring where the neck meets the chest
  - a soft rubber (TPU) rim around the screen, 3 mm proud of the glass: the I1 screen used to be the frontmost part

Head weight comes from parts.py; the rim follows head A's outline (head_a.py runs first).

usage: neck_v2.py <body scene.xml> <out scene.xml>
"""
import sys

import mujoco
import numpy as np

import head_a
import parts

SHELL, DARK, RUBBER = [0.9, 0.91, 0.93, 1], [0.16, 0.17, 0.2, 1], [0.1, 0.1, 0.11, 1]
SERVO = [0.25, 0.45, 0.8, 0]                       # hidden inside (alpha 0), weight counted
ST_KG, XC330_KG = 0.055, 0.023                     # our ST3215 vs ToddlerBot's neck motor (already in its torso weight)
PIVOT = np.array([-0.0116, 0.0, 0.028])            # nod pin, in the head's frame: under the head's balance point
RIM_FRONT, RIM_DEPTH = 0.0325, 0.011                # glass face at x 0.0294 in the head frame, shell front at 0.0234
RIM_W, RIM_H, FACE_Z = (head_a.WIN_W + head_a.W) / 2, (head_a.WIN_H + head_a.H) / 2, 0.06   # between face window and shell edge
RIM_N, RIM_HALF, RIM_PIECES = head_a.N, 0.004, 64


def subtree(body):
    out = [body]
    for c in body.bodies:
        out += subtree(c)
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    spec = mujoco.MjSpec.from_file(src)
    gone_joints, gone_sites = set(), set()
    for name in ("neck_yaw_gear_drive", "neck_pitch_plate"):
        for b in subtree(spec.body(name)):
            gone_joints |= {j.name for j in b.joints}
            gone_sites |= {s.name for s in b.sites}
    gone_sites |= {s.name for s in spec.body("neck_yaw_link").sites}
    for e in list(spec.equalities):
        if {e.name1, e.name2} & (gone_joints | gone_sites):
            spec.delete(e)
    for a in list(spec.actuators):
        if a.target in gone_joints:
            spec.delete(a)
    for k in list(spec.keys):                      # the pose loses the deleted joints; rebuilt below
        spec.delete(k)
    old = mujoco.MjModel.from_xml_path(src)
    old_q = {mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, j): old.key_qpos[0][old.jnt_qposadr[j]:old.jnt_qposadr[j] + (7 if old.jnt_type[j] == 0 else 1)]
             for j in range(old.njnt)}
    for name in ("neck_yaw_gear_drive", "neck_pitch_plate"):
        spec.delete(spec.body(name))

    torso, link, head = spec.body("torso"), spec.body("neck_yaw_link"), spec.body("head")
    head.pos = [head.pos[0], 0, head.pos[2]]           # centred on the neck (ToddlerBot's head sat 4.4 mm to the left;
                                                       # with his chest weight 4.6 mm right he stood leaning 1.8 deg)
    add = lambda body, **kw: body.add_geom(contype=0, conaffinity=0, density=0, **kw)
    cyl, box = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX
    base = np.array(link.pos)
    # turn servo upright in the top of the chest (its extra weight over ToddlerBot's motor on a hidden ball)
    add(torso, type=box, size=[0.0124, 0.0226, 0.0175], pos=base - [0, 0, 0.0175], rgba=SERVO)
    w = torso.add_body(name="neck_yaw_servo_weight", pos=base - [0, 0, 0.0175])
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.008, 0, 0], mass=ST_KG - XC330_KG, contype=0, conaffinity=0, group=4)
    add(torso, type=cyl, size=[0.024, 0.003, 0], pos=base + [0, 0, 0.001], rgba=RUBBER)          # bumper ring
    for g in link.geoms:
        g.rgba = [0, 0, 0, 0]                                                                    # ToddlerBot's neck part
    top = np.array(head.pos) + PIVOT                                                             # pin, in the link frame
    add(link, type=cyl, size=[0.017, (top[2] - 0.012) / 2, 0], pos=[top[0] / 2, 0, (top[2] - 0.012) / 2], rgba=SHELL)
    for y in (-0.016, 0.016):                                                                    # the fork
        add(link, type=box, size=[0.008, 0.003, 0.009], pos=[top[0], y, top[2] - 0.004], rgba=DARK)
    add(link, type=cyl, size=[0.0025, 0.021, 0], pos=top, quat=[np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0], rgba=DARK)
    j = head.joints[0]
    j.pos = PIVOT
    j.armature, j.damping, j.frictionloss = 0.005, [0.02, 0, 0], 0.01                       # range stays ToddlerBot's
    add(head, type=box, size=[0.0226, 0.0175, 0.0124], pos=PIVOT + [0, 0, 0.012], rgba=SERVO)   # nod servo, shaft on the pin
    head.mass = parts.grams("head") / 1000                                                   # every head part, nod servo included
    dims = ((2 * head_a.W, 2 * head_a.H), (head_a.DEPTH, 2 * head_a.H), (head_a.DEPTH, 2 * head_a.W))
    head.inertia, head.iquat = [head.mass / 12 * (a * a + b * b) for a, b in dims], [1, 0, 0, 0]
    # soft rubber rim around the screen, standing 3 mm proud of the glass, so a face-plant lands on the rim: it fills
    # the band between the glass edge and the shell edge, following the head's rounded-square outline (32 pieces)
    t = np.linspace(0, 2 * np.pi, RIM_PIECES + 1)
    yz = np.stack([RIM_W * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / RIM_N),
                   RIM_H * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / RIM_N)], 1)
    for k in range(RIM_PIECES):
        (y0, z0), (y1, z1) = yz[k], yz[k + 1]
        a = np.arctan2(z1 - z0, y1 - y0)
        name = f"rim_{k}"
        head.add_geom(name=name, type=box, size=[RIM_DEPTH / 2, np.hypot(y1 - y0, z1 - z0) / 2 + 0.0008, RIM_HALF],
                      pos=[RIM_FRONT - RIM_DEPTH / 2, (y0 + y1) / 2, FACE_Z + (z0 + z1) / 2],
                      quat=[np.cos(a / 2), np.sin(a / 2), 0, 0], rgba=RUBBER, density=0, contype=0, conaffinity=0)
        spec.add_pair(geomname1="floor", geomname2=name)
    for name, joint in (("neck_yaw", "neck_yaw_driven"), ("neck_pitch", "neck_pitch")):
        a = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
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
    print(f"{out}: mass {m.body_subtreemass[0]:.2f} kg, {m.nu} actuators, {len(spec.equalities)} constraints")


if __name__ == "__main__":
    main()
