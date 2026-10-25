#!/usr/bin/env python3
"""Little Guy V2 body with the new arms and five-finger hands bolted on (after the new legs, v2_newlegs.py).

Removes ToddlerBot's two arms and everything that pointed at them (motors, gear couplings, floor contacts, sensors),
puts two arms from arm_v2.py on the same shoulder spots, and writes a new standing pose: arms hanging a little away
from the body, elbows slightly bent, hands open; legs, neck and waist as before.

usage: v2_newarms.py <body scene with new legs.xml> <out scene.xml>
"""
import re, sys

import mujoco
import numpy as np

import arm_v2

POSE = {"shoulder_roll": 0.12, "elbow": 0.3}       # radians; everything else at zero


def subtree_bodies(body):
    out = [body]
    for child in body.bodies:
        out += subtree_bodies(child)
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    old_m = mujoco.MjModel.from_xml_path(src)
    old_key = {mujoco.mj_id2name(old_m, mujoco.mjtObj.mjOBJ_JOINT, j): old_m.key_qpos[0][old_m.jnt_qposadr[j]]
               for j in range(old_m.njnt) if old_m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}

    spec = mujoco.MjSpec.from_file(src)
    mounts, gone_joints, gone_geoms, gone_sites = {}, set(), set(), set()
    for side in ("left", "right"):
        top = spec.body(f"{side}_shoulder_pitch_link")
        mounts[side] = np.array(top.pos)
        for b in subtree_bodies(top):
            gone_joints |= {j.name for j in b.joints}
            gone_geoms |= {g.name for g in b.geoms if g.name}
            gone_sites |= {s.name for s in b.sites if s.name}
    for a in list(spec.actuators):
        if a.target in gone_joints:
            spec.delete(a)
    for e in list(spec.equalities):
        if e.name1 in gone_joints or e.name2 in gone_joints or e.name1 in gone_sites or e.name2 in gone_sites:
            spec.delete(e)
    for p in list(spec.pairs):
        if p.geomname1 in gone_geoms or p.geomname2 in gone_geoms:
            spec.delete(p)
    for s in list(spec.sensors):
        if s.objname in gone_sites or s.objname in gone_joints:
            spec.delete(s)
    for k in list(spec.keys):
        spec.delete(k)
    for side in ("left", "right"):
        spec.delete(spec.body(f"{side}_shoulder_pitch_link"))
    xml = spec.to_xml()

    bodies, eqs, tendons, acts = "", "", "", ""
    for side in ("left", "right"):
        body, eq, ten, act = arm_v2.arm(side, " ".join(f"{v:.6f}" for v in mounts[side]), actuator="motor")
        bodies, eqs, tendons, acts = bodies + body, eqs + eq, tendons + ten, acts + act
    xml = re.sub(r'(<body name="torso"[^>]*>)', lambda mt: mt.group(1) + bodies, xml, count=1)
    for tag, extra in (("equality", eqs), ("tendon", tendons), ("actuator", acts)):
        if f"<{tag}>" in xml:
            xml = xml.replace(f"</{tag}>", extra + f"</{tag}>", 1)
        else:
            xml = xml.replace("</mujoco>", f"<{tag}>{extra}</{tag}></mujoco>")

    spec2 = mujoco.MjSpec.from_string(xml)
    m = spec2.compile()
    q = m.qpos0.copy()
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in old_key and name not in gone_joints:
            q[m.jnt_qposadr[j]] = old_key[name]
        for side in ("left", "right"):
            for joint, v in POSE.items():
                if name == f"{side}_{joint}":
                    q[m.jnt_qposadr[j]] = v
    q[0:7] = old_m.key_qpos[0][0:7]
    spec2.add_key(name="home", qpos=q.tolist())
    m = spec2.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    top = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "i1_i1")][2] + 0.042
    open(out, "w").write(spec2.to_xml())
    print(f"{out}: height {top * 100:.1f} cm, mass {m.body_subtreemass[0]:.2f} kg (was {old_m.body_subtreemass[0]:.2f}), "
          f"{m.nu} actuators ({sum(1 for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON)} "
          f"finger pulls), {m.ntendon} tendons, {len(spec2.equalities)} constraints")


if __name__ == "__main__":
    main()
