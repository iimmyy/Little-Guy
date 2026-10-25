#!/usr/bin/env python3
"""Little Guy V2 body with the new legs bolted on (Immy, 2026-10-25: "I want to see him all bolted on before we move on").

Takes a body scene from v2_body.py, removes ToddlerBot's two legs (and everything that pointed at them: their motors,
the hip-yaw gear couplings, their floor contacts, the old standing pose) and puts two legs from leg_v2.py in their
place. Each new leg hangs straight above where ToddlerBot's foot stood, its hip pitch axis at ToddlerBot's height. The new standing
pose keeps the old arm, neck and waist angles, with straight legs, feet flat on the floor.

usage: v2_newlegs.py <body scene.xml> <out scene.xml>
"""
import re, sys

import mujoco
import numpy as np

import leg_v2

HIP_DROP = 0.06                                    # new leg: root (hip yaw) sits this far above the hip pitch axis


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

    old_d = mujoco.MjData(old_m)
    mujoco.mj_resetDataKeyframe(old_m, old_d, 0)
    mujoco.mj_forward(old_m, old_d)
    feet_y = {s: old_d.xpos[mujoco.mj_name2id(old_m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")][1] -
              old_d.xpos[mujoco.mj_name2id(old_m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")][1] for s in ("left", "right")}

    spec = mujoco.MjSpec.from_file(src)
    mounts, gone_joints, gone_geoms, gone_sites = {}, set(), set(), set()
    for side in ("left", "right"):
        hip = spec.body(f"{side}_hip_pitch_link")
        mounts[side] = np.array(hip.pos) + [0, 0, HIP_DROP]
        mounts[side][1] = feet_y[side]                    # straight legs: hips right above where ToddlerBot's feet stand
        for b in subtree_bodies(hip):
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
        spec.delete(spec.body(f"{side}_hip_pitch_link"))
    xml = spec.to_xml()

    bodies, eqs, acts, pairs = "", "", "", ""
    for side in ("left", "right"):
        body, eq, act, feet = leg_v2.leg(side, " ".join(f"{v:.6f}" for v in mounts[side]), actuator="motor")
        bodies, eqs, acts = bodies + body, eqs + eq, acts + act
        pairs += "".join(f'<pair geom1="floor" geom2="{g}"/>' for g in feet)
    xml = re.sub(r'(<body name="pelvis_link"[^>]*>)', lambda mt: mt.group(1) + bodies, xml, count=1)
    for tag, extra in (("equality", eqs), ("actuator", acts), ("contact", pairs)):
        if f"<{tag}>" in xml:
            xml = xml.replace(f"</{tag}>", extra + f"</{tag}>", 1)
        else:
            xml = xml.replace("</mujoco>", f"<{tag}>{extra}</{tag}></mujoco>")

    spec2 = mujoco.MjSpec.from_string(xml)
    m = spec2.compile()
    d = mujoco.MjData(m)
    q = m.qpos0.copy()
    for j in range(m.njnt):                        # old arm, neck, waist angles; new legs straight
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in old_key and name not in gone_joints:   # the new legs reuse ToddlerBot's joint names
            q[m.jnt_qposadr[j]] = old_key[name]
    q[0:3], q[3:7] = [0, 0, 1.0], [1, 0, 0, 0]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    soles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_sole") for s in ("left", "right")]
    bottom = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in soles)
    q[2] = 1.0 - bottom
    spec2.add_key(name="home", qpos=q.tolist())
    m = spec2.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    top = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "i1_i1")][2] + 0.042
    open(out, "w").write(spec2.to_xml())
    print(f"{out}: height {top * 100:.1f} cm, mass {m.body_subtreemass[0]:.2f} kg (was {old_m.body_subtreemass[0]:.2f}), "
          f"{m.nu} motors, {len(spec2.equalities)} linkage/coupling constraints")


if __name__ == "__main__":
    main()
