#!/usr/bin/env python3
"""Little Guy V2, first body sketch: ToddlerBot's skeleton at our size, with our servos' weight and our head.

What changes from ToddlerBot's open model (toddlerbot_2xc):
  size    every length grows by one factor so he stands 60 cm to the top of the head (Immy's height)
  servos  each joint's motor weight becomes ours: STS3250 74.5 g in the legs, ST3215 69 g everywhere else. ToddlerBot's
          motors (ROBOTIS spec sheets): XC330 23 g, XC430 65 g, XM430 82 g, 2XC430 102 g and 2XL430 98.2 g (two joints
          each). Where ours is heavier, the extra sits at the joint as a small fixed weight; where ours is lighter
          (knee, ankle pitch: 7.5 g each) nothing is taken off, so the sketch errs heavy.
  frame   the rest (printed parts, Jetson, battery, wiring) grows with volume, size^3: heavier than a printed shell would,
          again erring heavy
  head    ToddlerBot's head is hidden and the I1 head (tools/body/heads.py) sits on the same neck, facing forward;
          head weight set to 160 g (estimate: 3.5" screen ~45 g, shell ~80 g, two cameras, mic board, screws)
  armor   optional extra weight (kg) spread over the body in proportion to each part's frame weight

Joint layout, joint limits, motor friction and the keyframe pose stay ToddlerBot's.

usage: v2_body.py <out folder> [armor kg ...]   writes v2_<armor>.xml scenes (load them with servo_check.py / watch.py)
"""
import math, os, sys

import mujoco
import numpy as np

import heads

REF = os.path.expanduser("~/.littleguy/ref/toddlerbot/toddlerbot/descriptions/toddlerbot_2xc")
HEIGHT = 0.60
HEAD_KG = 0.160
HEAD_W, HEAD_H, HEAD_D = 0.068, 0.042, 0.07       # I1 half-width, half-height, depth (heads.slab defaults)

TB_MOTOR_G = {"XC330": 23.0, "XC430": 65.0, "XM430": 82.0, "2XC430": 51.0, "2XL430": 49.1}   # per joint
OUR_SERVO_G = {"STS3250": 74.5, "ST3215": 69.0}


def tb_motor(joint):
    """ToddlerBot's motor at this joint (toddlerbot/descriptions/default.yml)"""
    if joint.startswith(("neck", "waist")) or "hip_yaw" in joint:
        return "XC330"
    if "hip_pitch" in joint or "hip_roll" in joint:
        return "2XC430"
    if "knee" in joint or "ankle_pitch" in joint:
        return "XM430"
    if "ankle_roll" in joint or "shoulder_pitch" in joint:
        return "XC430"
    return "2XL430"


def our_servo(joint):
    return "STS3250" if any(k in joint for k in ("hip", "knee", "ankle")) else "ST3215"


def head_spec(out):
    """the I1 head as its own little model, every part weightless (the head's weight is set on the neck body)"""
    heads.face_png(os.path.join(out, "wide.png"), "^_^", wide=2)
    meshes, geoms = heads.slab(0, "I1", 3.0, 0.7)
    geoms = geoms[:-1]                                   # drop the neck bridge; ToddlerBot's neck is there
    xml = f"""<mujoco><default><geom density="0" contype="0" conaffinity="0" group="2"/></default>
  <asset><texture name="wide" type="2d" file="{os.path.abspath(os.path.join(out, 'wide.png'))}"/>
    <material name="wide" texture="wide" emission="0.6"/>{meshes}</asset>
  <worldbody><body name="i1">{chr(10).join(geoms)}</body></worldbody></mujoco>"""
    return mujoco.MjSpec.from_string(xml)


def build(out, s, armor_kg=0.0):
    spec = mujoco.MjSpec.from_file(os.path.join(REF, "scene.xml"))
    spec.meshdir = os.path.join(REF, "assets")
    names = [a.target for a in spec.actuators]
    tb_motors = sum(TB_MOTOR_G[tb_motor(j)] for j in names) / 1000
    bodies = [b for b in spec.bodies if b.name != "world"]
    tb_total = sum(b.mass for b in bodies)
    k = (s ** 3 * (tb_total - tb_motors) + tb_motors) / tb_total     # frame grows with volume, motors don't
    frame_kg = {b.name: b.mass * k for b in bodies}
    total_frame = sum(frame_kg.values())
    for b in bodies:
        km = k * (1 + armor_kg / total_frame)
        b.pos = b.pos * s
        b.ipos = b.ipos * s
        b.mass = b.mass * km
        b.fullinertia = b.fullinertia * km * s * s
        for j in b.joints:
            j.pos = j.pos * s
        for site in b.sites:
            site.pos = site.pos * s
            site.size = site.size * s
        for g in b.geoms:
            g.pos = g.pos * s
            if g.type != mujoco.mjtGeom.mjGEOM_MESH:
                g.size = g.size * s
    for mesh in spec.meshes:
        mesh.scale = mesh.scale * s
    for cam in spec.cameras:
        cam.pos = cam.pos * s
    key = spec.keys[0]
    key.qpos = [q * s if i < 3 else q for i, q in enumerate(key.qpos)]
    added = 0.0
    for b in bodies:                                     # our servos' extra weight, at each joint
        for j in b.joints:
            if j.name not in names:
                continue
            extra = (OUR_SERVO_G[our_servo(j.name)] - TB_MOTOR_G[tb_motor(j.name)]) / 1000
            if extra > 0:
                w = b.add_body(name=f"servo_{j.name}", pos=j.pos)   # weight on a small hidden ball: the saved file
                w.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=extra,   # drops a bare body mass
                           contype=0, conaffinity=0, group=4)
                added += extra
    head = spec.body("head")
    ctr = np.array([-0.0105 * s, 0.0, 0.0164 * s + HEAD_H])   # I1 sits where ToddlerBot's head box sits, bottom on the neck
    head.ipos, head.mass = ctr, HEAD_KG
    head.fullinertia = [HEAD_KG / 12 * (a * a + b * b) for a, b in
                        (((2 * HEAD_W), (2 * HEAD_H)), (HEAD_D, 2 * HEAD_H), (HEAD_D, 2 * HEAD_W))] + [0, 0, 0]
    for g in head.geoms:
        if g.name == "head_collision":                   # ToddlerBot's falls-onto-the-head box, resized to I1
            g.pos, g.size = ctr, [HEAD_D / 2, HEAD_W, HEAD_H]
        else:
            g.rgba = [0, 0, 0, 0]
    frame = head.add_frame(pos=ctr, quat=[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])  # heads.py faces -y, he faces +x
    frame.attach_body(head_spec(out).body("i1"), "i1_", "")
    return spec, added


def measure(spec):
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    top = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "i1_i1")][2] + HEAD_H
    return m, top


def main():
    out = sys.argv[1]
    armors = [float(a) for a in sys.argv[2:]] or [0.0]
    os.makedirs(out, exist_ok=True)
    s = 1.0
    for _ in range(4):                                   # find the size factor that makes him 60 cm tall
        spec, _ = build(out, s)
        _, top = measure(spec)
        s *= HEIGHT / top
    for armor in armors:
        spec, added = build(out, s, armor)
        m, top = measure(spec)
        path = os.path.join(out, f"v2_armor{armor:g}.xml")
        open(path, "w").write(spec.to_xml())
        print(f"{path}: size x{s:.3f}, height {top * 100:.1f} cm, mass {m.body_subtreemass[0]:.2f} kg "
              f"(servo extra {added * 1000:.0f} g, armor {armor:g} kg, head {HEAD_KG * 1000:.0f} g)")


if __name__ == "__main__":
    main()
