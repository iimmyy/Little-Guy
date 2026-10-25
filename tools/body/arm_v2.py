#!/usr/bin/env python3
"""Little Guy V2 arm + hand, sketch 1: muscles up high, tendons down (same rule as leg_v2.py).

Arm, 7 joints like ToddlerBot's: shoulder pitch / roll / twist, elbow, forearm twist, wrist flex / wrist side-tilt.
  - shoulder pitch servo in the torso, shoulder roll in the shoulder cap
  - upper arm: shoulder twist + elbow servos lying along the bone (ST3215, 55 g)
  - forearm: a small STS3032 (20.6 g) twists everything below it; below that, the forearm "muscles": seven Feetech
    SCS0009 micro servos (13.2 g, 23.2 x 12.1 x 25.25 mm) in three rows, tapering toward the wrist
  - wrist: two of those servos push the hand through two rods, exactly like the ankle (same way = flex, opposite = tilt)

Hand, 5 fingers, Immy's pull-pack design (2026-10-25): one wire per pull, each wound on a small spool (8 mm radius) on
its own servo, fingers sprung open:
  1 index   2 middle   3 ring + pinky (one wire splitting over a balance pulley, like the shared muscle)
  4 thumb curl   5 thumb swing across the palm

usage: arm_v2.py <out folder>   writes arm_v2_left/right.xml (one arm on a fixed shoulder) and tests fingers + wrist
"""
import math, os, sys

import mujoco
import numpy as np

UPPER, FORE_TOP, FORE_LOW = 0.105, 0.04, 0.077    # shoulder roll -> elbow, elbow -> forearm twist, twist -> wrist
ST = (0.0247 / 2, 0.035 / 2, 0.0452 / 2, 0.055)    # ST3215 half sizes (x, y-shaft, z-long) + kg
STS32 = (0.012 / 2, 0.0275 / 2, 0.032 / 2, 0.0206)  # STS3032
SCS = (0.02525 / 2, 0.0121 / 2, 0.0232 / 2, 0.0132)  # SCS0009, shaft along x here
SPOOL_N = 0.226 / 0.008                             # SCS0009 stall 2.3 kg.cm on an 8 mm spool = 28 N of pull
SHELL, DARK, SERVO, ROD, WIRE = "0.9 0.91 0.93 1", "0.16 0.17 0.2 1", "0.25 0.45 0.8", "0.75 0.75 0.78 1", "0.95 0.6 0.2 1"
ARMATURE = {"big": 'armature="0.005" damping="0.02" frictionloss="0.01"',
            "small": 'armature="0.001" damping="0.005" frictionloss="0.002"'}
FINGERS = {"index": (0.0128, (0.015, 0.009, 0.007)), "middle": (0.0043, (0.016, 0.010, 0.008)),  # x on the palm, bones
           "ring": (-0.0043, (0.015, 0.0095, 0.0075)), "pinky": (-0.0128, (0.012, 0.0075, 0.0055))}
FR, PALM_Z, MOMENT = 0.0035, 0.032, 0.004          # finger radius, wrist -> knuckles, tendon distance from the joints


def box(name, half, pos, alpha=0.0):
    return (f'<geom name="{name}" type="box" size="{half[0]} {half[1]} {half[2]}" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'mass="{half[3]}" rgba="{SERVO} {alpha}"/>')


def finger(p, s, name, x, bones):
    """three bones hinged at the knuckles, a wire running down the palm side, springs holding it open"""
    xml, sites, z = "", [], 0.0
    for i, length in enumerate(bones):
        xml += (f'<body name="{p}{name}{i}" pos="{x if i == 0 else 0} 0 {-PALM_Z if i == 0 else -bones[i - 1]}">'
                f'<joint name="{p}{name}_j{i}" type="hinge" axis="{-s} 0 0" range="-0.1 1.6" stiffness="0.02" damping="0.0005" frictionloss="0"/>'
                f'<geom type="capsule" fromto="0 0 -0.001 0 0 {-length + 0.001}" size="{FR}" rgba="{SHELL}" mass="0.002"/>'
                f'<site name="{p}{name}_s{i}a" pos="0 {-s * MOMENT} -0.002" size="0.0008"/>'
                f'<site name="{p}{name}_s{i}b" pos="0 {-s * MOMENT} {-length + 0.001}" size="0.0008"/>')
        sites += [f"{p}{name}_s{i}a", f"{p}{name}_s{i}b"]
    return xml + "</body>" * len(bones), sites


def thumb(p, s):
    """thumb on the front edge of the palm: swing (across the palm), then two bones that curl"""
    xml = (f'<body name="{p}thumb_base" pos="0.019 {-s * 0.003} -0.006" euler="0 -0.6 0">'
           f'<joint name="{p}thumb_swing" type="hinge" axis="0 0 {-s}" range="-0.1 1.4" stiffness="0.03" damping="0.0005" frictionloss="0"/>'
           f'<geom type="capsule" fromto="0 0 0 0 0 -0.016" size="{FR + 0.001}" rgba="{SHELL}" mass="0.003"/>'
           f'<site name="{p}thumb_swing_end" pos="0 {-s * (MOMENT + 0.002)} -0.008" size="0.0008"/>')
    sites = []
    for i, length in enumerate((0.012, 0.010)):
        xml += (f'<body name="{p}thumb{i}" pos="0 0 {-0.016 if i == 0 else -0.012}">'
                f'<joint name="{p}thumb_j{i}" type="hinge" axis="0 1 0" range="-0.1 1.4" stiffness="0.02" damping="0.0005" frictionloss="0"/>'
                f'<geom type="capsule" fromto="0 0 -0.001 0 0 {-length + 0.001}" size="{FR}" rgba="{SHELL}" mass="0.002"/>'
                f'<site name="{p}thumb_s{i}a" pos="{-MOMENT} 0 -0.002" size="0.0008"/>'
                f'<site name="{p}thumb_s{i}b" pos="{-MOMENT} 0 {-length + 0.001}" size="0.0008"/>')
        sites += [f"{p}thumb_s{i}a", f"{p}thumb_s{i}b"]
    return xml + "</body></body></body>", sites


def rod(p, tag, pivot, arm, end_pt, wrist_z):
    """servo arm turning about x at pivot (forearm frame), rod down to end_pt (forearm frame, hand at rest)"""
    tip = pivot + arm
    v = end_pt - tip
    length = float(np.linalg.norm(v))
    zx = v / length
    xml = (f'<body name="{p}wcrank_{tag}" pos="{pivot[0]:.5f} {pivot[1]:.5f} {pivot[2]:.5f}">'
           f'<joint name="{p}wrist_{tag}" {ARMATURE["small"]} type="hinge" axis="1 0 0" range="-1.2 1.2"/>'
           f'<geom type="capsule" fromto="0 0 0 {arm[0]} {arm[1]} {arm[2]}" size="0.0025" mass="0.002" rgba="{DARK}"/>'
           f'<body name="{p}wrod_{tag}" pos="{arm[0]} {arm[1]} {arm[2]}" zaxis="{zx[0]:.6f} {zx[1]:.6f} {zx[2]:.6f}">'
           f'<joint name="{p}wrod_{tag}_1" type="hinge" axis="1 0 0" damping="0.0005" frictionloss="0"/>'
           f'<joint name="{p}wrod_{tag}_2" type="hinge" axis="0 1 0" damping="0.0005" frictionloss="0"/>'
           f'<geom type="capsule" fromto="0 0 0 0 0 {length:.6f}" size="0.0016" mass="0.002" rgba="{ROD}"/>'
           f'<site name="{p}wrod_{tag}_end" pos="0 0 {length:.6f}"/></body></body>')
    hand_site = (f'<site name="{p}whand_{tag}" pos="{end_pt[0]:.5f} {end_pt[1]:.5f} {end_pt[2] - wrist_z:.5f}" '
                 f'size="0.0015"/>')
    eq = f'<connect site1="{p}wrod_{tag}_end" site2="{p}whand_{tag}" solref="0.004 1" solimp="0.9999 0.9999 0.001 0.5 2"/>'
    return xml, hand_site, eq


def arm(side, pos="0 0 0", actuator="position"):
    """one arm, root = a fixed mount holding the shoulder pitch servo. Returns (body, equality, tendon, actuator xml)."""
    p, s = f"{side}_", 1 if side == "left" else -1
    rows = [-0.014, -0.039, -0.064]                       # forearm servo rows below the twist joint
    slots = [-SCS[1] * 2, 0.0, SCS[1] * 2]
    pulls = {"index": (0, 0), "middle": (0, 1), "ringpinky": (0, 2), "thumb_curl": (1, 0), "thumb_swing": (1, 1)}
    servos = "".join(box(f"{p}{n}_servo", SCS, (0, s * slots[c], rows[r])) for n, (r, c) in pulls.items())
    servos += box(f"{p}wrist_b_servo", SCS, (0, s * slots[2], rows[1])) + box(f"{p}wrist_a_servo", SCS, (0, s * slots[1] * 0.0 + s * 0.006, rows[2]))
    spool = "".join(f'<site name="{p}{n}_spool" pos="0 {s * slots[c] - 0.007:.4f} {rows[r]}" size="0.002"/>'
                    for n, (r, c) in pulls.items())
    wz = -FORE_LOW
    rod_a, hs_a, eq_a = rod(p, "a", np.array([SCS[0] + 0.002, s * 0.006, rows[2]]), np.array([0, s * 0.009, 0]),
                            np.array([0.012, s * 0.011, wz - 0.006]), wz)
    rod_b, hs_b, eq_b = rod(p, "b", np.array([-SCS[0] - 0.002, s * slots[2], rows[1]]), np.array([0, s * 0.009, 0]),
                            np.array([-0.012, s * 0.011, wz - 0.006]), wz)
    fingers, fsites = "", {}
    for name, (x, bones) in FINGERS.items():
        xml, sites = finger(p, s, name, x, bones)
        fingers += xml
        fsites[name] = sites
    th_xml, th_sites = thumb(p, s)
    body = f"""<body name="{p}shoulder_mount" pos="{pos}">
      {box(f"{p}shoulder_pitch_servo", (ST[2], ST[1], ST[0], ST[3]), (0, -s * 0.012, 0))}
      <body name="{p}shoulder_pitch_link">
        <joint name="{p}shoulder_pitch" {ARMATURE["big"]} type="hinge" axis="0 1 0" range="-3.1 3.1"/>
        <geom type="ellipsoid" size="0.03 0.03 0.03" pos="0 {s * 0.022} -0.004" rgba="{SHELL}" mass="0.015"/>
        {box(f"{p}shoulder_roll_servo", (ST[2], ST[0], ST[1], ST[3]), (0, s * 0.022, 0))}
        <body name="{p}upper_arm" pos="0 {s * 0.03} -0.012">
          <joint name="{p}shoulder_roll" {ARMATURE["big"]} type="hinge" axis="{s} 0 0" range="-0.4 1.6"/>
          <geom type="capsule" fromto="0 0 -0.012 0 0 {-UPPER + 0.014}" size="0.024" rgba="{SHELL}" mass="0.03"/>
          {box(f"{p}shoulder_yaw_servo", ST, (0, 0, -0.012 - ST[2]))}
          <body name="{p}upper_arm_lo" pos="0 0 -0.04">
            <joint name="{p}shoulder_yaw" {ARMATURE["big"]} type="hinge" axis="0 0 1" range="-1.6 1.6"/>
            {box(f"{p}elbow_servo", ST, (0, 0, -UPPER + 0.04 + ST[2] - 0.002))}
            <body name="{p}forearm" pos="0 0 {-UPPER + 0.04}">
              <joint name="{p}elbow" {ARMATURE["big"]} type="hinge" axis="0 -1 0" range="-0.1 2.4"/>
              <geom type="cylinder" fromto="0 -0.022 0 0 0.022 0" size="0.017" rgba="{DARK}" mass="0.008"/>
              <geom type="capsule" fromto="0 0 -0.012 0 0 {-FORE_TOP}" size="0.021" rgba="{SHELL}" mass="0.015"/>
              {box(f"{p}forearm_twist_servo", (STS32[0], STS32[1], STS32[2], STS32[3]), (0, 0, -0.006 - STS32[2]))}
              <body name="{p}forearm_lo" pos="0 0 {-FORE_TOP}">
                <joint name="{p}forearm_twist" {ARMATURE["small"]} type="hinge" axis="0 0 1" range="-1.6 1.6"/>
                <geom type="ellipsoid" size="0.02 0.026 0.05" pos="0 0 -0.03" rgba="{SHELL}" mass="0.015"/>
                {servos}
                {spool}
                <site name="{p}wrist_pass" pos="0 {-s * 0.002} {wz + 0.004}" size="0.0015"/>
                {rod_a}
                {rod_b}
                <body name="{p}wrist_cross" pos="0 0 {wz}">
                  <joint name="{p}wrist_flex" type="hinge" axis="{s} 0 0" range="-1.0 1.0" damping="0.002" frictionloss="0"/>
                  <geom type="box" size="0.006 0.006 0.004" rgba="{DARK}" mass="0.003"/>
                  <body name="{p}hand">
                    <joint name="{p}wrist_tilt" type="hinge" axis="0 1 0" range="-0.5 0.5" damping="0.002" frictionloss="0"/>
                    <geom name="{p}palm" type="box" size="0.018 0.006 0.0145" pos="0 0 -0.0175" rgba="{SHELL}" mass="0.012"/>
                    <geom type="capsule" fromto="-0.0155 0 -0.03 0.0155 0 -0.03" size="0.0065" rgba="{SHELL}" mass="0.002"/>
                    <site name="{p}palm_in" pos="0 {-s * MOMENT} -0.006" size="0.0012"/>
                    <site name="{p}palm_split" pos="-0.0085 {-s * MOMENT} -0.027" size="0.0012"/>
                    <site name="{p}palm_thumb" pos="0.004 {-s * (MOMENT + 0.002)} -0.009" size="0.0012"/>
                    {hs_a}
                    {hs_b}
                    {fingers}
                    {th_xml}
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""
    wire = lambda name, sites: (f'<spatial name="{p}{name}_wire" width="0.0006" rgba="{WIRE}">'
                                + "".join(f'<site site="{x}"/>' for x in sites) + "</spatial>")
    base = [f"{p}{{}}_spool", f"{p}wrist_pass", f"{p}palm_in"]
    tendons = wire("index", [b.format("index") for b in base] + fsites["index"])
    tendons += wire("middle", [b.format("middle") for b in base] + fsites["middle"])
    split = [b.format("ringpinky") for b in base] + [f"{p}palm_split"]
    tendons += (f'<spatial name="{p}ringpinky_wire" width="0.0006" rgba="{WIRE}">'
                + "".join(f'<site site="{x}"/>' for x in split)
                + '<pulley divisor="2"/>' + "".join(f'<site site="{x}"/>' for x in [f"{p}palm_split"] + fsites["ring"])
                + '<pulley divisor="2"/>' + "".join(f'<site site="{x}"/>' for x in [f"{p}palm_split"] + fsites["pinky"])
                + "</spatial>")
    tendons += wire("thumb_curl", [b.format("thumb_curl") for b in base] + th_sites)
    tendons += wire("thumb_swing", [f"{p}thumb_swing_spool", f"{p}wrist_pass", f"{p}palm_thumb", f"{p}thumb_swing_end"])
    gain = 'kp="20" kv="0.3"' if actuator == "position" else 'ctrlrange="-10 10"'
    acts = "".join(f'<{actuator} name="{p}{j}" joint="{p}{j}" {gain}/>'
                   for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "forearm_twist", "wrist_a", "wrist_b"))
    acts += "".join(f'<motor name="{p}{n}_pull" tendon="{p}{n}_wire" ctrlrange="{-SPOOL_N:.1f} 0"/>' for n in pulls)
    return body, eq_a + eq_b, tendons, acts


def arm_xml(side="left"):
    body, eq, tendons, acts = arm(side, "0 0 0.45")
    return f"""<mujoco model="little guy v2 arm sketch 1">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <visual><global offwidth="900" offheight="1100"/><quality shadowsize="4096"/></visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="0.6 -0.8 0.9" dir="-0.5 0.8 -0.5" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.5 0.5 0.9" dir="0.5 -0.5 -0.5" diffuse="0.35 0.35 0.35"/>
    <geom type="plane" size="1 1 0.01" rgba="0.2 0.3 0.4 1"/>
    {body}
  </worldbody>
  <equality>{eq}</equality>
  <tendon>{tendons}</tendon>
  <actuator>{acts}</actuator>
</mujoco>"""


def test(m, side="left"):
    """arm held still; pull each wire in turn, then the two wrist servos; read the joints"""
    d = mujoco.MjData(m)
    q = lambda n: math.degrees(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]])
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    watch = ["index_j0", "index_j1", "middle_j0", "ring_j0", "ring_j1", "pinky_j0", "pinky_j1", "thumb_swing",
             "thumb_j0", "thumb_j1", "wrist_flex", "wrist_tilt"]
    cases = [("rest", {})] + [(f"pull {n} 4 N", {f"{side}_{n}_pull": -4.0}) for n in
                              ("index", "middle", "ringpinky", "thumb_curl", "thumb_swing")]
    cases += [("wrist servos same +0.4", {f"{side}_wrist_a": 0.4, f"{side}_wrist_b": 0.4}),
              ("wrist servos opposite", {f"{side}_wrist_a": 0.4, f"{side}_wrist_b": -0.4})]
    rows = []
    for label, ctrl in cases:
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = 0
        for n, v in ctrl.items():
            d.ctrl[aid(n)] = v
        for _ in range(2000):
            mujoco.mj_step(m, d)
        bad = int(sum(w.number for w in d.warning))
        rows.append((label, bad, [round(q(f"{side}_{w}")) for w in watch]))
    return watch, rows


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for side in ("left", "right"):
        path = os.path.join(out, f"arm_v2_{side}.xml")
        open(path, "w").write(arm_xml(side))
        m = mujoco.MjModel.from_xml_path(path)
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_mount")
        print(f"\n{side} arm + hand mass {m.body_subtreemass[root]:.3f} kg, {m.nu} actuators, {m.ntendon} wires")
        watch, rows = test(m, side)
        print(f"{'case':26s} {'unst':>4s} " + " ".join(f"{w[:9]:>9s}" for w in watch))
        for label, bad, vals in rows:
            print(f"{label:26s} {bad:4d} " + " ".join(f"{v:9d}" for v in vals))


if __name__ == "__main__":
    main()
