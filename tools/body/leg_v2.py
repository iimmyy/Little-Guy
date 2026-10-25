#!/usr/bin/env python3
"""Little Guy V2 leg, sketch 1: muscles up high, tendons down (Immy agreed 2026-10-25).

ToddlerBot's ankle motors sit at the foot as big blocks. Here every servo lies along the limb instead, the way muscles
sit along a bone: the hip pitch and knee servos inside the thigh, both ankle servos up in the calf. The ankle servos
move the foot through two thin rods (a parallel linkage ankle, like the Asimov legs and Unitree G1), the way the calf
moves the foot through the Achilles tendon. Turning both servos the same way tips the toes up or down (pitch, the two
share the load); turning them opposite ways tilts the foot sideways (roll).

Servo boxes are the real Feetech size (45.2 x 24.7 x 35 mm, shaft along the 35 mm side) and weight (STS3250 74.5 g).
Lengths follow ToddlerBot x1.109 (thigh 113 mm, shin 122 mm). Covers are simple shapes for now; one left leg hangs from
a fixed hip so it can be checked on its own.

usage: leg_v2.py <out folder>   writes leg_v2.xml, runs the rod test, prints how the foot moves
"""
import math, os, sys

import mujoco
import numpy as np

SX, SY, SZ = 0.0247 / 2, 0.035 / 2, 0.0452 / 2     # servo half sizes: front-back, side (shaft), along the limb
SERVO_KG = 0.0745
THIGH, SHIN, ANKLE_DROP = 0.113, 0.122, 0.03       # hip pitch -> knee, knee -> ankle, ankle -> sole
SOLE = THIGH + SHIN + ANKLE_DROP                   # hip pitch height above the floor
CRANK, HEEL = 0.016, 0.025                         # servo arm length; rod attachment behind the ankle
SHELL, DARK, SERVO, ROD = "0.9 0.91 0.93 1", "0.16 0.17 0.2 1", "0.25 0.45 0.8 1", "0.75 0.75 0.78 1"


SERVO_JOINT = 'armature="0.005" damping="0.02" frictionloss="0.01"'   # gearbox inertia (ToddlerBot measured 0.002-0.008)


def servo(name, pos, along="z", alpha=1.0):
    """a servo box, long side along the limb; its shaft points sideways (y)"""
    size = f"{SX} {SY} {SZ}" if along == "z" else f"{SZ} {SY} {SX}"
    rgba = SERVO.rsplit(" ", 1)[0] + f" {alpha}"
    return f'<geom name="{name}" type="box" size="{size}" pos="{pos}" mass="{SERVO_KG}" rgba="{rgba}"/>'


def rod_chain(p, tag, shaft_z, y, ankle_z):
    """servo arm on the shin + rod down to the foot; returns (shin bodies, foot attachment point, equality)"""
    tip = np.array([-CRANK, y, shaft_z])
    foot_pt = np.array([-HEEL, y, ankle_z])       # behind the ankle, level with it
    v = foot_pt - tip
    length = float(np.linalg.norm(v))
    zx = v / length
    xml = f"""<body name="{p}crank_{tag}" pos="0 {y} {shaft_z}">
          <joint name="{p}ankle_{tag}" {SERVO_JOINT} type="hinge" axis="0 1 0" range="-1.2 1.2"/>
          <geom type="capsule" fromto="0 0 0 {-CRANK} 0 0" size="0.004" mass="0.005" rgba="{DARK}"/>
          <body name="{p}rod_{tag}" pos="{-CRANK} 0 0" zaxis="{zx[0]:.6f} {zx[1]:.6f} {zx[2]:.6f}">
            <joint name="{p}rod_{tag}_swing1" type="hinge" axis="1 0 0" damping="0.001" frictionloss="0"/>
            <joint name="{p}rod_{tag}_swing2" type="hinge" axis="0 1 0" damping="0.001" frictionloss="0"/>
            <geom type="capsule" fromto="0 0 0 0 0 {length:.6f}" size="0.0025" mass="0.004" rgba="{ROD}"/>
            <site name="{p}rod_{tag}_end" pos="0 0 {length:.6f}"/>
          </body>
        </body>"""
    eq = f'<connect site1="{p}rod_{tag}_end" site2="{p}foot_{tag}" solref="0.004 1" solimp="0.9999 0.9999 0.001 0.5 2"/>'
    return xml, foot_pt, eq


def leg(side, pos="0 0 0", actuator="position"):
    """one leg, root at the hip yaw servo; side "left" or "right" (mirror images). Returns (body xml, equality xml,
    actuator xml, foot geom names). The hip pitch axis sits 6 cm below the root."""
    p, s = f"{side}_", 1 if side == "left" else -1
    knee_z, ankle_z = -THIGH, -THIGH - SHIN
    a_shaft, b_shaft = knee_z - 0.037, knee_z - 0.084     # two calf servos stacked, like G1's
    rod_a, pa, eq_a = rod_chain(p, "a", a_shaft - knee_z, s * (SY + 0.005), ankle_z - knee_z)    # a on the outer side
    rod_b, pb, eq_b = rod_chain(p, "b", b_shaft - knee_z, -s * (SY + 0.005), ankle_z - knee_z)
    az = ankle_z - knee_z
    foot_site = lambda tag, q: f'<site name="{p}foot_{tag}" pos="{q[0]} {q[1]} {q[2] - az}" size="0.003"/>'
    feet = [f"{p}sole", f"{p}toe", f"{p}heel"]
    gain = 'kp="20" kv="0.3"' if actuator == "position" else 'ctrlrange="-10 10"'
    joints = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_a", "ankle_b"]
    acts = "".join(f'<{actuator} name="{p}{j}" joint="{p}{j}" {gain}/>' for j in joints)
    body = f"""<body name="{p}hip_yaw_link" pos="{pos}">
        <joint name="{p}hip_yaw" {SERVO_JOINT} type="hinge" axis="0 0 1" range="-1 1"/>
        {servo(f"{p}hip_yaw_servo", "0 0 0", along="x", alpha=0.0)}
        <geom type="ellipsoid" size="0.034 0.03 0.042" pos="0 0 -0.03" rgba="{SHELL}" mass="0.02"/>
        <body name="{p}hip_roll_link" pos="0 0 -0.035">
          <joint name="{p}hip_roll" {SERVO_JOINT} type="hinge" axis="1 0 0" range="-0.6 0.6"/>
          {servo(f"{p}hip_roll_servo", "0 0 -0.008", along="x", alpha=0.0)}
          <body name="{p}thigh" pos="0 0 -0.025">
            <joint name="{p}hip_pitch" {SERVO_JOINT} type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <geom type="capsule" fromto="0 0 -0.012 0 0 {-THIGH + 0.014}" size="0.028" rgba="{SHELL}" mass="0.04"/>
            {servo(f"{p}hip_pitch_servo", f"0 0 {-SZ + 0.009}", alpha=0.0)}
            {servo(f"{p}knee_servo", f"0 0 {-THIGH + SZ - 0.009}", alpha=0.0)}
            <body name="{p}shin" pos="0 0 {knee_z}">
              <joint name="{p}knee" {SERVO_JOINT} type="hinge" axis="0 1 0" range="-0.1 2.2"/>
              <geom type="cylinder" fromto="0 -0.024 0 0 0.024 0" size="0.02" rgba="{DARK}" mass="0.01"/>
              <geom type="capsule" fromto="-0.004 0 -0.03 0.002 0 {az + 0.03}" size="0.022" rgba="{SHELL}" mass="0.03"/>
              <geom type="ellipsoid" size="0.024 0.026 0.048" pos="-0.01 0 -0.062" rgba="{SHELL}" mass="0.01"/>
              {servo(f"{p}ankle_a_servo", f"-0.004 0 {a_shaft - knee_z + 0.012}", alpha=0.0)}
              {servo(f"{p}ankle_b_servo", f"-0.004 0 {b_shaft - knee_z + 0.012}", alpha=0.0)}
              {rod_a}
              {rod_b}
              <body name="{p}ankle_cross" pos="0 0 {az}">
                <joint name="{p}ankle_pitch" type="hinge" axis="0 1 0" range="-0.9 0.9" damping="0.01" frictionloss="0"/>
                <geom type="box" size="0.008 0.012 0.008" rgba="{DARK}" mass="0.01"/>
                <body name="{p}foot" pos="0 0 0">
                  <joint name="{p}ankle_roll" type="hinge" axis="1 0 0" range="-0.6 0.6" damping="0.01" frictionloss="0"/>
                  <geom name="{p}sole" type="box" size="0.032 0.03 0.007" pos="0.018 0 {-ANKLE_DROP + 0.007}" rgba="{SHELL}" mass="0.04"/>
                  <geom name="{p}toe" type="cylinder" size="0.03 0.007" pos="0.05 0 {-ANKLE_DROP + 0.007}" rgba="{SHELL}" mass="0.005"/>
                  <geom name="{p}heel" type="cylinder" size="0.03 0.007" pos="-0.014 0 {-ANKLE_DROP + 0.007}" rgba="{SHELL}" mass="0.005"/>
                  <geom type="ellipsoid" size="0.024 0.026 0.012" pos="-0.006 0 {-ANKLE_DROP + 0.016}" rgba="{SHELL}" mass="0.01"/>
                  {foot_site("a", pa)}
                  {foot_site("b", pb)}
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>"""
    return body, eq_a + eq_b, acts, feet


def leg_xml():
    body, eq, acts, _ = leg("left", "0 0 0.06")
    return f"""<mujoco model="little guy v2 leg sketch 1">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <visual><global offwidth="900" offheight="1100"/><quality shadowsize="4096"/></visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="0.6 -0.8 0.6" dir="-0.5 0.8 -0.5" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.5 0.5 0.6" dir="0.5 -0.5 -0.5" diffuse="0.35 0.35 0.35"/>
    <geom type="plane" size="1 1 0.01" rgba="0.2 0.3 0.4 1"/>
    <body name="pelvis" pos="0 0 {SOLE}">
      <geom type="box" size="0.035 0.045 0.015" pos="0 -0.03 0.075" rgba="{SHELL}"/>
      {body}
    </body>
  </worldbody>
  <equality>{eq}</equality>
  <actuator>{acts}</actuator>
</mujoco>"""


def rod_test(m):
    """hold the hip still, turn the two calf servos, read where the foot ends up"""
    d = mujoco.MjData(m)
    jp = lambda n: d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]]
    rows = []
    for a, b in ((0, 0), (0.4, 0.4), (-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4)):
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = [0, 0, 0, 0, a, b]                # hip and knee held straight
        for _ in range(3000):
            mujoco.mj_step(m, d)
        gap = max(float(np.linalg.norm(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"left_rod_{t}_end")] -
                                       d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"left_foot_{t}")])) for t in "ab")
        bad = int(sum(w.number for w in d.warning))
        rows.append((a, b, bad, math.degrees(jp("left_ankle_pitch")), math.degrees(jp("left_ankle_roll")), gap * 1000))
    return rows


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "leg_v2.xml")
    open(path, "w").write(leg_xml())
    m = mujoco.MjModel.from_xml_path(path)
    print(f"leg mass {m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_hip_yaw_link')]:.3f} kg")
    print(f"{'servo a':>8s} {'servo b':>8s} {'unstable':>8s} {'foot pitch':>11s} {'foot roll':>10s} {'rod gap mm':>11s}")
    for a, b, bad, p, r, g in rod_test(m):
        print(f"{a:8.2f} {b:8.2f} {bad:8d} {p:10.1f}° {r:9.1f}° {g:11.2f}")


if __name__ == "__main__":
    main()
