#!/usr/bin/env python3
"""Rerun the standing tests on one or more V2 bodies, same code for each, so a change is compared like for like
(after the chest insides + 2 cm deeper chest, 2026-10-25). The shove and power parts are the scripts that made
ear-bumper-1025/shoves_hard_floor.txt and power-1025.txt, unchanged; falls run fall_test.py itself.

  balance   weight, and the balance point over the feet (x forward, y left, height), mm
  shoves    biggest 0.2 s push at the chest he stays standing after: side, back, front (2 N steps)
  power     servo watts standing still, standing + talking with his hands, and idle (datasheet torque constants)

usage: retest.py <out folder> <name=scene.xml> ...     prints and writes retest.txt
"""
import collections, os, sys

import mujoco
import numpy as np

from live import Servos
import servos as S

KGCM = 0.0981
# datasheet: supply current = idle + torque / Kt (Kt in kg.cm per A); volts
SPEC = {"STS3250": (0.024, 11, 12), "ST3215": (0.030, 11, 12), "STS3032": (0.006, 3.5, 6), "SCS0009": (0.006, 2.3, 6)}
SPOOL = 0.008


def balance(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    feet = (d.xpos[m.body("left_foot").id] + d.xpos[m.body("right_foot").id]) / 2
    return m.body_subtreemass[0], (d.subtree_com[0] - feet) * 1000


def stays_up(m, f):
    d = mujoco.MjData(m)
    s = Servos(m, d)
    t = m.body("torso").id
    while d.time < 4:
        s.apply(d)
        d.xfrc_applied[t, :3] = f if 1.0 <= d.time < 1.2 else 0
        mujoco.mj_step(m, d)
    return d.xpos[t][2] > 0.3


def shoves(m):
    res = []
    for way, v in (("side", [0, 1, 0]), ("back", [-1, 0, 0]), ("front", [1, 0, 0])):
        best = 0
        for F in range(4, 41, 2):
            if stays_up(m, np.array(v) * F):
                best = F
            else:
                break
        res.append(f"{way} {best} N")
    return ", ".join(res)


def power(m):
    kind = []
    for a in range(m.nu):
        n = m.actuator(a).name
        kind.append("SCS0009" if n.endswith("_pull") else ("ST3215" if n.startswith("neck_servo") else S.servo_for(n)))

    def watts(gesture, T=6.0):
        d = mujoco.MjData(m)
        s = Servos(m, d)
        E, n = np.zeros(m.nu), 0
        joints = list(m.actuator_trnid[s.ja, 0])
        k = joints.index(m.joint("left_shoulder_pitch").id)
        k2 = joints.index(m.joint("right_shoulder_roll").id)
        g0 = s.goal.copy()
        while d.time < T:
            if gesture and d.time > 1:      # talking-with-hands: both arms moving, ~0.5 Hz
                s.goal[k] = g0[k] + 0.6 * np.sin(2 * np.pi * 0.5 * d.time)
                s.goal[k2] = g0[k2] + 0.4 * np.sin(2 * np.pi * 0.7 * d.time)
            s.apply(d)
            mujoco.mj_step(m, d)
            if d.time > 1:
                E += np.abs(d.actuator_force) * np.array([SPOOL if kind[a] == "SCS0009" and m.actuator(a).name.endswith("_pull")
                                                          else 1 for a in range(m.nu)])
                n += 1
        tau = E / n
        by = collections.defaultdict(float)
        for a in range(m.nu):
            idle, kt, v = SPEC[kind[a]]
            by[kind[a]] += v * (idle + tau[a] / (kt * KGCM))
        return sum(by.values())

    idle = sum(SPEC[k][2] * SPEC[k][0] for k in kind)
    return f"standing still {watts(False):.1f} W, talking with his hands {watts(True):.1f} W, idle {idle:.1f} W"


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    lines = []
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        m = mujoco.MjModel.from_xml_path(path)
        kg, c = balance(m)
        for line in (f"{name}: {path}", f"  weight {kg:.3f} kg, balance point over the feet {np.round(c, 1)} mm",
                     f"  biggest 0.2 s shove he stays up after: {shoves(m)}", f"  servo power: {power(m)}"):
            print(line, flush=True)
            lines.append(line)
    open(os.path.join(out, "retest.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
