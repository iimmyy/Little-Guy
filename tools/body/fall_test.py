#!/usr/bin/env python3
"""Stand, then a shove forward: he falls on his face with his servos still holding the standing pose (like tripping).

Reports what the neck takes: the biggest force through the nod pin, the biggest twist (torque) the neck has to hold
about the nod axis (servo + end stop + linkage, not the floor's own push), the head's hardest stop (in g, standing
still reads 1 g), and which part of the head touches the floor first. Same numbers for any body, so the old neck and
the new one can be compared. Since 2026-10-25 the floor is hard like a desk (floor.py; before that it gave like a
foam mat, 22 mm deep in a side fall, and the head stops read ~5x softer). The shell and rubber are still perfectly
rigid in the model, so the peaks are an upper bound: compare bodies with these numbers, don't read them as real
breaking loads.

With neck B (neck_b.py: two servos sharing nod and tilt) it also reports the tilt twist and each servo's share; the
neck servos' holding stiffness can be set with NECK_KP (default 25, the same as every other joint). Every servo is held
to its real strength and speed (servos.py; before 2026-10-25 evening they could all push 10 Nm).

usage: fall_test.py <scene.xml> <out folder> [shove N, default 12] [direction: front|back|side, default front]
writes fall_<direction>.npz (time series) and prints the summary
"""
import os, sys

import mujoco
import numpy as np

from replay import Recorder
import servos as S

NECK_KP = float(os.environ.get("NECK_KP", 25))     # neck B servos' holding stiffness (env, to try a softer neck)
NECK_KD = NECK_KP * 0.6 / 25
STAND, SHOVE, AFTER = 1.0, 0.2, 2.5                 # seconds standing, shoving, then falling and lying there
CONTACT = [int(t) for t in (mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS, mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
                            mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)]


def real(tau, vel, stall, speed):
    servo = ~np.isnan(stall)
    out = tau.copy()
    out[servo] = S.cap(tau[servo], vel[servo], stall[servo], speed[servo])
    return out


def main():
    scene, out = sys.argv[1], sys.argv[2]
    push = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    way = sys.argv[4] if len(sys.argv) > 4 else "front"
    os.makedirs(out, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ja = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT]
    jnt = m.actuator_trnid[ja, 0]
    qa, va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
    home = d.qpos[qa].copy()
    ta = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON]   # neck B's shared servos
    ten = m.actuator_trnid[ta, 0]
    ten_home = d.ten_length[ten].copy()
    stall, speed = S.limits(m)                          # real servo strength and speed (servos.py)
    roll = m.jnt_dofadr[m.joint("neck_roll").id] if "neck_roll" in [m.joint(j).name for j in range(m.njnt)] else None
    torso, head = m.body("torso").id, m.body("head").id
    nod = m.jnt_dofadr[m.joint("neck_pitch").id]
    floor = m.geom("floor").id
    under = lambda b: b == head or (b > 0 and under(m.body_parentid[b]))
    head_geoms = {g for g in range(m.ngeom) if under(m.geom_bodyid[g])}
    force = {"front": [1, 0, 0], "back": [-1, 0, 0], "side": [0, 1, 0]}[way]
    rows, first = [], None
    rec = Recorder(m, f"fall test: shoved {way}, {push:.0f} N")
    for i in range(int((STAND + SHOVE + AFTER) / m.opt.timestep)):
        d.ctrl[ja] = real(25 * (home - d.qpos[qa]) - 0.6 * d.qvel[va], d.qvel[va], stall[ja], speed[ja])
        d.ctrl[ta] = real(NECK_KP * (ten_home - d.ten_length[ten]) - NECK_KD * d.ten_velocity[ten],
                          d.ten_velocity[ten], stall[ta], speed[ta])
        d.xfrc_applied[torso, :3] = np.array(force) * push if STAND <= d.time < STAND + SHOVE else 0
        mujoco.mj_step(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        rec.step(d)
        if first is None:
            for c in d.contact[:d.ncon]:
                g = {c.geom1, c.geom2}
                if floor in g and g & head_geoms:
                    first = (round(d.time, 3), m.geom((g - {floor}).pop()).name or "head shell")
        # force through the nod pin = what the neck post passes to the head (cfrc_int: torque[3], force[3])
        # twist the neck holds about the nod axis = servo + end stop + linkage (every constraint except floor contacts)
        f = d.efc_force.copy()
        f[np.isin(d.efc_type, CONTACT)] = 0
        held = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, held, f)
        acc = np.zeros(6)                               # head's own acceleration at its balance point (still = 1 g)
        mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_XBODY, head, acc, 0)
        tilt = held[roll] + d.qfrc_actuator[roll] if roll is not None else 0.0
        rows.append((d.time, np.linalg.norm(d.cfrc_int[head, 3:]), held[nod] + d.qfrc_actuator[nod],
                     np.linalg.norm(acc[3:]) / 9.81, d.xpos[head][2], tilt))
    t, pin, tq, acc, hz, tl = np.array(rows).T
    np.savez(os.path.join(out, f"fall_{way}.npz"), t=t, pin_N=pin, nod_Nm=tq, head_g=acc, head_z=hz, tilt_Nm=tl,
             qpos=np.array(rec.frames), fps=rec.fps)
    rec.publish(scene)                              # Immy's live window replays it (live.py)
    fall = t >= STAND
    weight = m.body_subtreemass[head] * 9.81
    print(f"{os.path.basename(scene)}  shove {push:.0f} N {way}  head+neck above pin {m.body_subtreemass[head]:.3f} kg")
    print(f"  first head part on the floor: {first}")
    print(f"  pin force: standing {pin[~fall].mean():.1f} N (its weight {weight:.1f} N), fall peak {pin[fall].max():.1f} N "
          f"at {t[fall][pin[fall].argmax()]:.3f} s")
    print(f"  nod twist: standing {np.abs(tq[~fall]).mean():.3f} Nm, fall peak {np.abs(tq[fall]).max():.2f} Nm")
    if roll is not None:     # neck B: left servo = (nod + tilt) / 2, right = (nod - tilt) / 2 of what the neck holds
        sl, sr = (tq + tl) / 2, (tq - tl) / 2
        print(f"  tilt twist: fall peak {np.abs(tl[fall]).max():.2f} Nm; per servo (shared): left peak "
              f"{np.abs(sl[fall]).max():.2f} Nm, right peak {np.abs(sr[fall]).max():.2f} Nm")
    print(f"  head hardest stop {acc[fall].max():.0f} g; torso height at end {d.xpos[torso][2]:.3f} m")


if __name__ == "__main__":
    main()
