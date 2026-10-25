#!/usr/bin/env python3
"""Can our cheap servos hold a ToddlerBot-shaped body up?

Loads a MuJoCo body (ToddlerBot's open model as the template), drives every joint the way a hobby servo does (a
stiff spring toward a target angle), and limits each joint's force to what the servo we picked can give at that speed.
Legs get the STS3250 (50 kg.cm, 75 rpm), everything else the ST3215 (30 kg.cm, 45 rpm). The servo's force falls in a
straight line from full at standstill to zero at top speed, the usual first guess for a DC motor before measuring one.

The body stands for 2 s, sinks into a squat over 3 s, holds it 2 s, rises over 3 s, and holds 2 s. Run twice: once with
the servo limits and once without, so the second run shows how much force each joint actually asks for.

Not yet in: our servos' weight (heavier than ToddlerBot's smallest motors), our 60 cm height, the armor. Those come
when the body is ours; this only asks whether the servo class is in the right range.

usage: servo_check.py <scene.xml> <out folder> [squat depth: knee angle in rad, default 1.2]
"""
import json, math, os, sys

import mujoco
import numpy as np

SERVOS = {                                      # stall torque (Nm), rated torque (Nm), top speed (rad/s)
    "STS3250": dict(stall=50 * 0.0981, rated=16 * 0.0981, speed=75 * 2 * math.pi / 60),
    "ST3215": dict(stall=30 * 0.0981, rated=10 * 0.0981, speed=45 * 2 * math.pi / 60),
}
KP, KD = 25.0, 0.6                              # spring and damping of the servo's own angle control
LEG = ("hip", "knee", "ankle")


def servo_for(name):
    return "STS3250" if any(k in name for k in LEG) else "ST3215"


def run(m, home, depth, limited, d=None, on_step=None):
    """one stand-squat-stand; on_step(d) is called after every physics step (watch.py uses it to show it live)"""
    d = d or mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    jnt = m.actuator_trnid[:, 0]
    qadr, vadr = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
    kind = [SERVOS[servo_for(n)] for n in names]
    squat = home.copy()
    for i, n in enumerate(names):                # deeper knee bend; hip and ankle keep their share so the foot stays flat
        if n.endswith("_knee"):
            k, s = home[i], np.sign(home[i]) * depth
            for part in ("hip_pitch", "ankle_pitch"):
                j = names.index(n.replace("knee", part))
                squat[j] = home[j] / k * s
            squat[i] = s
    phases = [(2, 0, 0), (3, 0, 1), (2, 1, 1), (3, 1, 0), (2, 0, 0)]
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    peak, mean, n_steps, low = np.zeros(m.nu), np.zeros(m.nu), 0, 1.0
    clipped, lag = np.zeros(m.nu), np.zeros(m.nu)
    for dur, a, b in phases:
        steps = int(dur / m.opt.timestep)
        for s in range(steps):
            f = a + (b - a) * 0.5 * (1 - math.cos(math.pi * s / steps))
            target = home + f * (squat - home)
            q, v = d.qpos[qadr], d.qvel[vadr]
            tau = KP * (target - q) - KD * v
            lag = np.maximum(lag, np.abs(target - q))
            if limited:
                for i, sv in enumerate(kind):        # what the motor can give at this speed, in this direction
                    pushing_along = np.sign(tau[i]) == np.sign(v[i])
                    cap = sv["stall"] * max(0.0, 1 - abs(v[i]) / sv["speed"]) if pushing_along else sv["stall"]
                    if abs(tau[i]) > cap:
                        clipped[i] += 1
                        tau[i] = np.sign(tau[i]) * cap
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            if on_step and on_step(d) is False:
                return None
            peak = np.maximum(peak, np.abs(tau))
            mean += np.abs(tau)
            n_steps += 1
            low = min(low, float(d.xpos[torso][2]))
    mean /= n_steps
    rows = [dict(joint=n, servo=servo_for(n), peak_Nm=round(float(peak[i]), 3), mean_Nm=round(float(mean[i]), 3),
                 peak_vs_stall=round(float(peak[i] / kind[i]["stall"]), 2),
                 mean_vs_rated=round(float(mean[i] / kind[i]["rated"]), 2),
                 time_at_limit=round(float(clipped[i] / n_steps), 3), worst_lag_rad=round(float(lag[i]), 3)) for i, n in enumerate(names)]
    return dict(limited=limited, torso_lowest_m=round(low, 3), torso_end_m=round(float(d.xpos[torso][2]), 3),
                fell=low < 0.15, joints=rows)


def main():
    scene, out = sys.argv[1], sys.argv[2]
    depth = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    os.makedirs(out, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    home = d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]].copy()
    print(f"body mass {m.body_subtreemass[0]:.2f} kg, {m.nu} servos, squat knee {depth} rad")
    result = dict(scene=os.path.basename(scene), mass_kg=round(float(m.body_subtreemass[0]), 3), depth=depth,
                  servos=SERVOS, kp=KP, kd=KD, runs=[run(m, home, depth, lim) for lim in (False, True)])
    for r in result["runs"]:
        print(f"\n{'with our servo limits' if r['limited'] else 'no limits (what the joints ask for)'}: "
              f"torso lowest {r['torso_lowest_m']} m, end {r['torso_end_m']} m, fell: {r['fell']}")
        print(f"  {'joint':26s} {'servo':8s} {'peak Nm':>8s} {'peak/stall':>10s} {'mean/rated':>10s} {'at limit':>8s} {'lag rad':>7s}")
        for j in sorted(r["joints"], key=lambda j: -j["peak_vs_stall"])[:12]:
            print(f"  {j['joint']:26s} {j['servo']:8s} {j['peak_Nm']:8.3f} {j['peak_vs_stall']:10.2f} "
                  f"{j['mean_vs_rated']:10.2f} {j['time_at_limit']:8.3f} {j['worst_lag_rad']:7.3f}")
    json.dump(result, open(os.path.join(out, "servo_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
