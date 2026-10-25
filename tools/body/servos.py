#!/usr/bin/env python3
"""How strong and fast each of Little Guy's servos really is, so the simulation can't make him stronger than his parts
(Immy, 2026-10-25, flinging his arm in the live window: "would he move like this in real life?"; until now every
servo in the model could push 10 Nm, about 3x a real ST3215).

Maker numbers (Feetech datasheets):
  STS3250  legs                         50 kg.cm stall, 16 rated, 75 rpm
  ST3215   waist, shoulders, elbows,    30 kg.cm stall, 10 rated, 45 rpm (12 V)
           neck (turn + the two shared nod/tilt servos)
  STS3032  forearm twist                4.5 kg.cm stall, 1.5 rated, 111 rpm (6 V)
  SCS0009  wrist (two, through rods)    2.3 kg.cm stall, 0.75 rated, 100 rpm (6 V)
The finger pulls already have their own limit (28 N of wire pull, set where the hand is built).

A servo's push falls as it speeds up: at top speed it has nothing left to push with in that direction (it can still
brake at full strength). Same rule as servo_check.py.
"""
import math

import mujoco
import numpy as np

KGCM = 0.0981                                     # 1 kg.cm in Nm
RPM = 2 * math.pi / 60
SPECS = {
    "STS3250": dict(stall=50 * KGCM, rated=16 * KGCM, speed=75 * RPM),
    "ST3215": dict(stall=30 * KGCM, rated=10 * KGCM, speed=45 * RPM),
    "STS3032": dict(stall=4.5 * KGCM, rated=1.5 * KGCM, speed=111 * RPM),
    "SCS0009": dict(stall=2.3 * KGCM, rated=0.75 * KGCM, speed=100 * RPM),
}


def servo_for(name):
    """which servo drives the actuator with this name (None = not a servo, e.g. a finger pull)"""
    if name.endswith("_pull"):
        return None
    if any(k in name for k in ("hip", "knee", "ankle")):
        return "STS3250"
    if "forearm_twist" in name:
        return "STS3032"
    if "wrist" in name:
        return "SCS0009"
    return "ST3215"


def limits(m):
    """stall torque and top speed per actuator (nan where the actuator isn't a servo)"""
    stall, speed = np.full(m.nu, np.nan), np.full(m.nu, np.nan)
    for a in range(m.nu):
        kind = servo_for(m.actuator(a).name)
        if kind:
            stall[a], speed[a] = SPECS[kind]["stall"], SPECS[kind]["speed"]
    return stall, speed


def cap(tau, vel, stall, speed):
    """what each servo can actually give: full stall when braking, less the faster it already turns that way"""
    along = np.sign(tau) == np.sign(vel)
    room = np.where(along, stall * np.clip(1 - np.abs(vel) / speed, 0, 1), stall)
    return np.clip(tau, -room, room)
