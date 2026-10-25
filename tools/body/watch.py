#!/usr/bin/env python3
"""Watch the servo check live: opens MuJoCo's viewer and repeats stand, squat, stand at real speed, with the same
servo limits as servo_check.py. Drag to turn the camera, scroll to zoom, double-click a part to select it; close the
window to stop. On a Mac the viewer needs MuJoCo's own launcher: mjpython watch.py <scene.xml> [squat depth]."""
import sys, time

import mujoco
import mujoco.viewer

from servo_check import run

scene = sys.argv[1]
depth = float(sys.argv[2]) if len(sys.argv) > 2 else 1.9
m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
home = d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]].copy()

with mujoco.viewer.launch_passive(m, d) as v:
    start = [time.time(), d.time]

    def show(d):
        if not v.is_running():
            return False
        ahead = (d.time - start[1]) - (time.time() - start[0])   # keep the sim at wall-clock speed
        if ahead > 0:
            time.sleep(ahead)
        if int(d.time / m.opt.timestep) % 8 == 0:                 # ~60 frames a second
            v.sync()

    while v.is_running():
        start[:] = [time.time(), 0.0]
        if run(m, home, depth, True, d, show) is None:
            break
