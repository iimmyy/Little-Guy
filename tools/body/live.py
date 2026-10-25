#!/usr/bin/env python3
"""Little Guy V2, live: MuJoCo's own viewer window on the newest body, so Immy can look at him any time, like a
Blender viewport (Immy, 2026-10-25: "could I just view it? ... open up the program and look what you're doing").

It shows the body file CURRENT.xml in the body-v2 results folder. Every time a new body is built, the build copies it
there, and this window notices within a second and reloads by itself (the window closes and reopens).
He stands with every servo holding his standing pose, like the fall test, so you can also poke him.

In the window:
  left-drag = turn around him, right-drag = move, scroll = zoom
  play with him (teach mode, Immy: "grab his head and move it just to see how it would move"): double-click a part,
    then Ctrl + right-drag pulls it (Ctrl + left-drag twists it). While you pull, the servos in that limb go soft and
    let it move, only as far as its joints allow; let go and they hold the new pose, like a real robot posed by hand
  7 = hang him on a stand (feet off the floor, so he can't fall while you play) / put him back down
  8 = stand him back up in his standing pose (also Backspace)
  Tab / Shift-Tab = hide or show the side panels; T = see-through, C = contact points, J = joint axes
When a test finishes (fall_test.py and later ones), the window replays it: once at real speed, once at 1/4 speed, with
the test's name and the time in the top left, then goes back to him standing. A test on a different body than the
newest one opens that body for the replay.

usage (on a Mac the viewer needs MuJoCo's own launcher):
  mjpython live.py [body scene.xml, default CURRENT.xml]
"""
import os, sys, time

import mujoco
import mujoco.viewer
import numpy as np

from replay import REPLAY
import servos as S

CURRENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".run", "results", "mind", "body-v2",
                       "CURRENT.xml")
SLOW = 4                                                 # slow-motion replays run at 1/4 speed


def stand_still(m, d):
    """every servo holds its standing pose (same holding as fall_test.py)"""
    servos = Servos(m, d)
    return lambda m, d: servos.apply(d)


class Servos:
    """every servo holds a goal pose; in teach mode the servos of the limb being pulled go soft and take the new pose"""
    STIFF, SOFT = (25, 0.6), (1.5, 0.05)                 # holding stiffness (Nm/rad) and damping

    def __init__(self, m, d):
        self.m = m
        self.ja = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT]
        self.jnt = m.actuator_trnid[self.ja, 0]
        self.qa, self.va = m.jnt_qposadr[self.jnt], m.jnt_dofadr[self.jnt]
        self.ta = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON]
        self.ten = m.actuator_trnid[self.ta, 0]
        stall, speed = S.limits(m)                         # real servo strength and speed (servos.py)
        self.jlim, self.tlim = (stall[self.ja], speed[self.ja]), (stall[self.ta], speed[self.ta])
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.fq, self.fv = m.jnt_qposadr[free], m.jnt_dofadr[free]
        self.reset(d)

    def reset(self, d):
        mujoco.mj_resetDataKeyframe(self.m, d, 0)
        mujoco.mj_forward(self.m, d)
        self.goal, self.ten_goal = d.qpos[self.qa].copy(), d.ten_length[self.ten].copy()
        self.soft, self.ten_soft = np.zeros(len(self.ja), bool), np.zeros(len(self.ta), bool)
        self.offset = np.zeros(len(self.ja))               # added on top of the goals (the balance reflex)
        self.pin = None

    def subtree(self, root):
        return {b for b in range(self.m.nbody) if self._under(b, root)}

    def _under(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    def grab(self, body):
        """the limb holding this body goes soft: from the body up to where the limb meets the torso (or the hips)"""
        m, torso = self.m, self.m.body("torso").id
        pelvis = m.body("pelvis_link").id
        chain, b = [], body
        while b > 0 and b != torso:
            chain.append(b)
            b = m.body_parentid[b]
        if not chain:
            return
        if pelvis in chain:
            legs = [c for c in chain if m.body_parentid[c] == pelvis]
            if not legs:                                   # the hips themselves: the waist servos give
                self.soft[:] = [m.joint(j).name.startswith("waist") for j in self.jnt]
                return
            root = legs[0]                                 # a leg, from the hip down
        else:
            root = chain[-1]                               # an arm, or the neck and head
        bodies = self.subtree(root)
        self.soft[:] = [m.jnt_bodyid[j] in bodies for j in self.jnt]
        self.ten_soft[:] = m.body("head").id in bodies

    def let_go(self, d):
        """the soft servos take the pose they are in as their new goal, then stiffen again"""
        self.goal[self.soft] = d.qpos[self.qa][self.soft]
        self.ten_goal[self.ten_soft] = d.ten_length[self.ten][self.ten_soft]
        self.soft[:], self.ten_soft[:] = False, False

    def hang(self, d, on):
        """hold him up on an invisible stand, feet just off the floor"""
        if on:
            self.pin = self.m.key_qpos[0][self.fq:self.fq + 7].copy()
            self.pin[2] += 0.06
        else:
            self.pin = None

    def apply(self, d):
        k = np.where(self.soft, self.SOFT[0], self.STIFF[0])
        c = np.where(self.soft, self.SOFT[1], self.STIFF[1])
        d.ctrl[self.ja] = real(k * (self.goal + self.offset - d.qpos[self.qa]) - c * d.qvel[self.va], d.qvel[self.va],
                               *self.jlim)
        kt = np.where(self.ten_soft, self.SOFT[0], self.STIFF[0])
        ct = np.where(self.ten_soft, self.SOFT[1], self.STIFF[1])
        d.ctrl[self.ta] = real(kt * (self.ten_goal - d.ten_length[self.ten]) - ct * d.ten_velocity[self.ten],
                               d.ten_velocity[self.ten], *self.tlim)
        if self.pin is not None:
            d.qpos[self.fq:self.fq + 7] = self.pin
            d.qvel[self.fv:self.fv + 6] = 0


def real(tau, vel, stall, speed):
    """no servo pushes harder than the real part can (actuators that aren't servos pass through)"""
    servo = ~np.isnan(stall)
    out = tau.copy()
    out[servo] = S.cap(tau[servo], vel[servo], stall[servo], speed[servo])
    return out


def play(v, m, d, frames, fps, label, speed=1.0):
    """show recorded frames (recorded at fps) at the given speed; stops early if the window is closed"""
    t0 = time.time()
    for i, q in enumerate(frames):
        if not v.is_running():
            return
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        v.set_texts((None, None, label, f"{i / fps:.2f} s"))
        v.sync()
        wait = t0 + (i + 1) / (fps * speed) - time.time()
        if wait > 0:
            time.sleep(wait)


def same_file(a, b):
    return open(a, "rb").read() == open(b, "rb").read()


LEGEND = ("double-click a part, Ctrl + right-drag: pull it\n7\n8",
          "(the servos give while you pull, then hold)\nhang on a stand / put down\nstand back up")
KEY_HANG, KEY_STAND = 55, 56                             # number keys 7 and 8 (every letter is a viewer shortcut)


def legend(v):
    v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, *LEGEND))


def main():
    path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else CURRENT)
    seen = os.path.getmtime(REPLAY) if os.path.exists(REPLAY) else 0
    cam = None
    pending = None                                       # a replay whose body differs from the one on screen
    keys = []
    while True:
        scene = pending["scene"] if pending is not None else path
        stamp = os.path.getmtime(path)
        m = mujoco.MjModel.from_xml_path(scene)
        d = mujoco.MjData(m)
        servos = Servos(m, d)
        reopen = False
        with mujoco.viewer.launch_passive(m, d, key_callback=keys.append) as v:
            if cam is None:
                v.cam.lookat[:] = d.xpos[m.body("torso").id]
                v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.1, 150, -10
            else:
                v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = cam
            if pending is not None:                      # play it, then back to the newest body standing
                show_replay(v, m, d, pending)
                pending, reopen = None, True
            legend(v)
            start = check = shown = time.time()
            last, grabbing = d.time, False
            while v.is_running() and not reopen:
                while keys:
                    k = keys.pop(0)
                    if k == KEY_HANG:
                        servos.hang(d, servos.pin is None)
                    elif k == KEY_STAND:
                        servos.reset(d)
                if d.time < last:                        # Backspace reset: back in the stand pose, clock restarts
                    servos.reset(d)
                    start = time.time()
                last = d.time
                with v.lock():
                    sel, active = v.perturb.select, v.perturb.active
                if active and sel > 0 and not grabbing:  # teach mode: the limb being pulled goes soft
                    servos.grab(sel)
                    grabbing = True
                elif not active and grabbing:            # let go: it holds the new pose
                    servos.let_go(d)
                    grabbing = False
                servos.apply(d)
                mujoco.mj_step(m, d)
                ahead = d.time - (time.time() - start)   # keep to real time
                if ahead > 0:
                    time.sleep(ahead)
                if time.time() - shown > 1 / 60:          # ~60 frames a second
                    shown = time.time()
                    v.sync()
                if time.time() - check > 1.0:
                    check = time.time()
                    if os.path.getmtime(path) != stamp:  # a new body was built: reload
                        reopen = True
                    elif os.path.exists(REPLAY) and os.path.getmtime(REPLAY) != seen:   # a test just finished
                        seen = os.path.getmtime(REPLAY)
                        r = dict(np.load(REPLAY))
                        r = {k: (v_.item() if v_.ndim == 0 else v_) for k, v_ in r.items()}
                        if same_file(r["scene"], scene):
                            show_replay(v, m, d, r)
                            servos.reset(d)
                            legend(v)
                            start, last = time.time(), d.time
                        else:
                            pending, reopen = r, True
            cam = (v.cam.lookat.copy(), v.cam.distance, v.cam.azimuth, v.cam.elevation)
            running = v.is_running()
        if not running and not reopen:
            return
        time.sleep(0.3)                                  # let a copy finish


def show_replay(v, m, d, r):
    title = str(r["title"])
    play(v, m, d, r["qpos"], r["fps"], f"replay: {title}")
    time.sleep(0.5)
    play(v, m, d, r["qpos"], r["fps"], f"slow motion (1/{SLOW}): {title}", 1 / SLOW)
    time.sleep(1.0)


if __name__ == "__main__":
    main()
