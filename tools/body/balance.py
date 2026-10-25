#!/usr/bin/env python3
"""Little Guy's balance reflex (Immy, 2026-10-25: "rubber edge on the ear cups, and then balance reflex"; he picked the
hand-built reflex first, a learned one later once he walks).

How the small servo humanoids do it (balance-1025/RESEARCH.md: DARwIn-OP/HP, SURENA-Mini): a tilt sensor in the chest
(IMU: gyro + accelerometer, like the inner ear), read ~100 times a second, plus every servo's own angle (encoders).
Together they say where his balance point (centre of mass) is over his feet and how fast it is moving. From that,
the capture point: where his balance point is heading, i.e. the spot he would have to step on to stop. While that spot
is still under his feet he can save himself by pushing with his ankles, like swaying on a bus (ankle strategy): the
ankle servos shift their goals so the feet push the body back.

  lean    = angle of the balance point over the middle of the ankles (from IMU tilt + joint angles)
  capture = lean + lean speed / w0,  w0 = sqrt(g / height of the balance point)
  ankles  = K * (capture - capture when standing still), toes-down / toes-up and foot tilt, through the two rods

The IMU here is modelled, not perfect: sampled at 100 Hz, gyro noise and a gyro offset (bias), accelerometer noise,
one tick of delay, and a complementary filter (gyro for fast changes, accelerometer for which way is down) like a
real board would run. Its numbers are typical small MEMS IMU values (BMI088 class), not a chosen part yet.

usage:
  balance.py imu <body scene.xml> <out scene.xml>     build step: adds the IMU site and sensors in the chest
  from balance import Reflex; r = Reflex(m, servos); then every sim step: r.step(d) before servos.apply(d)
"""
import sys

import mujoco
import numpy as np

RATE = 100                                   # control loop, Hz (DARwIn-OP: 100, DARwIn-HP: 60)
GYRO_NOISE, GYRO_BIAS, ACC_NOISE = 0.005, 0.01, 0.03   # rad/s rms, rad/s, m/s^2 rms
BLEND = 0.98                                 # complementary filter: share of the gyro path each tick (~0.5 s)
K = float(__import__("os").environ.get("BALANCE_K", 1.0))   # ankle angle per capture-point angle
RW = float(__import__("os").environ.get("BALANCE_RW", 1.0))  # weight of the lean-speed part of the capture point
KH = float(__import__("os").environ.get("BALANCE_KH", 0.0))  # hip bend per capture-point angle (hip strategy)
KHR = float(__import__("os").environ.get("BALANCE_KHR", 0.0))  # hip roll per sideways capture-point angle
DEAD = float(__import__("os").environ.get("BALANCE_DEAD", 0.0))  # hips only act past this capture error, rad
HIP_LIMIT = 0.6                              # biggest hip correction, rad
LIMIT = 0.3                                  # biggest ankle correction, rad
PITCH_RATIO, ROLL_RATIO = 0.64, 0.71         # foot angle per servo angle through the rods (measured, hanging)


def add_imu(src, out):
    spec = mujoco.MjSpec.from_file(src)
    torso = spec.body("torso")
    m0 = spec.compile()
    site = torso.add_site(name="imu", pos=m0.body_ipos[m0.body("torso").id], size=[0.005, 0, 0])
    spec.add_sensor(name="imu_gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname=site.name)
    spec.add_sensor(name="imu_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname=site.name)
    spec.compile()
    open(out, "w").write(spec.to_xml())
    print(f"{out}: IMU in the chest (gyro + accelerometer at the torso's balance point)")


class Reflex:
    def __init__(self, m, servos, seed=0):
        self.m, self.s, self.on = m, servos, True
        self.rng = np.random.default_rng(seed)
        self.every = max(1, round(1 / RATE / m.opt.timestep))
        self.gyro = m.sensor_adr[m.sensor("imu_gyro").id]
        self.acc = m.sensor_adr[m.sensor("imu_acc").id]
        self.site = m.site("imu").id
        names = [m.joint(j).name for j in servos.jnt]
        self.ankle = {side: (names.index(f"{side}_ankle_a"), names.index(f"{side}_ankle_b")) for side in ("left", "right")}
        self.hips = [names.index(f"{side}_hip_pitch") for side in ("left", "right")]
        self.hip_rolls = [names.index(f"{side}_hip_roll") for side in ("left", "right")]
        self.feet = [m.body("left_foot").id, m.body("right_foot").id]
        self.free_q = servos.fq
        self.scratch = mujoco.MjData(m)
        self.reset()

    def reset(self):
        self.n, self.q_est, self.pending = 0, None, None
        self.bias = self.rng.normal(0, GYRO_BIAS, 3)
        self.prev, self.rate, self.base = None, np.zeros(2), None
        self.s.offset[:] = 0

    def _tilt(self, d):
        """estimated chest orientation (quaternion) from the noisy IMU, complementary filter"""
        dt = 1 / RATE
        w = d.sensordata[self.gyro:self.gyro + 3] + self.bias + self.rng.normal(0, GYRO_NOISE, 3)
        a = d.sensordata[self.acc:self.acc + 3] + self.rng.normal(0, ACC_NOISE, 3)
        site_q = np.zeros(4)
        mujoco.mju_mat2Quat(site_q, d.site_xmat[self.site])
        if self.q_est is None:                         # power-on: trust the accelerometer for which way is down
            self.q_est = site_q.copy()                 # (yaw doesn't matter for balance)
            return self.q_est
        q = self.q_est.copy()
        mujoco.mju_quatIntegrate(q, w, dt)             # gyro path
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        up_est = R.reshape(3, 3).T @ np.array([0, 0, 1.0])    # where the estimate thinks "up" is, in the chest frame
        up_acc = a / (np.linalg.norm(a) + 1e-9)               # where the accelerometer says up is
        err = np.cross(up_acc, up_est)                        # small turn of the estimate that lines them up
        mujoco.mju_quatIntegrate(q, err * (1 - BLEND) * RATE, dt)
        mujoco.mju_normalize4(q)
        self.q_est = q
        return q

    def _lean(self, d, q_chest):
        """balance point over the ankles (forward, sideways angles) using the estimated chest tilt + joint angles"""
        sd = self.scratch
        sd.qpos[:] = d.qpos
        torso_q = np.zeros(4)                          # free joint holds the torso; IMU site is fixed in it
        mujoco.mju_mat2Quat(torso_q, d.xmat[self.m.body("torso").id])
        site_q = np.zeros(4)
        mujoco.mju_mat2Quat(site_q, d.site_xmat[self.site])
        rel = np.zeros(4)                              # chest -> torso body, fixed
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, site_q)
        mujoco.mju_mulQuat(rel, neg, torso_q)
        est = np.zeros(4)
        mujoco.mju_mulQuat(est, q_chest, rel)
        sd.qpos[self.free_q + 3:self.free_q + 7] = est
        mujoco.mj_kinematics(self.m, sd)
        mujoco.mj_comPos(self.m, sd)
        v = sd.subtree_com[0] - (sd.xpos[self.feet[0]] + sd.xpos[self.feet[1]]) / 2
        return np.array([np.arctan2(v[0], v[2]), np.arctan2(v[1], v[2])]), v[2]

    def step(self, d):
        self.n += 1
        if self.n % self.every:
            return
        q = self._tilt(d)
        lean, h = self._lean(d, q)
        if self.prev is not None:                      # lean speed, lightly smoothed
            self.rate = 0.7 * self.rate + 0.3 * (lean - self.prev) * RATE
        self.prev = lean
        capture = lean + RW * self.rate / np.sqrt(9.81 / max(h, 0.05))
        if self.base is None:
            self.base = capture.copy()
        out = np.clip(K * (capture - self.base), -LIMIT, LIMIT) if self.on else np.zeros(2)
        pitch, roll = out[0], -out[1]                  # forward lean: toes down (+pitch); left lean: foot rolls -
        e = capture - self.base
        e = np.sign(e) * np.maximum(np.abs(e) - DEAD, 0)          # small sways: ankles only
        hip = np.clip(np.array([KH, KHR]) * e, -HIP_LIMIT, HIP_LIMIT) if self.on else np.zeros(2)
        cmd = self.pending                             # one tick of delay (IMU read -> servo bus)
        self.pending = (pitch, roll, hip)
        if cmd is None:
            return
        pitch, roll, hip = cmd
        for h in self.hips:
            self.s.offset[h] = hip[0]
        for h in self.hip_rolls:
            self.s.offset[h] = hip[1]
        for side, (a, b) in self.ankle.items():
            sr = roll if side == "left" else -roll     # the right foot's rods are mirrored
            self.s.offset[a] = pitch / PITCH_RATIO + sr / ROLL_RATIO   # same way = pitch, opposite = roll
            self.s.offset[b] = pitch / PITCH_RATIO - sr / ROLL_RATIO


if __name__ == "__main__":
    if sys.argv[1] == "imu":
        add_imu(sys.argv[2], sys.argv[3])
