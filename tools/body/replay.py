#!/usr/bin/env python3
"""Hand a test's motion to Immy's live window (live.py): a test records his pose ~60 times a second and publishes it
here when it finishes; the window plays it at real speed, then in slow motion, then goes back to him standing.

  rec = Recorder(model, "fall test: front, 15 N")   then, inside the test's loop:  rec.step(data)
  rec.publish(scene_path)                           when the test is done

The newest run is always REPLAY.npz in the body-v2 results folder (one file, replaced each time; each test also keeps
its own copy in its own results folder).
"""
import os

import numpy as np

REPLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".run", "results", "mind", "body-v2",
                      "REPLAY.npz")
FPS = 60


class Recorder:
    def __init__(self, m, title):
        self.title, self.every, self.frames = title, max(1, round(1 / FPS / m.opt.timestep)), []
        self.fps = 1 / (self.every * m.opt.timestep)
        self.n = 0

    def step(self, d):
        if self.n % self.every == 0:
            self.frames.append(d.qpos.copy())
        self.n += 1

    def publish(self, scene):
        tmp = REPLAY + ".tmp.npz"
        np.savez(tmp, scene=os.path.abspath(scene), title=self.title, fps=self.fps, qpos=np.array(self.frames))
        os.replace(tmp, REPLAY)                          # the window never sees a half-written file
