#!/usr/bin/env python3
"""A hard floor, like a desk (found 2026-10-25 while testing the rubber edge on the ears).

Every part that can touch the floor has a contact pair with it. Those used MuJoCo's default softness (solref time
constant 0.02 s), which is fine for standing (feet sink 0.4 mm) but in a fall acts like a thick foam mat: the head
hit at 2.8 m/s and sank 22 mm into the floor. That made every fall test softer than a real desk and hid which part
really lands.

This sets the floor contacts to 0.004 s, the stiffest MuJoCo stays steady with at our 2 ms step (it wants at least
2 steps). In the same side fall the deepest sink is now 2.5 mm (a 1 ms step only gets it to 1.7 mm and runs half as
fast, so the step stays). Standing, the feet sink 0.02 mm. Rubber parts get the same hard contact for now: how soft
the real TPU is gets measured on a printed part, not guessed here.

Runs last in the build chain.
usage: floor.py <body scene.xml> <out scene.xml>
"""
import sys

import mujoco

TIMECONST = 0.004


def main():
    src, out = sys.argv[1], sys.argv[2]
    spec = mujoco.MjSpec.from_file(src)
    n = 0
    for p in spec.pairs:
        if "floor" in (p.geomname1, p.geomname2):
            p.solref = [TIMECONST, 1]
            n += 1
    m = spec.compile()
    open(out, "w").write(spec.to_xml())
    print(f"{out}: hard floor ({n} contact pairs at {TIMECONST} s), mass {m.body_subtreemass[0]:.2f} kg")


if __name__ == "__main__":
    main()
