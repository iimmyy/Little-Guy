#!/usr/bin/env python3
"""Two folding carry handles on Little Guy's upper back (Immy, 2026-10-25: "Just two vertical ones that I can just grab
and pick him up"; "They should be foldable. They should fold inwards, and then I can fold them out and then click them
into place").

Each handle is sized like a coffee mug handle (Immy: "most coffee mugs are big enough to fit four fingers"): 8 cm of
bar (7 cm opening), 1.4 cm thick, 2.2 cm of finger room behind it (first try was 10 / 2 / 3 cm: "way too big").
Each one swings on a vertical hinge where it meets the back, like a door:
  folded  the handle swings in toward his spine and lies flat against the back, either side of the woofer grille
          (the hinges sit 6 cm out from the middle so the folded bars clear it: Immy, 2026-10-25, the first
          version's bars lay over the grille; the way in is now a door on his chest, so they no longer lock a hatch)
  out     swung 90 deg straight back and clicked into place (a spring detent in the hinge, like a luggage handle)
Vents (Immy: "the handles themselves are vents ... tiny little holes or slits ... mainly at the top and the bottom,
not the actual part where I wrap my palm around it"): each handle is hollow; warm air from inside the torso goes
through the hollow hinge into the top and bottom arms and out of small slits on their outer faces (top face of the
top arm, underside of the bottom arm). The grip bar has none, so the part in your palm stays cool: the heat leaves
before it gets there. The back's own exhaust grille stays as it was (folded, the grips lie
just off its outer ends, so air still gets out round them).
The handles sit either side of the exhaust grille and the woofer grille, and they touch the floor first in a fall
backwards, so they count as bumpers in both positions.

usage: handles.py <body scene.xml> <out scene.xml> [out|folded, default out]
"""
import math, sys

import mujoco
import numpy as np

import parts
import torso_options as T
import torso_v2

Y, Z0, Z1 = 0.060, 0.055, 0.135         # hinge line across the back, bottom and top of the bar
BAR_R, GAP, POST = 0.007, 0.022, 0.006  # grip radius, finger room behind the bar, arm half-thickness (hollow)
REACH = GAP + BAR_R                     # hinge to bar centre
SLITS = 5                               # vent slits per handle arm
DARK = [0.16, 0.17, 0.2, 1]


def main():
    src, out = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "out"
    spec = mujoco.MjSpec.from_file(src)
    torso = spec.body("torso")
    rings = T.OPTIONS["B"][2]
    def back(z, y=0.0):
        """the back surface: front-back position at height z, y across (the shell is a rounded square, n=3)"""
        w, d = torso_v2.profile(rings, z)
        return T.CX - d * max(0.0, 1 - abs(y / w) ** 3) ** (1 / 3)
    kg = parts.PARTS["carry handles (2), printed"][0] / 1000 / 2
    add = lambda **kw: torso.add_geom(rgba=DARK, density=0, contype=0, conaffinity=0, **kw)
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    for s in (-1, 1):
        name = f"handle_{'left' if s > 0 else 'right'}"
        zc = (Z0 + Z1) / 2
        rear = min(back(z, s * Y) for z in np.linspace(Z0, Z1, 21))         # the back bulges: the bar clears its rearmost point
        if mode == "out":
            xbar = rear - REACH                                     # straight back, clear of the shell
            ends = [(xbar, s * Y, z) for z in (Z0, Z1)]
        else:
            ends = [(rear - BAR_R - 0.003, s * (Y - REACH), z) for z in (Z0, Z1)]          # 3 mm off the vent
        bar = torso.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*ends[0], *ends[1]],
                             size=[BAR_R, 0, 0], rgba=DARK, density=0, contype=0, conaffinity=0)
        spec.add_pair(geomname1="floor", geomname2=name)
        for (xe, ye, z) in ends:                                    # posts: hinge to bar
            xh, yh = back(z, s * Y), s * Y
            ln = math.hypot(xe - xh, ye - yh)
            yaw = math.atan2(ye - yh, xe - xh)
            add(type=box, size=[ln / 2, POST, POST], pos=[(xe + xh) / 2, (ye + yh) / 2, z],
                quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            add(type=cyl, size=[POST * 1.3, POST * 1.2, 0], pos=[xh - 0.001, yh, z])   # hollow hinge barrel + detent
            face = POST if z == Z1 else -POST                       # vent slits: top of the top arm, under the bottom arm
            for f in np.linspace(-0.3, 0.3, SLITS):
                cx, cy = (xe + xh) / 2 + f * (xe - xh), (ye + yh) / 2 + f * (ye - yh)
                torso.add_geom(type=box, size=[0.0007, POST * 0.7, 0.0004], pos=[cx, cy, z + face],
                               quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], rgba=T.GLASS, density=0,
                               contype=0, conaffinity=0)
        w = torso.add_body(name=f"{name}_weight", pos=[np.mean([e[0] for e in ends]), np.mean([e[1] for e in ends]), zc])
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.008, 0, 0], mass=kg, group=4,   # torso has a fixed weight,
                   contype=0, conaffinity=0)                                                 # so handles ride on a ball
    m = spec.compile()
    open(out, "w").write(spec.to_xml())
    print(f"{out}: handles {mode}, mass {m.body_subtreemass[0]:.2f} kg, bars {(Z1 - Z0) * 100:.0f} cm, "
          f"{BAR_R * 200:.1f} cm thick, {GAP * 100:.1f} cm finger room")


if __name__ == "__main__":
    main()
