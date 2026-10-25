#!/usr/bin/env python3
"""Make an already-built V2 body's chest deeper front to back, without rebuilding the whole body chain (Immy,
2026-10-25: "2cm is fine", so all 6 battery cells fit below the Jetson with the woofer kept low on the back).

The chest shell grows the same amount at every height, half in front and half behind its centre line (T.CX), so every
point on its front surface moves forward by ADD and every point on its back surface moves back by ADD. What sits on
the shell follows its wall:
  front   the door, its seam, finger notches, charge port and power button          x + ADD
  back    woofer grille, exhaust grille, both handles (and their weights), woofer    x - ADD
  ledge   the underside vent slots run from the back edge to the waist ring         centre - ADD/2, length + ADD
  hit box the chest's collision box (what touches the floor in a fall)               half-depth + ADD
Shoulders, neck and the side intake slots stay where they are (width and height unchanged).

The shell itself is redrawn from torso_options' B rings, which already carry the new depth.

usage: deepen.py <body scene.xml> <out.xml> [ADD mm, default 10]
"""
import sys

import mujoco
import numpy as np

import torso_options as T

FRONT_FROM = T.CX + 0.03          # torso geoms in front of this sit on the front wall
BACK_FROM = T.CX - 0.03           # ... and behind this on the back wall


def deepen(spec, add):
    torso = spec.body("torso")
    rings = T.OPTIONS["B"][2]
    spec.mesh("torso_B").uservert = T.ring_points(rings, T.OPTIONS["B"][1], T.CX)
    moved = []
    for g in torso.geoms:
        x, y, z = g.pos
        if g.name == "torso_collision_0":
            g.size = [g.size[0] + add, g.size[1], g.size[2]]
            moved.append(("hit box half-depth +", g.name, add))
        elif g.type == mujoco.mjtGeom.mjGEOM_MESH:
            continue
        elif z < 0 and x < T.CX and g.type == mujoco.mjtGeom.mjGEOM_BOX:      # underside vent slots
            g.pos = [x - add / 2, y, z]
            g.size = [g.size[0] + add / 2, g.size[1], g.size[2]]
            moved.append(("ledge slot", g.name or "slot", -add / 2))
        elif x > FRONT_FROM:
            g.pos = [x + add, y, z]
            moved.append(("front", g.name or str(g.type).split("_")[-1], add))
        elif x < BACK_FROM and g.name != "torso_collision_1":
            g.pos = [x - add, y, z]
            moved.append(("back", g.name or str(g.type).split("_")[-1], -add))
    for b in torso.bodies:
        if b.name == "woofer_weight" or b.name.startswith("handle_"):
            b.pos = [b.pos[0] - add, b.pos[1], b.pos[2]]
            moved.append(("back", b.name, -add))
    return moved


def main():
    src, out = sys.argv[1], sys.argv[2]
    add = (float(sys.argv[3]) if len(sys.argv) > 3 else 10.0) / 1000
    spec = mujoco.MjSpec.from_file(src)
    moved = deepen(spec, add)
    spec.compile()
    open(out, "w").write(spec.to_xml())
    counts = {}
    for where, name, dx in moved:
        counts.setdefault((where, round(dx * 1000, 1)), []).append(name)
    for (where, dx), names in counts.items():
        print(f"{where:22s} {dx:+5.1f} mm  x{len(names):2d}  {', '.join(sorted(set(names)))}")
    print("wrote", out)


if __name__ == "__main__":
    main()
