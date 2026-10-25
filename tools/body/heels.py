#!/usr/bin/env python3
"""Longer heels (Immy, 2026-10-25, flinging his arms in the live window: "it's a little wobbly ... we should have his
heels stick out a little bit more").

A foot can only keep him up while his balance point stays over it (real term: base of support). Leg sketch 1's foot
reaches 4.4 cm behind the ankle and 8 cm in front, so leaning back has the least room. This moves the round heel back
and stretches the sole under it; the toe and the ankle rods stay where they are.

Wider feet (Immy, same day: "it could also be a little wider too. Maybe more on the outside. Like towards his hands"):
the optional 4th number grows each foot sideways, outward only, so the gap between the feet stays the same. The sole
widens and a second round toe and heel sit at the new outer edge, so the outline stays rounded.

usage: heels.py <body scene.xml> <out scene.xml> [extra heel, mm, default 15] [wider outward, mm, default 0]
"""
import sys

import mujoco


def main():
    src, out = sys.argv[1], sys.argv[2]
    extra = (float(sys.argv[3]) if len(sys.argv) > 3 else 15) / 1000
    wide = (float(sys.argv[4]) if len(sys.argv) > 4 else 0) / 1000
    spec = mujoco.MjSpec.from_file(src)
    for side in ("left", "right"):
        heel, sole = spec.geom(f"{side}_heel"), spec.geom(f"{side}_sole")
        heel.pos = [heel.pos[0] - extra, heel.pos[1], heel.pos[2]]
        sole.size = [sole.size[0] + extra / 2, sole.size[1], sole.size[2]]
        sole.pos = [sole.pos[0] - extra / 2, sole.pos[1], sole.pos[2]]
        if wide > 0:
            out_y = (1 if side == "left" else -1) * wide          # outward = away from the other foot
            sole.size = [sole.size[0], sole.size[1] + wide / 2, sole.size[2]]
            sole.pos = [sole.pos[0], sole.pos[1] + out_y / 2, sole.pos[2]]
            foot = spec.body(f"{side}_foot")
            for part in (heel, spec.geom(f"{side}_toe")):
                twin = foot.add_geom(name=f"{part.name}_out", type=part.type, size=part.size, rgba=part.rgba,
                                     mass=part.mass, contype=0, conaffinity=0,
                                     pos=[part.pos[0], part.pos[1] + out_y, part.pos[2]])
                spec.add_pair(geomname1="floor", geomname2=twin.name)
    m = spec.compile()
    open(out, "w").write(spec.to_xml())
    h = spec.geom("left_heel")
    print(f"{out}: heels {extra * 1000:.0f} mm longer, foot now reaches {(-h.pos[0] + h.size[0]) * 100:.1f} cm behind the "
          f"ankle, {(spec.geom('left_sole').size[1]) * 200:.1f} cm wide; mass {m.body_subtreemass[0]:.2f} kg")


if __name__ == "__main__":
    main()
