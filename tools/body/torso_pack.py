#!/usr/bin/env python3
"""Pack Little Guy's insides into the torso and check that everything fits (Immy, 2026-10-25: chest speakers with
"pretty decent quality" for music, next to the Jetson and the battery).

Every part is a box in the torso's own frame (x forward, y to his left, z up from the waist). The check:
  - no two parts overlap, and no part overlaps what is already there (shoulder servos, neck turn servo)
  - every corner of every part sits inside the shell with room for its wall
Then it draws the torso see-through with the parts in colour, and can write a scene with them in.

Part sizes (maker numbers, 2026-10-25):
  Jetson Orin Nano Super dev kit  103 x 90.5 x 34.8 mm incl. carrier, module and heatsink+fan, 174-176 g (NVIDIA)
  Tang Band T2-2136SA speaker     70 x 70 x 58 mm: 57 mm aluminium driver + passive radiator in its own sealed box,
                                  80 Hz-20 kHz, 4 ohm, 10 W (Parts Express listing / TB spec)
  TAS5805M amp board              65 x 56 x 20 mm (Sonocotta Louder Raspberry Hat): I2S in, 2 x 23 W, 15-band DSP EQ
  battery                         NOT CHOSEN YET: placeholder 3S2P 18650 Li-ion pack, 65 x 57 x 37 mm, ~290 g (estimate)

usage: torso_pack.py <scene.xml> <out folder> [layout name]
"""
import itertools, os, sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torso_options as T
import torso_v2

WALL = 0.0025
RINGS = T.OPTIONS["B"][2]
FIXED = {   # already in the torso (centre, half size), from the current body
    "shoulder servo L": ((-0.022, 0.062, 0.113), (0.0226, 0.0175, 0.0124)),
    "shoulder servo R": ((-0.022, -0.062, 0.113), (0.0226, 0.0175, 0.0124)),
    "neck turn servo": ((-0.001, 0.0, 0.12), (0.0124, 0.0226, 0.0175)),
}
COLOURS = {"jetson": (0.2, 0.7, 0.3), "speaker": (0.95, 0.6, 0.1), "amp": (0.6, 0.3, 0.8), "battery": (0.85, 0.2, 0.2),
           "fixed": (0.25, 0.45, 0.8)}

# layouts: name -> {part: (kind, centre, half size)}
LAYOUTS = {
    "stereo-high": {
        "Jetson (on the back wall)": ("jetson", (-0.0605, 0.0, 0.088), (0.0174, 0.0515, 0.04525)),
        "speaker L": ("speaker", (-0.009, 0.036, 0.075), (0.029, 0.035, 0.035)),
        "speaker R": ("speaker", (-0.009, -0.036, 0.075), (0.029, 0.035, 0.035)),
        "amp": ("amp", (-0.03, 0.0, 0.025), (0.01, 0.0325, 0.028)),
        "battery": ("battery", (0.0, 0.0, 0.022), (0.0185, 0.0325, 0.0285)),
    },
}


def inside(p):
    """is point p (torso frame) inside the shell, keeping room for the wall?"""
    x, y, z = p
    if not (RINGS[0][0] + WALL <= z <= RINGS[-1][0] - WALL):
        return False
    w, d = torso_v2.profile(RINGS, z)
    w, d = w - WALL, d - WALL
    return (abs(y) / w) ** 3 + (abs(x - T.CX) / d) ** 3 <= 1


def corners(c, h):
    return [np.array(c) + np.array(s) * np.array(h) for s in itertools.product((-1, 1), repeat=3)]


def overlap(a, b):
    (ca, ha), (cb, hb) = a, b
    gap = np.abs(np.array(ca) - np.array(cb)) - (np.array(ha) + np.array(hb))
    return gap.max() < 0, gap    # overlapping if every axis gap is negative


def check(layout):
    parts = {n: (c, h) for n, (_, c, h) in layout.items()}
    problems = []
    for (na, a), (nb, b) in itertools.combinations(list(parts.items()) + list(FIXED.items()), 2):
        if na in FIXED and nb in FIXED:
            continue
        hit, gap = overlap(a, b)
        if hit:
            problems.append(f"{na} overlaps {nb} by {(-gap).min() * 1000:.1f} mm")
    for n, (c, h) in parts.items():
        out = [p for p in corners(c, h) if not inside(p)]
        if out:
            problems.append(f"{n}: {len(out)} of 8 corners outside the shell (e.g. {np.round(out[0], 3)})")
    return problems


def scene(src, layout):
    spec = mujoco.MjSpec.from_file(src)
    torso = spec.body("torso")
    for n, (kind, c, h) in layout.items():
        torso.add_geom(name=f"pack_{n}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c, size=h,
                       rgba=list(COLOURS[kind]) + [1], density=0, contype=0, conaffinity=0, group=1)
    for n, (c, h) in FIXED.items():
        torso.add_geom(name=f"pack_{n}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c, size=h,
                       rgba=list(COLOURS["fixed"]) + [1], density=0, contype=0, conaffinity=0, group=1)
    return spec


def render(spec, out_png, title):
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    torso = m.body("torso").id
    for g in range(m.ngeom):        # torso shell and everything else on the torso see-through, packed parts solid
        if m.geom_bodyid[g] == torso and not m.geom(g).name.startswith("pack_") and m.geom_rgba[g][3] > 0:
            m.geom_rgba[g][3] = 0.12
    r = mujoco.Renderer(m, 600, 600)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    opt.sitegroup[:] = 0
    font = ImageFont.load_default(size=22)
    tiles = []
    for az, name in ((180, "front"), (90, "side"), (140, "front, angled"), (0, "back")):
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [-0.03, 0, 0.40], 0.45, az, -12
        r.update_scene(d, cam, opt)
        img = Image.fromarray(r.render())
        ImageDraw.Draw(img).text((300, 575), name, fill=(240, 240, 240), font=font, anchor="mm")
        tiles.append(img)
    r.close()
    sheet = Image.new("RGB", (600 * len(tiles), 660), (245, 245, 245))
    for i, t in enumerate(tiles):
        sheet.paste(t, (600 * i, 0))
    ImageDraw.Draw(sheet).text((300 * len(tiles), 630), title, fill=(20, 20, 20), font=font, anchor="mm")
    sheet.save(out_png)


def main():
    src, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or list(LAYOUTS)
    os.makedirs(out, exist_ok=True)
    for name in names:
        problems = check(LAYOUTS[name])
        print(f"{name}: {'fits' if not problems else str(len(problems)) + ' problems'}")
        for p in problems:
            print("   ", p)
        spec = scene(src, LAYOUTS[name])
        render(spec, os.path.join(out, f"pack-{name}.png"),
               f"{name}: Jetson green, speakers orange, amp purple, battery red (placeholder), servos blue")


if __name__ == "__main__":
    main()


# ---- automatic search: place parts one by one (largest first) wherever they fit, any of 6 orientations ----
PARTS_SEARCH = {   # name: (kind, full size in metres, any order) — sizes per the docstring
    "Jetson on compact carrier": ("jetson", (0.091, 0.0605, 0.030)),     # Forecr DSBOARD-ORNXS 91 x 60.5 + module + heatsink
    "speaker module": ("speaker", (0.070, 0.070, 0.058)),                 # Tang Band T2-2136SA
    "battery (placeholder)": ("battery", (0.065, 0.057, 0.037)),
    "amp": ("amp", (0.045, 0.035, 0.012)),                                # bare TAS5805M board, placeholder size
}


def search(depth_add=0.0, step=0.003):
    global RINGS
    RINGS = [(z, w, d + depth_add) for z, w, d in T.OPTIONS["B"][2]]
    placed = {}
    for name, (kind, size) in PARTS_SEARCH.items():
        best = None
        for perm in set(itertools.permutations(size)):
            h = np.array(perm) / 2
            for x in np.arange(-0.085, 0.03, step):
                for z in np.arange(0.0, 0.15, step):
                    for y in (0.0,) if name != "amp" else np.arange(-0.06, 0.061, step):
                        c = (x, y, z)
                        if not all(inside(p) for p in corners(c, h)):
                            continue
                        others = list(FIXED.values()) + [(pc, ph) for _, pc, ph in placed.values()]
                        if any(overlap((c, h), o)[0] for o in others):
                            continue
                        score = abs(x - (0.0 if kind == "speaker" else -0.06)) + abs(z - 0.08) * 0.3   # speaker forward, rest back
                        if best is None or score < best[0]:
                            best = (score, c, tuple(h))
        if best is None:
            return placed, name
        placed[name] = (kind, best[1], best[2])
    return placed, None


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "search":
    pass
