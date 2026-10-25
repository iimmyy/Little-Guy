#!/usr/bin/env python3
"""Torso shapes for Little Guy V2, to look at and pick from (like heads.py did for the head).

Each option hides ToddlerBot's torso box, waist gears and pelvis plate and puts a smooth shell over the same inside
space (ToddlerBot's torso fits the Jetson + battery: 10 x 14.4 x 13.4 cm), so nothing inside has to move and the
shoulders, hips, neck and arms stay exactly where they are (Immy: keep the long arms). Shells are drawn weightless;
their weight goes in once one is picked.

  A  Alpha chest     chunky, squared-off box with rounded edges, like the Alpha 1S
  B  Atlas V         wide chest narrowing to a slim waist, sleek
  C  bean            soft rounded egg, toy-like
  D  head family     the same squircle shape as his I1 head, with a small dark chest window

Each shape is a stack of rounded rings (|y/w|^n + |x/d|^n = 1 at several heights) wrapped in one smooth convex skin.

usage: torso_options.py <body scene.xml> <out folder> [letters, default ABCD]
"""
import os, sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SHELL, DARK, GLASS, ACCENT = [0.9, 0.91, 0.93, 1], [0.16, 0.17, 0.2, 1], [0.03, 0.03, 0.04, 1], [0.35, 0.75, 0.95, 1]
CX = -0.028                                         # torso centre, front-back, in the torso's own frame
HIDE = ("torso", "waist_gears", "pelvis_link", "waist_gear_drive", "waist_gear_drive_2")

# rings: (height above the torso origin, half-width, half-depth); n = squareness (2 = round, bigger = squarer)
OPTIONS = {
    "A": ("A  Alpha chest", 5.0, [(0.0, 0.066, 0.050), (0.02, 0.072, 0.054), (0.11, 0.078, 0.056), (0.135, 0.070, 0.050),
                                  (0.146, 0.050, 0.038)], 5.0, False),
    # B is 1 cm thicker front to back than first drawn (half-depth +5 mm): the room the lower-back woofer, the Jetson and
    # the battery need together (torso_pack fit check, 2026-10-25)
    # and 2 cm thicker again (half-depth +10 mm, Immy 2026-10-25: "2cm is fine"): all 6 real battery cells below the
    # Jetson with the woofer kept low on the back (insides.py fit search: as drawn before, only 2 of 6 fit)
    "B": ("B  Atlas V", 3.0, [(0.0, 0.050, 0.059), (0.05, 0.064, 0.065), (0.11, 0.082, 0.071), (0.135, 0.074, 0.065),
                              (0.147, 0.045, 0.050)], 3.0, False),
    "C": ("C  bean", 2.2, [(-0.005, 0.052, 0.046), (0.03, 0.070, 0.056), (0.09, 0.077, 0.058), (0.13, 0.066, 0.050),
                           (0.152, 0.030, 0.025)], 2.2, False),
    "D": ("D  head family", 3.0, [(0.0, 0.064, 0.050), (0.13, 0.078, 0.056), (0.148, 0.062, 0.046)], 3.0, True),
}
PELVIS = [(-0.03, 0.050, 0.038), (-0.005, 0.068, 0.044), (0.012, 0.064, 0.042)]   # pelvis frame, below the waist


def ring_points(rings, n, cx, seg=48):
    pts = []
    for z, w, dpt in rings:
        for t in np.linspace(0, 2 * np.pi, seg, endpoint=False):
            c, s = np.cos(t), np.sin(t)
            pts.append([cx + dpt * np.sign(c) * abs(c) ** (2 / n), w * np.sign(s) * abs(s) ** (2 / n), z])
    return np.array(pts).ravel().tolist()


def window_points(cx, z, w, h, depth, n=3.0, seg=48):
    """thin squircle plate facing forward (+x), for the chest window"""
    pts = []
    for x in (cx, cx + depth):
        for t in np.linspace(0, 2 * np.pi, seg, endpoint=False):
            c, s = np.cos(t), np.sin(t)
            pts.append([x, w * np.sign(c) * abs(c) ** (2 / n), z + h * np.sign(s) * abs(s) ** (2 / n)])
    return np.array(pts).ravel().tolist()


def build(scene, key):
    label, n, rings, pn, window = OPTIONS[key]
    spec = mujoco.MjSpec.from_file(scene)
    for name in HIDE:
        for g in spec.body(name).geoms:
            if g.group == 2:                          # ToddlerBot's visible meshes (collision boxes stay, invisible)
                g.rgba = [0, 0, 0, 0]
    spec.add_mesh(name=f"torso_{key}", uservert=ring_points(rings, n, CX))
    spec.body("torso").add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"torso_{key}", rgba=SHELL, density=0,
                                contype=0, conaffinity=0)
    spec.body("torso").add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.022, 0.006, 0], pos=[CX, 0, -0.004],
                                rgba=DARK, density=0, contype=0, conaffinity=0)   # waist joint ring
    spec.add_mesh(name=f"pelvis_{key}", uservert=ring_points(PELVIS, pn, -0.025))
    spec.body("pelvis_link").add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"pelvis_{key}", rgba=SHELL, density=0,
                                      contype=0, conaffinity=0)
    if window:
        front = CX + rings[1][2] * 0.97
        spec.add_mesh(name=f"window_{key}", uservert=window_points(front - 0.002, 0.095, 0.032, 0.02, 0.003))
        spec.body("torso").add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"window_{key}", rgba=GLASS, density=0,
                                    contype=0, conaffinity=0)
        spec.body("torso").add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.001, 0.014, 0.0025],
                                    pos=[front + 0.0015, 0, 0.095], rgba=ACCENT, density=0, contype=0, conaffinity=0)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 600, 900
    return label, spec


def main():
    scene, out = sys.argv[1], sys.argv[2]
    keys = sys.argv[3] if len(sys.argv) > 3 else "ABCD"
    os.makedirs(out, exist_ok=True)
    tiles = []
    for key in keys:
        label, spec = build(scene, key)
        open(os.path.join(out, f"torso_{key}.xml"), "w").write(spec.to_xml())
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 900, 600)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        opt.sitegroup[:] = 0
        tile = Image.new("RGB", (600, 1860), (245, 245, 245))
        for i, az in enumerate((180, 140)):            # front, then three-quarter
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0, 0, 0.31], 1.05, az, -6
            r.update_scene(d, cam, opt)
            tile.paste(Image.fromarray(r.render()), (0, 900 * i))
        r.close()                                       # one drawing window at a time, or the next one comes out garbled
        ImageDraw.Draw(tile).text((300, 1830), label, fill=(20, 20, 20), font=ImageFont.load_default(size=32), anchor="mm")
        tiles.append(tile)
    sheet = Image.new("RGB", (600 * len(tiles), 1860))
    for i, t in enumerate(tiles):
        sheet.paste(t, (600 * i, 0))
    sheet.save(os.path.join(out, "torsos.png"))
    print("wrote", os.path.join(out, "torsos.png"))


if __name__ == "__main__":
    main()
