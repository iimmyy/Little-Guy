#!/usr/bin/env python3
"""Little Guy V2 torso B (Immy's pick, 2026-10-25: the Atlas V) bolted on, with a way in and a way for air through.

Front: a door over the lower chest, the way in (Immy, 2026-10-25: "a door on the front, like a door panel"; the
charging port and power button sit on it). Back: the woofer's round grille low on the back (Immy: "think like lower
back"; it fires backward from its own sealed chamber) and an exhaust grille across the upper back, where the Jetson's
fan blows out. Sides: intake slots low on both sides, under the arms. Underside: a
vent at the back of the torso's bottom ledge, blowing down (Immy).
Air path: cool air in low at the sides, past the battery, through the Jetson's heatsink and fan, out high at the back.
Warm air rises, so the path also works a little with the fan off (like low windows and a roof vent in a house).

The Jetson Orin Nano Super makes up to 25 W of heat in its default fast mode (NVIDIA power tables, JetPack 6.2+).
Shell details are weightless: the torso keeps ToddlerBot's torso weight (scaled up), which already counts its case.

usage: torso_v2.py <body scene.xml> <out folder>   writes v2_torsoB.xml and v2-torsoB-views.png
"""
import math, os, sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import parts
import torso_options as T

SEAM, PANEL = [0.3, 0.31, 0.34, 1], [0.84, 0.85, 0.87, 1]
WOOFER_X = -0.054                      # woofer chamber centre, front-back (moved back with the back wall, chest +2 cm)
WOOFER_Z = 0.039                       # woofer centre height on the lower back (torso_pack fit check, 2026-10-25)


def profile(rings, z):
    zs, ws, ds = zip(*rings)
    return float(np.interp(z, zs, ws)), float(np.interp(z, zs, ds))


def quat_y(a):
    return [math.cos(a / 2), 0, math.sin(a / 2), 0]


def quat_x(a):
    return [math.cos(a / 2), math.sin(a / 2), 0, 0]


def details(spec, rings):
    torso = spec.body("torso")
    # the torso's fixed weight is ToddlerBot's (scaled), balance point included; theirs sat 4.6 mm to his right, ours
    # sits on the centre line (woofer, Jetson, battery all on it: torso_pack fit check), so centre it until the parts
    # pass replaces that weight with our own parts (Immy saw him lean right, 2026-10-25)
    torso.ipos = [torso.ipos[0], 0, torso.ipos[2]]
    # waist gears: ToddlerBot models +-1.7 deg of play (backlash) in the waist roll gears; standing, his chest flopped to
    # one end of it after any nudge and stayed leaning 1.8 deg (Immy: "he has a lean again ... to the other side").
    # V2 uses split anti-backlash gears (two gear halves sprung against each other fill the gap between teeth); the
    # +-0.25 deg left over is an estimate for the servo's own gears (a 1 N nudge then leaves 0.26 deg)
    spec.tendon("waist_roll_coupling").range = np.radians([-0.25, 0.25])
    add = lambda **kw: torso.add_geom(density=0, contype=0, conaffinity=0, **kw)
    box = mujoco.mjtGeom.mjGEOM_BOX
    # front door, lower chest: follows the chest's slope
    z0, z1 = 0.02, 0.075
    d0, d1 = profile(rings, z0)[1], profile(rings, z1)[1]
    tilt = math.atan((d1 - d0) / (z1 - z0))
    zc, dc = (z0 + z1) / 2, (d0 + d1) / 2
    x = T.CX + dc
    add(type=box, size=[0.0006, 0.0395, 0.029], pos=[x + 0.0003, 0, zc], quat=quat_y(tilt), rgba=SEAM)
    add(type=box, size=[0.0009, 0.0375, 0.027], pos=[x + 0.0009, 0, zc], quat=quat_y(tilt), rgba=PANEL)
    for y in (-0.03, 0.03):                                       # two finger notches to pull the door open
        add(type=box, size=[0.001, 0.005, 0.0015], pos=[x + 0.0016, y, z1 - 0.004], quat=quat_y(tilt), rgba=SEAM)
    add(type=box, size=[0.001, 0.0045, 0.0016], pos=[x + 0.0016, 0.018, z0 + 0.008], quat=quat_y(tilt), rgba=T.GLASS)
    add(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0035, 0.001, 0], pos=[x + 0.0016, -0.018, z0 + 0.008],
        quat=[math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0], rgba=T.ACCENT)     # power button, lit ring
    # woofer grille, lower back, square to the back's slope (the woofer sits right behind it, torso_pack fit check)
    zw = WOOFER_Z
    slope = (profile(rings, zw + 0.005)[1] - profile(rings, zw - 0.005)[1]) / 0.01
    xw = T.CX - profile(rings, zw)[1]
    q = quat_y(-math.pi / 2 - math.atan(slope))
    add(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.025, 0.0006, 0], pos=[xw - 0.0002, 0, zw], quat=q, rgba=SEAM)
    add(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0225, 0.0009, 0], pos=[xw - 0.0006, 0, zw], quat=q, rgba=T.GLASS)
    kg = parts.PARTS["woofer, 2\" long-throw, in its sealed chamber"][0] / 1000
    w = torso.add_body(name="woofer_weight", pos=[WOOFER_X, 0, zw])        # torso has a fixed weight, so the woofer
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=kg, group=4, contype=0, conaffinity=0)   # rides on a ball
    # exhaust grille, upper back
    for i in range(6):
        z = 0.088 + i * 0.006
        add(type=box, size=[0.0012, 0.03, 0.0013], pos=[T.CX - profile(rings, z)[1] - 0.0002, 0, z], rgba=T.GLASS)
    # underside vent at the back, blowing straight down (Immy, 2026-10-25): slots in the ledge where the torso's
    # bottom overhangs the waist, between the back edge and the waist ring
    x_back, x_ring = T.CX - rings[0][2], T.CX - 0.022
    for i in range(7):
        add(type=box, size=[(x_ring - x_back) / 2 - 0.004, 0.0013, 0.0004], pos=[(x_back + x_ring) / 2, -0.03 + i * 0.01, -0.0004],
            rgba=T.GLASS)
    # intake slots, low on both sides
    for s in (-1, 1):
        zc = 0.03
        w = profile(rings, zc)[0]
        slope = (profile(rings, zc + 0.01)[0] - profile(rings, zc - 0.01)[0]) / 0.02
        for i in range(4):
            add(type=box, size=[0.0013, 0.001, 0.009], pos=[T.CX - 0.009 + i * 0.006, s * (w + 0.0003), zc],
                quat=quat_x(-s * math.atan(slope)), rgba=T.GLASS)


def main():
    scene, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    label, spec = T.build(scene, "B")
    details(spec, T.OPTIONS["B"][2])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 600, 900
    path = os.path.join(out, "v2_torsoB.xml")
    open(path, "w").write(spec.to_xml())
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 900, 600)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    views = ((180, "front", 1.05, 0.31), (0, "back", 1.05, 0.31), (35, "back, angled", 1.05, 0.31),
             (90, "side (intake slots)", 1.05, 0.31), (20, "back close-up", 0.42, 0.39))
    tiles = []
    for az, name, dist, z in views:
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [-0.028, 0, z], dist, az, -6
        r.update_scene(d, cam, opt)
        img = Image.fromarray(r.render())
        ImageDraw.Draw(img).text((300, 870), name, fill=(240, 240, 240), font=ImageFont.load_default(size=28), anchor="mm")
        tiles.append(img)
    r.close()
    sheet = Image.new("RGB", (600 * len(tiles), 960), (245, 245, 245))
    for i, t in enumerate(tiles):
        sheet.paste(t, (600 * i, 0))
    ImageDraw.Draw(sheet).text((300 * len(tiles), 930), "V2 torso B: front door (charge port, lit power "
                               "button), exhaust grille, side intakes", fill=(20, 20, 20),
                               font=ImageFont.load_default(size=28), anchor="mm")
    sheet.save(os.path.join(out, "v2-torsoB-views.png"))
    print("wrote", path, "and", os.path.join(out, "v2-torsoB-views.png"))


if __name__ == "__main__":
    main()
