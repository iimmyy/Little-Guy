#!/usr/bin/env python3
"""The current leg (ToddlerBot's bones at our size) next to the new leg sketch, from the front, the outer side and
behind; the new leg twice: with its covers ("skin") and see-through ("X-ray") so the servos and rods show.

usage: leg_pics.py <current body scene.xml> <new leg.xml> <out.png>
"""
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWS = ((180, "front"), (270, "outer side"), (310, "behind"))
W, H, DIST = 420, 620, 0.62


def subtree(m, root):
    ids = {root}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p == root:
                ids.add(b)
                break
            p = m.body_parentid[p]
    return ids


def render(m, d, lookat, label):
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0                                  # hide helper points
    tiles = []
    for az, name in VIEWS:
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, DIST, az, -6
        r.update_scene(d, cam, opt)
        tiles.append(Image.fromarray(r.render()))
    row = Image.new("RGB", (W * len(VIEWS), H + 50), (245, 245, 245))
    for i, t in enumerate(tiles):
        row.paste(t, (W * i, 0))
        ImageDraw.Draw(row).text((W * i + W // 2, H - 22), VIEWS[i][1], fill=(235, 235, 235),
                                 font=ImageFont.load_default(size=24), anchor="mm")
    ImageDraw.Draw(row).text((W * len(VIEWS) // 2, H + 25), label, fill=(20, 20, 20),
                             font=ImageFont.load_default(size=28), anchor="mm")
    return row


def main():
    body_xml, leg_xml, out = sys.argv[1:4]
    spec = mujoco.MjSpec.from_file(body_xml)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    keep = subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link"))
    for g in range(m.ngeom):                              # show the left leg only
        if m.geom_bodyid[g] not in keep and m.geom_bodyid[g] != 0:
            m.geom_rgba[g, 3] = 0
    hip = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")]
    old = render(m, d, [hip[0], hip[1], hip[2] / 2 + 0.01], "now: ToddlerBot's bones at our size (left leg)")

    m2 = mujoco.MjModel.from_xml_path(leg_xml)
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    hip2 = d2.xpos[mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "left_thigh")]
    sole = 0.0
    look = [0, 0, (hip2[2] + sole) / 2 + 0.01]
    skin = render(m2, d2, look, "new, sketch 1: servos along the bones, ankle moved by two rods (skin)")
    shell = np.array([0.9, 0.91, 0.93])
    for g in range(m2.ngeom):                             # X-ray: covers see-through, servos solid
        name = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.endswith("_servo"):
            m2.geom_rgba[g, 3] = 1
        elif np.allclose(m2.geom_rgba[g, :3], shell, atol=0.01):
            m2.geom_rgba[g, 3] = 0.18
    xray = render(m2, d2, look, "new, X-ray: blue = servos (6 per leg), grey bars = the two ankle rods")

    sheet = Image.new("RGB", (old.width, old.height * 3), (245, 245, 245))
    for i, row in enumerate((old, skin, xray)):
        sheet.paste(row, (0, row.height * i))
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
