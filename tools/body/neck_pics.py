#!/usr/bin/env python3
"""Close-ups of the neck: skin on (how he looks) and X-ray (head and chest see-through, so the post, fork, pin and
both neck servos show).

usage: neck_pics.py <scene.xml> <out.png> [qpos.npy]
"""
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 600, 600
HIDDEN = ([0.25, 0.45, 0.8], [0.2, 0.55, 0.3])


def render(m, d, xray):
    rgba = m.geom_rgba.copy()
    head = m.body("head").id
    in_head = lambda b: b == head or (b > 0 and in_head(m.body_parentid[b]))
    if xray:
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if any(np.allclose(rgba[g], c + [0]) for c in HIDDEN):
                m.geom_rgba[g][3] = 1                          # hidden parts: servos (blue), screen board (green)
            elif (b == m.body("torso").id or in_head(b)) and not m.geom(g).name.startswith("rim") and rgba[g][3] > 0 \
                    and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                m.geom_rgba[g][3] = 0.15                       # shells see-through
    r = mujoco.Renderer(m, H, W)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    opt.sitegroup[:] = 0
    tiles = []
    for az in (180, 90, 35):
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [-0.015, 0, 0.5], 0.32, az, -8
        r.update_scene(d, cam, opt)
        tiles.append(Image.fromarray(r.render()))
    r.close()                                                   # one drawing window at a time
    m.geom_rgba[:] = rgba
    return tiles


def main():
    scene, out = sys.argv[1], sys.argv[2]
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if len(sys.argv) > 3:
        d.qpos[:] = np.load(sys.argv[3])
    mujoco.mj_forward(m, d)
    rows = [render(m, d, False), render(m, d, True)]
    sheet = Image.new("RGB", (W * 3, H * 2 + 60), (245, 245, 245))
    font = ImageFont.load_default(size=24)
    for j, row in enumerate(rows):
        for i, img in enumerate(row):
            sheet.paste(img, (W * i, H * j))
            ImageDraw.Draw(sheet).text((W * i + W / 2, H * j + H - 24), ("front", "side", "back, angled")[i] +
                                       (" (X-ray)" if j else ""), fill=(240, 240, 240), font=font, anchor="mm")
    ImageDraw.Draw(sheet).text((W * 1.5, H * 2 + 30), "neck: thick post, fork + steel pin under the head's balance point, "
                               "servos blue, screen board green, rubber rim round the screen", fill=(20, 20, 20), font=font, anchor="mm")
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
