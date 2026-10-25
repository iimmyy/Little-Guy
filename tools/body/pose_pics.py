#!/usr/bin/env python3
"""Pictures of a body in its home pose: front, three-quarter and side, with a 10 cm ruler beside him.

usage: pose_pics.py <scene.xml> <out.png> [label] [qpos.npy: pose to show instead of the standing pose; also
       writes <out>_hands.png, close-ups of both hands]
"""
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    scene, out = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    pose = np.load(sys.argv[4]) if len(sys.argv) > 4 else None
    spec = mujoco.MjSpec.from_file(scene)
    for i in range(7):                                   # ruler: alternating 10 cm blocks next to him
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.05], pos=[0, 0.25, 0.05 + 0.1 * i],
                                rgba=[0.9, 0.3, 0.2, 1] if i % 2 == 0 else [0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 700, 1000
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if pose is not None:
        d.qpos[:] = pose
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 1000, 700)
    cam = mujoco.MjvCamera()
    tiles = []
    for az, name in ((180, "front"), (145, "three-quarter"), (90, "side")):
        cam.lookat[:] = [0, 0.1, 0.32]
        cam.distance, cam.azimuth, cam.elevation = 1.35, az, -8
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        ImageDraw.Draw(img).text((350, 970), name, fill=(240, 240, 240), font=ImageFont.load_default(size=30), anchor="mm")
        tiles.append(img)
    sheet = Image.new("RGB", (700 * 3, 1060), (245, 245, 245))
    for i, t in enumerate(tiles):
        sheet.paste(t, (700 * i, 0))
    ImageDraw.Draw(sheet).text((1050, 1030), label + "   (ruler blocks = 10 cm)", fill=(20, 20, 20),
                               font=ImageFont.load_default(size=30), anchor="mm")
    sheet.save(out)
    print("wrote", out)
    if pose is not None:
        hands = []
        for side, az in (("left", 150), ("right", 210)):
            cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand")] - [0, 0, 0.02]
            cam.distance, cam.azimuth, cam.elevation = 0.2, az, -10
            r.update_scene(d, cam)
            hands.append(Image.fromarray(r.render()))
        pair = Image.new("RGB", (1400, 1000))
        pair.paste(hands[0], (0, 0))
        pair.paste(hands[1], (700, 0))
        pair.save(out.replace(".png", "_hands.png"))
        print("wrote", out.replace(".png", "_hands.png"))


if __name__ == "__main__":
    main()
