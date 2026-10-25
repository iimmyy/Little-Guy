#!/usr/bin/env python3
"""Close-ups of the V2 hand, palm side on: open, fist and pointing, each made by pulling the finger wires (4 s of
simulation each) on the single-arm model from arm_v2.py.

usage: hand_pics.py <arm_v2_left.xml> <out.png> [pull in newtons, default 10]
"""
import sys

import mujoco
from PIL import Image, ImageDraw, ImageFont

POSES = {"open": (), "fist": ("index", "middle", "ringpinky", "thumb_curl", "thumb_swing"),
         "pointing": ("middle", "ringpinky", "thumb_curl", "thumb_swing")}


def main():
    path, out = sys.argv[1], sys.argv[2]
    pull = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 700, 600)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hand")
    tiles = []
    for label, wires in POSES.items():
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = 0
        for w in wires:
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{w}_pull")] = -pull
        for _ in range(4000):
            mujoco.mj_step(m, d)
        cam.lookat[:] = d.xpos[hand] - [0, 0, 0.035]
        cam.distance, cam.azimuth, cam.elevation = 0.17, 90, 0    # straight at the palm
        r.update_scene(d, cam, opt)
        img = Image.fromarray(r.render())
        ImageDraw.Draw(img).text((300, 670), label, fill=(240, 240, 240), font=ImageFont.load_default(size=30), anchor="mm")
        tiles.append(img)
    sheet = Image.new("RGB", (600 * len(tiles), 760), (245, 245, 245))
    for i, t in enumerate(tiles):
        sheet.paste(t, (600 * i, 0))
    ImageDraw.Draw(sheet).text((300 * len(tiles), 730), f"V2 left hand, palm side: wires pulled at {pull:g} N "
                               "(orange = the finger wires)", fill=(20, 20, 20), font=ImageFont.load_default(size=26), anchor="mm")
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
