#!/usr/bin/env python3
"""First head shapes for Little Guy V2, to look at and pick from (not final parts).

All four have a flat screen as the face (showing a ^_^ face), two small cameras and a 4-microphone array, sized for a
~60 cm robot (head about 12 cm, screen about 3.5 inches). Built from MuJoCo's simple shapes so the chosen one can go
straight onto his simulated body.

  A  screen box    rounded-off box, screen fills the front, cameras in the top corners of the frame, mics on top
  B  disc          round puck like electric Atlas, round screen, cameras in the ring at 10 and 2 o'clock, mics on the rim
  C  visor         helmet with a dark glass visor as the screen; cameras sit behind the glass, mics in side pods
  D  brow          Alpha-style box with a brow strip holding the cameras, ear pods holding the mics
  E  wall-e eyes   two binocular eye pods, each a round screen with a camera in it (the eyes really see); the pods
                   tilt on their own, like WALL-E's sad and happy eyes (two extra small servos)
  F  wall-e sad    the same head with the pods tilted, to show what that tilt does
  G  binocular     WALL-E's binocular housing as one piece, one wide screen across both eyes
  H  soft slab      one wide screen in a rounded slab (squircle front), cameras in the top corners, mics on top
  I  tapered        same front, but the head narrows and rounds toward the back, like a helmet seen from the side
  J  brows          the soft slab with two brow plates on top that tilt (WALL-E's sad/happy without two eyes)
  K  pebble         softer, rounder outline, closer to a pebble or EVE

usage: heads.py <out folder> [letters, e.g. EFG]   writes heads.xml, the face pictures and heads.png (side by side)
"""
import os, sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SHELL, BEZEL, GLASS, LENS, MIC, ACCENT = "0.88 0.89 0.9 1", "0.12 0.12 0.14 1", "0.03 0.03 0.04 1", \
    "0.1 0.2 0.45 1", "0.25 0.25 0.27 1", "0.35 0.75 0.95 1"


def face_png(path, text="^_^", size=512, wide=1):
    img = Image.new("RGB", (size * wide, size), (6, 6, 8))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=size // 3 if len(text) > 1 else int(size * 0.8))
    draw.text((size * wide / 2, size / 2 if len(text) > 1 else size * 0.62), text, fill=(90, 200, 245), font=font, anchor="mm")
    img.transpose(Image.FLIP_TOP_BOTTOM).save(path)       # MuJoCo reads textures bottom row first


def screen(name, x, z, w, h, y, material="face"):
    """a flat screen facing the viewer (-y), w x h metres, showing a face picture"""
    vert = f"{-w/2} {-0.001} {-h/2}  {w/2} {-0.001} {-h/2}  {w/2} {-0.001} {h/2}  {-w/2} {-0.001} {h/2}"
    return (f'<mesh name="{name}" inertia="shell" vertex="{vert}" texcoord="0 0 1 0 1 1 0 1" face="0 1 2 0 2 3"/>',
            f'<geom type="mesh" mesh="{name}" material="{material}" pos="{x} {y} {z}" contype="0" conaffinity="0"/>')


def dot(x, y, z, r, rgba, axis="0 1 0"):
    return f'<geom type="cylinder" size="{r} 0.002" pos="{x} {y} {z}" zaxis="{axis}" rgba="{rgba}"/>'


def head_a(x):
    w, h, d = 0.06, 0.045, 0.04
    mesh, scr = screen("scrA", x, 0, 0.088, 0.062, -d - 0.004)
    g = [f'<geom type="box" size="{w} {d} {h}" pos="{x} 0 0" rgba="{SHELL}"/>',
         f'<geom type="box" size="{w - 0.004} 0.002 {h - 0.004}" pos="{x} {-d - 0.002} 0" rgba="{BEZEL}"/>', scr,
         dot(x - 0.047, -d - 0.004, 0.035, 0.003, LENS), dot(x + 0.047, -d - 0.004, 0.035, 0.003, LENS)]
    g += [dot(x + dx, dy, h + 0.001, 0.0018, MIC, "0 0 1") for dx, dy in ((-0.03, -0.02), (0.03, -0.02), (-0.03, 0.02), (0.03, 0.02))]
    for sx in (-1, 1):                                     # soften the side edges
        g.append(f'<geom type="cylinder" size="{d} {h}" pos="{x + sx * w} 0 0" rgba="{SHELL}"/>')
    return mesh, g


def head_b(x):
    r, d = 0.062, 0.035
    mesh, scr = screen("scrB", x, 0, 0.07, 0.07, -d - 0.006)
    g = [f'<geom type="cylinder" size="{r} {d}" pos="{x} 0 0" zaxis="0 1 0" rgba="{SHELL}"/>',
         f'<geom type="cylinder" size="{r - 0.004} 0.002" pos="{x} {-d - 0.002} 0" zaxis="0 1 0" rgba="{ACCENT}"/>',
         f'<geom type="cylinder" size="{r - 0.011} 0.002" pos="{x} {-d - 0.004} 0" zaxis="0 1 0" rgba="{GLASS}"/>', scr]
    for a in (np.radians(150), np.radians(30)):          # cameras at 10 and 2 o'clock, in the ring
        g.append(dot(x + (r - 0.0075) * np.cos(a), -d - 0.005, (r - 0.0075) * np.sin(a), 0.0028, LENS))
    for a in (0, 90, 180, 270):                            # mics around the rim
        a = np.radians(a)
        g.append(dot(x + r * np.cos(a), 0, r * np.sin(a), 0.0018, MIC, f"{np.cos(a):.3f} 0 {np.sin(a):.3f}"))
    return mesh, g


def head_c(x):
    mesh, scr = screen("scrC", x, -0.004, 0.08, 0.05, -0.066)
    g = [f'<geom type="ellipsoid" size="0.062 0.06 0.064" pos="{x} 0 0" rgba="{SHELL}"/>',
         f'<geom type="ellipsoid" size="0.052 0.018 0.04" pos="{x} -0.046 -0.004" rgba="{GLASS}"/>', scr]
    for sx in (-1, 1):                                     # side pods with the mics
        g.append(f'<geom type="cylinder" size="0.02 0.008" pos="{x + sx * 0.062} 0 0" zaxis="1 0 0" rgba="{BEZEL}"/>')
        g += [dot(x + sx * 0.0705, dy, dz, 0.0016, MIC, "1 0 0") for dy, dz in ((-0.007, 0), (0.007, 0))]
    return mesh, g


def head_d(x):
    w, h, d = 0.055, 0.05, 0.045
    mesh, scr = screen("scrD", x, -0.008, 0.08, 0.055, -d - 0.004)
    g = [f'<geom type="box" size="{w} {d} {h}" pos="{x} 0 0" rgba="{SHELL}"/>',
         f'<geom type="box" size="{w - 0.006} 0.002 0.034" pos="{x} {-d - 0.002} -0.008" rgba="{BEZEL}"/>', scr,
         f'<geom type="box" size="{w - 0.006} 0.006 0.006" pos="{x} {-d - 0.004} 0.037" rgba="{BEZEL}"/>',
         dot(x - 0.03, -d - 0.0102, 0.037, 0.0035, LENS), dot(x + 0.03, -d - 0.0102, 0.037, 0.0035, LENS)]
    for sx in (-1, 1):                                     # ear pods with the mics
        g.append(f'<geom type="cylinder" size="0.022 0.01" pos="{x + sx * (w + 0.01)} 0 0" zaxis="1 0 0" rgba="{ACCENT}"/>')
        g += [dot(x + sx * (w + 0.0205), dy, 0, 0.0016, MIC, "1 0 0") for dy in (-0.008, 0.008)]
    return mesh, g


def eye_pod(name, x, z, tilt, out):
    """one WALL-E eye: a short tube, a round screen showing the eye, a camera under the screen, two mics on top;
    tilt (degrees) turns the pod about its viewing axis, which is what makes WALL-E look sad or happy"""
    r, d = 0.03, 0.03
    mesh, scr = screen(name, 0, 0.002, 0.036, 0.036, -d - 0.006, "eye")   # fits inside the round glass
    g = [f'<body pos="{x} 0 {z}" euler="0 {tilt} 0">',
         f'<geom type="cylinder" size="{r} {d}" zaxis="0 1 0" rgba="{SHELL}"/>',
         f'<geom type="cylinder" size="{r + 0.003} 0.004" pos="0 {-d} 0" zaxis="0 1 0" rgba="{SHELL}"/>',
         f'<geom type="cylinder" size="{r - 0.003} 0.002" pos="0 {-d - 0.004} 0" zaxis="0 1 0" rgba="{GLASS}"/>', scr,
         dot(0, -d - 0.0062, -0.022, 0.003, LENS),
         dot(-0.008 * out, 0, r, 0.0016, MIC, "0 0 1"), dot(0.008 * out, 0, r, 0.0016, MIC, "0 0 1"), '</body>']
    return mesh, g


def head_e(x, tilt=0):
    m1, g1 = eye_pod(f"eyeL{x}", x - 0.036, 0, tilt, 1)
    m2, g2 = eye_pod(f"eyeR{x}", x + 0.036, 0, -tilt, -1)
    bridge = [f'<geom type="box" size="0.012 0.018 0.01" pos="{x} 0.005 -0.018" rgba="{BEZEL}"/>',
              f'<geom type="box" size="0.008 0.008 0.02" pos="{x} 0.005 -0.045" rgba="{BEZEL}"/>']
    return m1 + "\n" + m2, g1 + g2 + bridge


def head_f(x):
    return head_e(x, tilt=18)


def head_g(x):
    r, d, w = 0.034, 0.035, 0.04
    mesh, scr = screen("scrG", x, 0, 0.096, 0.048, -d - 0.006, "wide")
    g = [f'<geom type="cylinder" size="{r} {d}" pos="{x + sx * w} 0 0" zaxis="0 1 0" rgba="{SHELL}"/>' for sx in (-1, 1)]
    g += [f'<geom type="box" size="{w} {d} {r}" pos="{x} 0 0" rgba="{SHELL}"/>',
          f'<geom type="box" size="{w + 0.012} 0.002 {r - 0.006}" pos="{x} {-d - 0.002} 0" rgba="{GLASS}"/>', scr,
          dot(x - w - 0.022, -d - 0.004, 0.0, 0.003, LENS), dot(x + w + 0.022, -d - 0.004, 0.0, 0.003, LENS)]
    g += [dot(x + dx, 0, r, 0.0016, MIC, "0 0 1") for dx in (-0.05, -0.017, 0.017, 0.05)]
    g.append(f'<geom type="box" size="0.012 0.018 0.01" pos="{x} 0.005 {-r - 0.008}" rgba="{BEZEL}"/>')
    return mesh, g


def squircle(name, w, h, d, n=4.0, taper=1.0, seg=64):
    """a convex shell: front outline |x/w|^n + |z/h|^n = 1 (n=2 ellipse, bigger n = squarer), depth d along y; the
    back outline is the front scaled by taper. MuJoCo wraps a mesh made of points in its tightest convex skin."""
    pts = []
    for y, k in ((-d / 2, 1.0), (d / 2, taper)):
        for t in np.linspace(0, 2 * np.pi, seg, endpoint=False):
            c, si = np.cos(t), np.sin(t)
            pts += [k * w * np.sign(c) * abs(c) ** (2 / n), y, k * h * np.sign(si) * abs(si) ** (2 / n)]
    return f'<mesh name="{name}" vertex="{" ".join(f"{v:.5f}" for v in pts)}"/>'


def slab(x0, key, n=4.0, taper=1.0, brows=0.0, w=0.068, h=0.042, d=0.07, rake=0.0):
    x = 0                                                 # built around its own centre, then placed (and raked back) at x0
    meshes = [squircle(f"shell{key}", w, h, d, n, taper), squircle(f"glass{key}", w - 0.007, h - 0.007, 0.004, n)]
    m, scr = screen(f"scr{key}", x, -0.002, 0.096, 0.048, -d / 2 - 0.005, "wide")
    meshes.append(m)
    g = [f'<geom type="mesh" mesh="shell{key}" pos="{x} 0 0" rgba="{SHELL}"/>',
         f'<geom type="mesh" mesh="glass{key}" pos="{x} {-d / 2 - 0.001} 0" rgba="{GLASS}"/>', scr]
    cx, cz = (w - 0.012) * (0.92 if n >= 4 else 0.8), h - 0.012
    g += [dot(x + sx * cx, -d / 2 - 0.0035, cz * (0.95 if n >= 4 else 0.75), 0.0028, LENS) for sx in (-1, 1)]
    g += [dot(x + dx, 0, h * (1 if taper == 1 else 0.97), 0.0016, MIC, "0 0 1") for dx in (-0.04, -0.013, 0.013, 0.04)]
    if brows:
        for sx in (-1, 1):                                 # brow plates, each on its own small servo
            g.append(f'<body pos="{x + sx * 0.027} {-d / 2 + 0.006} {h + 0.006}" euler="0 {-sx * brows} 0">'
                     f'<geom type="box" size="0.022 0.008 0.004" rgba="{ACCENT}"/></body>')
    g = [f'<body pos="{x0} 0 0" euler="{rake} 0 0">'] + g + ['</body>',
         f'<geom type="box" size="0.012 0.018 0.01" pos="{x0} 0.005 {-h - 0.006}" rgba="{BEZEL}"/>']
    return "\n".join(meshes), g


HEADS = {"A": ("A  screen box", head_a), "B": ("B  disc", head_b), "C": ("C  visor", head_c), "D": ("D  brow", head_d),
         "E": ("E  wall-e eyes", head_e), "F": ("F  wall-e, eyes tilted", head_f), "G": ("G  binocular, one screen", head_g),
         "H": ("H  soft slab", lambda x: slab(x, "H")), "I": ("I  tapered", lambda x: slab(x, "I", 3.0, 0.7)),
         "J": ("J  brows (tilted)", lambda x: slab(x, "J", brows=14)),
         "K": ("K  pebble", lambda x: slab(x, "K", 2.6, 0.85, h=0.046)),
         "1": ("I1  tapered (as before)", lambda x: slab(x, "I1", 3.0, 0.7)),
         "2": ("I2  longer, sleeker taper", lambda x: slab(x, "I2", 3.0, 0.55, d=0.085)),
         "3": ("I3  face raked back", lambda x: slab(x, "I3", 3.0, 0.6, d=0.08, rake=-12)),
         "4": ("I4  tapered + brows", lambda x: slab(x, "I4", 3.0, 0.7, brows=-14))}


def main():
    out = sys.argv[1]
    heads = [HEADS[k] for k in (sys.argv[2] if len(sys.argv) > 2 else "ABCD")]
    os.makedirs(out, exist_ok=True)
    face_png(os.path.join(out, "face.png"))
    face_png(os.path.join(out, "eye.png"), "^")
    face_png(os.path.join(out, "wide.png"), "^_^", wide=2)
    meshes, geoms = [], []
    for i, (_, make) in enumerate(heads):
        mesh, g = make(i * 0.4)
        meshes.append(mesh)
        geoms += g
        geoms.append(f'<geom type="capsule" size="0.012 0.03" pos="{i * 0.4} 0 -0.09" rgba="{BEZEL}"/>')   # neck stub
    xml = f"""<mujoco model="little guy heads">
  <visual><global offwidth="900" offheight="900"/><quality shadowsize="4096"/></visual>
  <asset>
    <texture name="face" type="2d" file="face.png"/>
    <material name="face" texture="face" emission="0.6"/>
    <texture name="eye" type="2d" file="eye.png"/>
    <material name="eye" texture="eye" emission="0.6"/>
    <texture name="wide" type="2d" file="wide.png"/>
    <material name="wide" texture="wide" emission="0.6"/>
    {chr(10).join(meshes)}
  </asset>
  <worldbody>
    <light pos="0.4 -1.2 1.0" dir="-0.2 1 -0.8" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.6 -0.6 0.8" dir="0.5 0.5 -0.6" diffuse="0.35 0.35 0.35"/>
    {chr(10).join(geoms)}
  </worldbody>
</mujoco>"""
    path = os.path.join(out, "heads.xml")
    open(path, "w").write(xml)
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 600, 600)
    cam = mujoco.MjvCamera()
    tiles = []
    for i, (label, _) in enumerate(heads):
        row = []
        for az, el in ((90, -5), (125, -18)):           # straight on, then from the side and a bit above
            cam.lookat[:] = [i * 0.4, 0, -0.01]
            cam.distance, cam.azimuth, cam.elevation = 0.36, az, el
            r.update_scene(d, cam)
            row.append(Image.fromarray(r.render()))
        tile = Image.new("RGB", (600, 1260), (245, 245, 245))
        tile.paste(row[0], (0, 0))
        tile.paste(row[1], (0, 600))
        ImageDraw.Draw(tile).text((300, 1230), label, fill=(20, 20, 20), font=ImageFont.load_default(size=34), anchor="mm")
        tiles.append(tile)
    sheet = Image.new("RGB", (600 * len(tiles), 1260))
    for i, t in enumerate(tiles):
        sheet.paste(t, (600 * i, 0))
    sheet.save(os.path.join(out, "heads.png"))
    print("wrote", os.path.join(out, "heads.png"))


if __name__ == "__main__":
    main()
