#!/usr/bin/env python3
"""Head A (Immy's pick, 2026-10-25): the I1 head reshaped around the real screen.

The screen (EDO E555HBM2 5.55" AMOLED: glass 131.2 x 72.8 x 1.8 mm, lit area 122.9 x 69.1 mm) is a sharp rectangle;
the I1 front was too round for it (its corners poked out through the shell). Head A keeps the I1's slab look and taper
but squares the corners off like a phone (outline power 3 -> 8) and grows a little to leave a shell wall round the
panel: front 14.3 x 8.6 cm (was 13.6 x 8.4). The first 8 mm behind the face stay full size, so the panel sits in a
straight-walled pocket before the head tapers back.

Also: the face window is the lit area (rounded the same way), the face picture fills it (a pixel face from faces.py),
the cameras move up into the frame above the screen (the screen now fills the old camera spots), the four mics sit
on top in a diamond (front, back, left, right).
Ears (Immy, 2026-10-25: speakers "into the ears", built in, not sticking out): a 40 mm full-range driver behind a round
grille, in an ear cup on each side wall. The head narrows toward the back, so its side walls face ~19 deg behind
straight sideways; like a person's ears, the cups are attached at their front edge and stand out at the back, so
the speakers turn forward, toward whoever he faces (Immy: "our ears naturally stick out at a tilt"; picked 20 deg).
They are the stereo pair; the bass is the woofer in his back.
Each ear has a soft rubber (TPU) ring round its edge, 2 mm proud of the grille (Immy, 2026-10-25): in a side fall the
ear lands first, so the ring takes the hit and the grille and speaker cone never touch the floor (like the rim round
the screen).
The rubber rim and the weights are set by neck_v2.py, which runs after this.

usage: head_a.py <body scene.xml> <out scene.xml> [face name, default happy] [back width / front width, default 0.7]
"""
import os, sys

import mujoco
import numpy as np

import faces

W, H, N, DEPTH, TAPER, STRAIGHT = 0.0715, 0.043, 8.0, 0.07, 0.7, 0.008   # shell front half-width/height, squareness
TAPER_W = TAPER                                                        # back width / front width (> 1 = wider at the back)
WIN_W, WIN_H = 0.0625, 0.0355                                          # face window (covers the lit area)
LIT_W, LIT_H = 0.12288, 0.06912                                        # lit area, full size
PANEL = (0.1312, 0.0018, 0.0728)                                       # panel glass, full size (width, thickness, height)
BOARD = (0.07, 0.008, 0.04)                                            # driver board, guessed until it arrives
PCB = [0.2, 0.55, 0.3, 0]                                              # hidden inside (alpha 0); X-ray shows it
EAR_Y, EAR_Z = -0.004, 0.008                                            # ear centre: depth (front is -), height
EAR_R, EAR_DEPTH = 0.020, 0.022                                        # 40 mm driver, its depth behind the grille
MIC_SIDE = 0.050                                                       # side mics, out from the middle
SPEAKER = [0.95, 0.6, 0.1, 0]                                          # hidden driver (X-ray shows it)
EAR_AIM = float(os.environ.get("EAR_AIM", 20))                         # ears turn this far forward of straight sideways
GRILLE, SEAM = [0.2, 0.21, 0.24, 1], [0.3, 0.31, 0.34, 1]
RUBBER = [0.1, 0.1, 0.11, 1]                                           # same black TPU as the rim round the screen
BUMP = 0.002                                                           # ear ring: round, 2 mm thick, 2 mm proud of grille


def ring(w, h, n, y, seg=96):
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([w * np.sign(c) * np.abs(c) ** (2 / n), np.full(seg, y), h * np.sign(s) * np.abs(s) ** (2 / n)], 1)


def main():
    src, out = sys.argv[1], sys.argv[2]
    face = sys.argv[3] if len(sys.argv) > 3 else "happy"
    tw = float(sys.argv[4]) if len(sys.argv) > 4 else TAPER_W
    spec = mujoco.MjSpec.from_file(src)
    shell = np.concatenate([ring(W, H, N, -DEPTH / 2), ring(W, H, N, -DEPTH / 2 + STRAIGHT),
                            ring(W * tw, H * TAPER, N, DEPTH / 2)])
    spec.mesh("i1_shellI1").uservert = shell.ravel().tolist()
    spec.mesh("i1_glassI1").uservert = np.concatenate([ring(WIN_W, WIN_H, N, -0.002), ring(WIN_W, WIN_H, N, 0.002)]).ravel().tolist()
    spec.mesh("i1_scrI1").uservert = [-LIT_W / 2, -0.001, -LIT_H / 2, LIT_W / 2, -0.001, -LIT_H / 2,
                                      LIT_W / 2, -0.001, LIT_H / 2, -LIT_W / 2, -0.001, LIT_H / 2]
    tex = os.path.join(os.path.dirname(os.path.abspath(out)), f"face_{face}.png")
    faces.texture(face, tex)
    spec.texture("i1_wide").file = tex
    part = spec.body("i1_i1").bodies[0]
    back0 = -DEPTH / 2 + STRAIGHT                                       # where the taper starts

    def top(x, y):                                                      # height of the shell's top at (x, y)
        k = 1 - (1 - TAPER) * max(0.0, y - back0) / (DEPTH - STRAIGHT)
        w = W * (1 - (1 - tw) * max(0.0, y - back0) / (DEPTH - STRAIGHT))
        return H * k * max(0.0, 1 - abs(x / w) ** N) ** (1 / N)
    mics = []
    for g in part.geoms:
        if np.allclose(g.rgba, [0.1, 0.2, 0.45, 1]):                   # cameras: into the frame above the screen
            g.pos = [np.sign(g.pos[0]) * 0.040, -0.0435, (WIN_H + H) / 2]
        elif np.allclose(g.rgba, [0.25, 0.25, 0.27, 1]):               # mics
            mics.append(g)
    # mics in a diamond on top (Immy: "one in the front, two to the sides, and one in the back"): a straight row can't
    # tell a sound in front from its mirror image behind; with all four spread out, each direction reaches them at a
    # different set of times
    # front and back mics 6 mm in from the outer wall (Immy, 2026-10-25, frozen for the head mic board): at 4 mm the
    # 2 mm wall and the 4 mm mic package left no board edge (mic_diamond.py)
    spots = [(0, back0 - 0.002), (0, DEPTH / 2 - 0.006), (MIC_SIDE, 0.0), (-MIC_SIDE, 0.0)]
    for g, (x, y) in zip(mics, spots):
        g.pos = [x, y, top(x, y) + 0.0005]
    add = lambda **kw: part.add_geom(density=0, contype=0, conaffinity=0, group=2, **kw)
    box = mujoco.mjtGeom.mjGEOM_BOX
    add(type=box, size=[s / 2 for s in PANEL], pos=[0, -0.0329, 0], rgba=[0.05, 0.05, 0.06, 1])   # panel behind the glass
    add(type=box, size=[s / 2 for s in BOARD], pos=[0, -0.0275, 0.010], rgba=PCB)              # driver board behind it
    lean = np.arctan(W * (1 - tw) / (DEPTH - STRAIGHT))                 # side walls lean in toward the back (< 0: out)
    xw = W - W * (1 - tw) * (EAR_Y - back0) / (DEPTH - STRAIGHT)        # side wall at the ear's depth (outline's widest)
    shell_rgba = [g.rgba for g in part.geoms if g.meshname == "i1_shellI1"][0]
    aim = np.radians(EAR_AIM)
    R = EAR_R + 0.0025                                                  # grille + its rim
    for s in (1, -1):
        nw = np.array([s * np.cos(lean), np.sin(lean), 0])              # straight out of the side wall (a bit back)
        nf = np.array([s * np.cos(aim), -np.sin(aim), 0])               # the ear's face: out and turned forward
        up = np.array([0, 0, 1.0])
        back_w = np.cross(up, nw) * s                                   # along the wall, toward the back of the head
        back_f = np.cross(up, nf) * s                                   # across the ear's face, toward its back edge
        wall = np.array([s * xw, EAR_Y, EAR_Z])
        fc = wall - back_w * R + back_f * R + nw * 0.001                # attached at its front edge, like an ear
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        rim = [fc + R * (np.cos(a) * back_f + np.sin(a) * up) for a in t]
        base = [p - ((p - wall) @ nw + 0.002) * nw for p in rim]        # the same ring pressed onto the wall
        cup = f"ear_cup_{'left' if s > 0 else 'right'}"
        spec.add_mesh(name=cup, uservert=np.ravel(rim + base).tolist())
        add(name=cup, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=cup, rgba=shell_rgba)
        spec.add_pair(geomname1="floor", geomname2=cup)                 # it sticks out: in a side fall it can land
        x = np.cross(up, nf)                                            # a cylinder's axis = nf
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.column_stack([x, np.cross(nf, x), nf]).flatten())
        cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
        add(type=cyl, size=[R, 0.0004, 0], pos=fc + nf * 0.0002, quat=q, rgba=SEAM)
        add(type=cyl, size=[EAR_R, 0.0007, 0], pos=fc + nf * 0.0005, quat=q, rgba=GRILLE)
        add(type=cyl, size=[EAR_R, EAR_DEPTH / 2, 0], pos=fc - nf * (EAR_DEPTH / 2 + 0.0015), quat=q, rgba=SPEAKER)
        mid = fc + nf * 0.0012                                          # the ring's middle sits level with the grille
        a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 48, endpoint=False), np.linspace(0, 2 * np.pi, 12, endpoint=False))
        a, b = a.ravel()[:, None], b.ravel()[:, None]
        ring_pts = mid + (R + BUMP * np.cos(b)) * (np.cos(a) * back_f + np.sin(a) * up) + BUMP * np.sin(b) * nf
        na, nb = 48, 12                                                 # faces, so it draws as a ring, not a disc
        ia, jb = np.meshgrid(np.arange(na), np.arange(nb))
        v00, v10 = jb * na + ia, jb * na + (ia + 1) % na
        v01, v11 = (jb + 1) % nb * na + ia, (jb + 1) % nb * na + (ia + 1) % na
        tris = np.stack([np.stack([v00, v10, v11], -1), np.stack([v00, v11, v01], -1)], -2).ravel()
        spec.add_mesh(name=f"{cup}_bumper", uservert=ring_pts.ravel().tolist(), userface=tris.tolist())
        add(name=f"{cup}_bumper", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{cup}_bumper", rgba=RUBBER)
        spec.add_pair(geomname1="floor", geomname2=f"{cup}_bumper")
    for g in spec.body("head").geoms:
        if g.name == "head_collision":                                  # the head's crash box, resized to head A
            g.size = [DEPTH / 2, W * max(1, tw), H]
    spec.compile()
    open(out, "w").write(spec.to_xml())
    print(f"{out}: head A front {2 * W * 100:.1f} x {2 * H * 100:.1f} cm, face '{face}'")


if __name__ == "__main__":
    main()
