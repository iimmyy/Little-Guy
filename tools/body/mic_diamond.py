#!/usr/bin/env python3
"""Measure the four head mics and the room for their board, from the body model (Immy, 2026-10-25: lock the current
head's mic diamond so the head mic board can be drawn).

Everything is in the head's own frame, in mm: x to his left, y to the BACK of the head (the face is at -y), z up.
  mics      the four mic spots on the outside of the head's top (head_a.py placed them on the shell, +0.5 mm)
  plane     the flat plane closest to all four (least squares): a flat board parallel to it can serve them all
  surface   the shell's real outer top, found by shooting rays down at its mesh (not head_a's formula)
  cover     over the board's outline: how far the outer surface sits above the mic plane; with the shell wall that
            is how deep the board can sit, and the spread is how much the acoustic ports' lengths differ
  clear     the board slab under the shell, checked against every other part in the head (screen panel, its driver
            board, ear speakers, cameras)

usage: mic_diamond.py <body scene.xml> <out folder>     writes mic-diamond.txt and mic-diamond.json
"""
import json, os, sys

import mujoco
import numpy as np

WALL = 2.0          # mm, printed shell wall (same as the chest)
GASKET = 0.5        # mm, foam gasket between board and shell round each port (estimate)
PCB = 1.6           # mm, board thickness
MIC = (4.0, 3.0, 1.1)   # mm, IM69D130 package (Infineon datasheet): it hangs under the board (bottom port)
EDGE = 1.0          # mm, board edge beyond the mic package


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(src)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mics = [g for g in range(m.ngeom) if np.allclose(m.geom_rgba[g], [0.25, 0.25, 0.27, 1])]
    b = m.geom_bodyid[mics[0]]
    R, o = d.xmat[b].reshape(3, 3), d.xpos[b]
    local = lambda p: R.T @ (p - o) * 1000
    P = np.array([local(d.geom_xpos[g]) for g in mics])
    names = {}
    names["front"] = P[np.argmin(P[:, 1])]
    names["back"] = P[np.argmax(P[:, 1])]
    names["left"] = P[np.argmax(P[:, 0])]
    names["right"] = P[np.argmin(P[:, 0])]
    shell = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
             and m.mesh(m.geom_dataid[g]).name == "i1_shellI1"][0]

    def surface(x, y):
        """outer top of the shell above head-frame (x, y) mm, by a ray straight down onto its mesh"""
        start = o + R @ (np.array([x, y, 200.0]) / 1000)
        dist = mujoco.mj_rayMesh(m, d, shell, start, -R[:, 2])
        return None if dist < 0 else 200.0 - dist * 1000

    # the shell's own front, back and side walls along the mic lines (rays sideways / front-back at mic height)
    def wall(p, axis, sign):
        start = o + R @ (p / 1000) + R[:, axis] * sign * 0.2
        dist = mujoco.mj_rayMesh(m, d, shell, start, -R[:, axis] * sign)
        return None if dist < 0 else (p[axis] + sign * (200.0 - dist * 1000))

    # plane through the four mics: z = a + bx x + by y
    A = np.c_[np.ones(4), P[:, 0], P[:, 1]]
    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    plane = lambda x, y: coef[0] + coef[1] * x + coef[2] * y
    resid = P[:, 2] - A @ coef
    tilt_back = np.degrees(np.arctan(-coef[2]))
    tilt_side = np.degrees(np.arctan(coef[1]))
    # board outline: a diamond whose tips reach EDGE beyond each mic package, measured along the tip direction
    reach = max(MIC[:2]) / 2 + EDGE
    tips = {k: v[:2] + np.sign(v[:2]) * reach * (np.abs(v[:2]) == np.abs(v[:2]).max()) for k, v in names.items()}
    poly = np.array([tips["front"], tips["left"], tips["back"], tips["right"]])

    def inside(x, y):
        s = None
        for i in range(4):
            a, c = poly[i], poly[(i + 1) % 4]
            cr = (c[0] - a[0]) * (y - a[1]) - (c[1] - a[1]) * (x - a[0])
            if s is None:
                s = cr >= 0
            elif (cr >= 0) != s:
                return False
        return True

    cover, missing = [], 0
    for x in np.arange(poly[:, 0].min(), poly[:, 0].max() + 0.01, 1.0):
        for y in np.arange(poly[:, 1].min(), poly[:, 1].max() + 0.01, 1.0):
            if inside(x, y):
                z = surface(x, y)
                if z is None:
                    missing += 1
                else:
                    cover.append((z - plane(x, y), x, y))
    cover = np.array(cover)
    low = cover[np.argmin(cover[:, 0])]
    # board top sits WALL + GASKET under the lowest outer surface point (measured square to the plane, near enough
    # at this tilt); board bottom and the mics hang below it
    top_off = low[0] - WALL - GASKET
    slab = (top_off - PCB - MIC[2], top_off)                     # mm, relative to the mic plane
    # walls along the mic lines, at the board's height
    walls = {"front": wall(np.array([0, names["front"][1], plane(0, names["front"][1]) + top_off - 1]), 1, -1),
             "back": wall(np.array([0, names["back"][1], plane(0, names["back"][1]) + top_off - 1]), 1, 1),
             "left": wall(np.array([names["left"][0], 0, plane(names["left"][0], 0) + top_off - 1]), 0, 1),
             "right": wall(np.array([names["right"][0], 0, plane(names["right"][0], 0) + top_off - 1]), 0, -1)}
    gaps = {}
    for k, v in names.items():
        ax = 1 if k in ("front", "back") else 0
        pkg = (MIC[0] if ax == 1 else MIC[1]) / 2             # package long side along the front-back line
        gaps[k] = abs(walls[k] - v[ax]) - pkg - WALL if walls[k] is not None else None
    # every other solid part in the head vs the board slab (box test in head frame)
    clashes = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or g in mics or g == shell:
            continue
        lo, hi = [], []
        c = local(d.geom_xpos[g])
        Rg = R.T @ d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            half = m.geom_aabb[g][3:] * 1000
            c = local(d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g][:3])
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = m.geom_size[g][:2] * 1000
            half = np.array([r, r, h])
        else:
            half = m.geom_size[g][:3] * 1000
        ext = np.abs(Rg) @ half
        lo, hi = c - ext, c + ext
        # sample the part's box top face corners and centre against the slab under the board outline
        for x in np.linspace(lo[0], hi[0], 9):
            for y in np.linspace(lo[1], hi[1], 9):
                if inside(x, y) and hi[2] > plane(x, y) + slab[0] and lo[2] < plane(x, y) + slab[1]:
                    clashes.append((m.geom(g).name or f"geom {g}", round(float(hi[2] - (plane(x, y) + slab[0])), 1)))
                    break
            else:
                continue
            break

    lines = [f"head mic diamond, {src}", "head frame, mm: x to his left, y to the back of the head (face at -y), z up", ""]
    for k, v in names.items():
        lines.append(f"  mic {k:5s} at ({v[0]:6.1f}, {v[1]:6.1f}, {v[2]:5.1f})")
    lines += ["", f"spacing: front-back {np.linalg.norm(names['front'] - names['back']):.1f} mm (along the slope), "
              f"left-right {np.linalg.norm(names['left'] - names['right']):.1f} mm",
              f"mic plane: z = {coef[0]:.2f} + {coef[1]:.4f} x + {coef[2]:.4f} y; tilts {tilt_back:.1f} deg down toward "
              f"the back, {tilt_side:.2f} deg sideways",
              "mics off that plane: " + ", ".join(f"{k} {v[2] - plane(v[0], v[1]):+.2f}" for k, v in names.items()) + " mm",
              f"board outline (diamond, tips {reach:.1f} mm past each mic centre): " +
              ", ".join(f"{k} ({t[0]:.1f}, {t[1]:.1f})" for k, t in zip(("front", "left", "back", "right"), poly)),
              f"outer surface above the mic plane over that outline: {cover[:, 0].min():+.2f} to {cover[:, 0].max():+.2f} mm "
              f"(lowest at x {low[1]:.0f}, y {low[2]:.0f}); {missing} outline points had no shell above them",
              f"board slab (shell wall {WALL} + gasket {GASKET}, board {PCB}, mics {MIC[2]} under it), relative to the "
              f"mic plane: {slab[0]:+.1f} to {slab[1]:+.1f} mm",
              "room from each mic package to the inside of the nearest side wall, at board height: " +
              ", ".join(f"{k} {v:.1f} mm" if v is not None else f"{k} no wall found" for k, v in gaps.items()),
              "parts in the way of the board slab: " + (", ".join(f"{n} (by {h} mm)" for n, h in clashes) or "none")]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "mic-diamond.txt"), "w").write(txt + "\n")
    json.dump({"frame": "head: x left, y back (face at -y), z up; mm",
               "mics": {k: [round(float(c), 2) for c in v] for k, v in names.items()},
               "plane": [round(float(c), 5) for c in coef], "residual_mm": [round(float(r), 3) for r in resid],
               "outline": {k: [round(float(c), 2) for c in t] for k, t in zip(("front", "left", "back", "right"), poly)},
               "cover_mm": [round(float(cover[:, 0].min()), 2), round(float(cover[:, 0].max()), 2)],
               "slab_mm": [round(float(s), 2) for s in slab],
               "wall_gap_mm": {k: (None if v is None else round(float(v), 2)) for k, v in gaps.items()},
               "clashes": clashes}, open(os.path.join(out, "mic-diamond.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
