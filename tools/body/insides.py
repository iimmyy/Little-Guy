#!/usr/bin/env python3
"""Put Little Guy's real insides into his chest, with their real weight (Immy, 2026-10-25: "while we build the PCB ...
you can start to wire in things that already exist, like the Jetson and all that other stuff").

Until now the chest weighed what ToddlerBot's chest weighs, scaled up (1341 g). That lump already holds THEIR computer,
battery, boards and shell (ToddlerBot BOM: Jetson Orin NX, 2000 mAh LiPo, IMU, buck converters, USB hub, power PCB,
PLA-CF shell). Adding our parts on top would count the electronics twice, so this replaces the lump:

  chest weight = printed shell + inner frame + screws + wiring        (worked out below, labelled estimate)
               + every bought part that is picked, as its own box     (maker numbers)
               + a kept-space box for every part not made or picked yet (our 4 PCBs, amp, LTE, hub; estimates)

Parts already modelled elsewhere keep their own weights and are not counted again: shoulder servos (shoulder mounts),
neck turn servo, waist servos (pelvis), woofer (100 g, its own body), handles (30 g, their own bodies).

Every part is a box in the torso's frame (x forward, y to his left, z up from the waist). A greedy search places them
one at a time in the order listed (Jetson, battery, camera, then the kept spaces), in any of 6 orientations, inside the shell (torso_pack.inside) and clear of each other,
each pulled toward where it belongs (Jetson high at the back by the exhaust grille, battery low: heat rises).

usage: insides.py <body scene.xml> <out folder>     writes v2_insides.xml, insides.txt and insides-views.png
"""
import itertools, math, os, sys

import mujoco
import numpy as np

import torso_options as T
import torso_pack as P

# ---- the printed shell and frame (no maker number exists: worked out from the shell's own shape) ----
SHELL_MM = 2.0             # printed wall; the shell's area is measured off its own mesh (shell_cm2)
PETG_CF = 1.25             # g/cm3 (Bambu PETG-CF, Welectron listing)
FRAME_G = 120              # estimate: battery cage, Jetson mount, servo brackets, neck mount, door hinge, ribs
SCREWS_G = 25              # estimate: screws, heat-set inserts, door magnets
WIRING_G = 50              # estimate: chest share of the harness (servo leads, power, USB, sensor lines)

# ---- parts: name -> (kind, full size mm (any order), grams, source, where it wants to be) ----
REAL = {
    "Jetson stack": ("jetson", (91, 61, 32), 148,
                     "Forecr DSBOARD-ORNXS carrier 91 x 61 mm 43 g (manual rev 2.0) + Orin Nano module 69.6 x 45 mm "
                     "~50 g (Farnell listing, may include packing) + active heatsink 58 x 38 x 18 mm 49 g (Connect Tech "
                     "spec) + SSD and Wi-Fi card ~6 g (estimate); stack height 32 mm (estimate)", "high_back"),
    "battery vertebra 1": ("battery", (43, 70.2, 21.6), 155,
                           "2 x Molicel P45B 21700: 21.55 x 70.15 mm, 70 g max each (Molicel datasheet v1.2) "
                           "+ holder, nickel strip, wrap ~15 g (estimate)", "low"),
    "battery vertebra 2": ("battery", (43, 70.2, 21.6), 155, "same", "low"),
    "battery vertebra 3": ("battery", (43, 70.2, 21.6), 155, "same", "low"),
    "depth camera": ("camera", (38, 38, 15), 50,
                     "Arducam ToF B0410: 38 x 38 mm (Arducam docs); 50 g is its SHIPPING weight (Welectron), so an "
                     "upper bound; depth 15 mm (estimate)", "front_high"),
}
KEPT = {   # not made or not picked yet: space and weight kept for them (first estimates, see PCB-BRIEF.md)
    "power board (ours)": ("pcb", (80, 60, 15), 40, "estimate", "low"),
    "spinal-cord board (ours)": ("pcb", (70, 50, 15), 30, "estimate", "middle"),
    "amp board (not picked)": ("pcb", (45, 35, 12), 15, "estimate", "front"),
    "LTE modem (not picked)": ("pcb", (51, 30, 6), 10, "estimate (mini PCIe size)", "high"),
    "USB hub (not picked)": ("pcb", (40, 25, 10), 10, "estimate", "middle"),
}
WOOFER = {"woofer chamber": ((-0.054, 0.0, 0.039), (0.0275, 0.030, 0.030))}   # 55 x 60 x 60 mm, already in (torso_pack)
COLOURS = {"jetson": (0.2, 0.7, 0.3, 1), "battery": (0.85, 0.2, 0.2, 1), "camera": (0.95, 0.75, 0.1, 1),
           "pcb": (0.6, 0.3, 0.8, 1), "fixed": (0.25, 0.45, 0.8, 1)}
FRONT = T.CX + T.OPTIONS["B"][2][1][2]    # front of the chest at the door height


def score(want, c):
    x, y, z = c
    return {"high_back": -z * 2 + (x + 0.06) ** 2 * 40,
            "low": z * 3 + abs(y) * 0.2,
            "front_high": -x * 2 - z,
            "middle": abs(z - 0.07) + abs(x - T.CX),
            "front": -x * 2 + z * 0.5,
            "high": -z}[want]


def search(parts, step=0.0025):
    taken = dict(P.FIXED, **WOOFER)
    placed, missing = {}, []
    order = list(parts)                     # in the order listed: real parts first, the battery right after the Jetson
    for name in order:
        kind, size, g, src, want = parts[name]
        best = None
        for perm in set(itertools.permutations(size)):
            h = np.array(perm) / 2000
            for x in np.arange(-0.10, 0.045, step):      # back wall to front wall, chest +2 cm
                for z in np.arange(0.0, 0.15, step):
                    for y in np.arange(-0.06, 0.0601, step * 2):
                        c = (x, y, z)
                        if any(overlap_(c, h, o) for o in taken.values()):
                            continue
                        if not all(P.inside(p) for p in P.corners(c, h)):
                            continue
                        if kind == "camera" and h[0] != min(h):     # it looks forward: thin side along x
                            continue
                        s = score(want, c)
                        if kind == "battery" and "Jetson stack" in placed:     # heat rises: battery below the Jetson
                            jc, jh = placed["Jetson stack"]
                            if z + h[2] > jc[2] - jh[2]:
                                continue
                        if best is None or s < best[0]:
                            best = (s, c, tuple(h))
        if best is None:
            missing.append(name)
            continue
        placed[name] = (best[1], best[2])
        taken[name] = (best[1], best[2])
    return placed, missing


def overlap_(c, h, o):
    return P.overlap((c, h), o)[0]


def shell_cm2(m):
    """outer surface of the chest shell (its mesh, sides + top + bottom), cm2"""
    i = m.mesh("torso_B").id
    v = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]]
    f = m.mesh_face[m.mesh_faceadr[i]:m.mesh_faceadr[i] + m.mesh_facenum[i]]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2 * 1e4)


def build(src, placed, parts):
    spec = mujoco.MjSpec.from_file(src)
    torso = spec.body("torso")
    m0 = spec.compile()
    before = m0.body_mass[m0.body("torso").id]
    area = shell_cm2(m0)
    shell_g = area * SHELL_MM / 10 * PETG_CF
    struct_g = shell_g + FRAME_G + SCREWS_G + WIRING_G
    torso.mass = struct_g / 1000                                   # the lump goes; the shell + frame stay
    torso.ipos = [T.CX, 0, 0.075]                                  # shell's middle (area-weighted height ~7.5 cm)
    a, b, c = 0.055, 0.07, 0.075                                   # its inertia: a thin shell about this big
    torso.inertia = [struct_g / 1000 * (b * b + c * c) / 3, struct_g / 1000 * (a * a + c * c) / 3,
                     struct_g / 1000 * (a * a + b * b) / 3]
    torso.explicitinertial = True
    for name, (cen, h) in placed.items():
        kind, _, g, _, _ = parts[name]
        body = torso.add_body(name="in_" + name.split(" (")[0].replace(" ", "_").replace("-", "_"), pos=list(cen))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(h), mass=g / 1000, rgba=list(COLOURS[kind]),
                      contype=0, conaffinity=0, group=1)
    m = spec.compile()
    return spec, m, before, struct_g, area, shell_g


def com(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    feet = (d.xpos[m.body("left_foot").id] + d.xpos[m.body("right_foot").id]) / 2
    return d.subtree_com[0] - feet


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    parts = dict(REAL, **KEPT)
    placed, missing = search(parts)
    spec, m, before, struct_g, area, shell_g = build(src, placed, parts)
    m_old = mujoco.MjModel.from_xml_path(src)
    c0, c1 = com(m_old), com(m)
    lines = [f"chest insides, {src}",
             f"old chest lump (ToddlerBot scaled): {before * 1000:.0f} g",
             f"new chest structure: shell {shell_g:.0f} g ({area:.0f} cm2 x {SHELL_MM} mm x {PETG_CF} g/cm3) + frame "
             f"{FRAME_G} + screws {SCREWS_G} + wiring {WIRING_G} = {struct_g:.0f} g (estimates)", ""]
    for group, items in (("bought, picked (maker numbers)", REAL), ("space kept, not made or picked yet", KEPT)):
        lines.append(group + ":")
        for n, (kind, size, g, srcs, want) in items.items():
            if n in placed:
                cen, h = placed[n]
                lines.append(f"  {n:28s} {g:4d} g  at ({cen[0] * 1000:5.1f}, {cen[1] * 1000:5.1f}, {cen[2] * 1000:5.1f}) mm, "
                             f"box {2000 * h[0]:.0f} x {2000 * h[1]:.0f} x {2000 * h[2]:.0f}  [{srcs}]")
            else:
                lines.append(f"  {n:28s} {g:4d} g  DOES NOT FIT anywhere in the chest")
    placed_g = sum(parts[n][2] for n in placed)
    lines += ["", f"chest now: {struct_g + placed_g:.0f} g (+ woofer, handles, servos as before)",
              f"whole robot: {m_old.body_subtreemass[0]:.3f} kg -> {m.body_subtreemass[0]:.3f} kg",
              f"balance point over the feet (x fwd, y left, height), mm: {np.round(c0 * 1000, 1)} -> {np.round(c1 * 1000, 1)}",
              f"missing: {missing or 'none'}"]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "insides.txt"), "w").write(txt + "\n")
    open(os.path.join(out, "v2_insides.xml"), "w").write(spec.to_xml())
    P.render(spec, os.path.join(out, "insides-views.png"),
             "Jetson green, battery red, depth camera yellow, kept space purple, servos + woofer blue")


if __name__ == "__main__":
    main()
