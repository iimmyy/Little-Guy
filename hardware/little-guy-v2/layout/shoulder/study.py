"""Two-axis right-shoulder screen using oriented boxes and assumed mass points.

All angles are design samples, never servo commands or calibrated joint limits.
Servo centres inherit study 01. Axis-based link references move upward 12.5 mm;
the fixed left-arm references retain their original study-01 positions.
"""
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / "parameters.json").read_text())
BODY = json.loads((ROOT.parent / "parameters.json").read_text())
COMPONENTS = json.loads((ROOT.parent / "components.json").read_text())
IDENTITY = np.eye(3)


def rotation(axis, degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    raise ValueError(axis)


def box(name, center, size, orientation=None, mass=0.0):
    return dict(name=name, center=np.array(center, dtype=float),
                size=np.array(size, dtype=float),
                orientation=IDENTITY.copy() if orientation is None else orientation,
                mass_kg=mass)


def vertices(part, padding=0):
    local = np.array(list(itertools.product((-1, 1), repeat=3))) * (part["size"] / 2 + padding)
    return local @ part["orientation"].T + part["center"]


def intersects(a, b, padding=0):
    """Separating axis theorem: face normals and all nonparallel edge crosses.

    Positive-volume overlap only; touching faces/edges are not an overlap.
    Padding expands each box in its own local frame.
    """
    aa, bb = a["orientation"].T, b["orientation"].T
    axes = list(aa) + list(bb) + [np.cross(x, y) for x in aa for y in bb]
    delta = b["center"] - a["center"]
    ha, hb = a["size"] / 2 + padding, b["size"] / 2 + padding
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis = axis / norm
        ra = np.dot(ha, np.abs(aa @ axis))
        rb = np.dot(hb, np.abs(bb @ axis))
        if abs(np.dot(delta, axis)) >= ra + rb - 1e-9:
            return False
    return True


PARENT_CENTER = np.array(BODY["proximal_shoulder_center_unsigned"], dtype=float)
CHILD_CENTER = np.array(BODY["distal_shoulder_center_unsigned"], dtype=float)
AXIS_OFFSET = np.array([0, 0, P["case_center_to_output_axis_longitudinal_mm"]])
PARENT_AXIS_POINT = PARENT_CENTER + AXIS_OFFSET
CHILD_AXIS_POINT = CHILD_CENTER + AXIS_OFFSET
LENGTH, WIDTH, AXIAL = BODY["servo_reference_envelope"]


def pose(side_lift, forward_lift, candidate=False):
    shift = P["spacing_candidate"] if candidate else {}
    dx = shift.get("arm_outboard_shift_mm", 0)
    dy = shift.get("child_and_arm_forward_shift_mm", 0)
    child_center = CHILD_CENTER + [0, dy, 0]
    child_axis_point = CHILD_AXIS_POINT + [0, dy, 0]
    parent = rotation("y", -side_lift)  # down-pointing right arm lifts outwards
    child = rotation("x", forward_lift)  # down-pointing arm lifts towards +Y
    combined = parent @ child

    def transform(point, child_moves=True):
        point = np.array(point, dtype=float)
        if child_moves:
            point = child_axis_point + child @ (point - child_axis_point)
        return PARENT_AXIS_POINT + parent @ (point - PARENT_AXIS_POINT)

    moving = [box("right distal shoulder", transform(child_center, False),
                  [AXIAL, WIDTH, LENGTH], parent, P["servo_mass_kg"])]
    x, y = BODY["arm_center_x_unsigned"]+dx, BODY["arm_center_y"]+dy
    z = child_axis_point[2]
    for link, length_key, mass_key in (
        ("upper arm", "upper_arm_length", "upper_structure_mass_kg"),
        ("forearm", "forearm_length", "forearm_structure_mass_kg"),
        ("hand", "hand_length", "hand_structure_mass_kg"),
    ):
        length = BODY[length_key]
        moving.append(box(f"right {link} reference", transform([x, y, z-length/2]),
                          [BODY["arm_structure_width"], BODY["arm_structure_depth"], length],
                          combined, P[mass_key]))
        z -= length
        if link != "hand":
            name = "elbow" if link == "upper arm" else "wrist"
            # Link endpoints are output axes; each servo case centre is lower.
            center = [x, y, z-P["case_center_to_output_axis_longitudinal_mm"]]
            moving.append(box(f"right {name}", transform(center), [AXIAL, WIDTH, LENGTH],
                              combined, P["servo_mass_kg"]))
    return moving, transform(child_axis_point, False), parent @ np.array([1., 0., 0.])


def obstacles():
    fixed = []
    moving_names = {p["name"] for p in pose(0, 0)[0]}
    for p in COMPONENTS:
        if p["name"] in moving_names:
            continue
        fixed.append(box(p["name"], p["center"], p["size"]))
    return fixed


def conflicts(moving, fixed, padding=0):
    result = []
    for a, b in itertools.product(moving, fixed):
        if intersects(a, b, padding):
            if b["name"] == "torso outer reference":
                corners = (vertices(a, padding)-b["center"]) @ b["orientation"]
                contained = np.all(np.abs(corners) <= b["size"]/2+padding+1e-9)
                category = "torso reference interior" if contained else "torso reference boundary"
            else:
                category = ("head reference" if b["name"] == "head placeholder" else
                            "reserved bay" if b["name"] == "future Jetson bay" else "fixed component/reference")
            result.append(dict(moving=a["name"], obstacle=b["name"], category=category))
    # Adjacent arm structure deliberately overlaps its actuator envelope.
    # Check the carried shoulder case against distal arm parts separately.
    for a in moving[1:]:
        if intersects(moving[0], a, padding):
            result.append(dict(moving=a["name"], obstacle=moving[0]["name"], category="moving parts"))
    return result


def gravity_torques(moving, child_point, child_axis):
    def torque(parts, point, axis):
        total = sum((np.cross((p["center"]-point)/1000,
                             [0, 0, -p["mass_kg"]*9.80665]) for p in parts), np.zeros(3))
        return float(np.dot(total, axis)/0.0980665)
    return {"parent_roll_signed_kgf_cm": torque(moving, PARENT_AXIS_POINT, [0, 1, 0]),
            "child_pitch_signed_kgf_cm": torque(moving[1:], child_point, child_axis)}


def evaluate(side_lift, forward_lift, fixed, candidate=False):
    moving, point, axis = pose(side_lift, forward_lift, candidate)
    return {"side_lift_deg":side_lift, "forward_lift_deg":forward_lift,
            "nominal_overlaps":conflicts(moving, fixed),
            "padded_overlaps":conflicts(moving, fixed, P["screening_padding_each_box_mm"]),
            "gravity_torque":gravity_torques(moving, point, axis)}


def generate(candidate=False):
    fixed = obstacles()
    step = P["sample_step_degrees"]
    parent_min, parent_max = P["parent_side_lift_degrees"]
    child_min, child_max = P["child_forward_lift_degrees"]
    samples = [evaluate(a, b, fixed, candidate)
               for a in range(parent_min, parent_max+1, step)
               for b in range(child_min, child_max+1, step)]
    named = {name:evaluate(a,b,fixed,candidate) for name,a,b in
             (("rest",0,0),("side_90",90,0),("forward_90",0,90),("combined_45",45,45))}
    pairs = {}
    padded_pairs = {}
    for sample in samples:
        for c in sample["padded_overlaps"]:
            key=(c["moving"],c["obstacle"],c["category"])
            padded_pairs[key]=padded_pairs.get(key,0)+1
        for c in sample["nominal_overlaps"]:
            key=(c["moving"],c["obstacle"],c["category"])
            pairs[key]=pairs.get(key,0)+1
    report = {"status":P["status"], "variant":"spacing candidate" if candidate else "baseline", "scope":"Right arm only; left arm stays in rest pose. Elbow and wrist locked straight. Discrete samples, no continuous collision guarantee.",
              "parent_axis_point_xyz_mm":PARENT_AXIS_POINT.tolist(),
              "child_axis_point_xyz_mm_at_rest":pose(0,0,candidate)[1].tolist(),
              "total_carried_mass_kg":sum(p["mass_kg"] for p in pose(0,0)[0]),
              "sample_count":len(samples),
              "samples_with_non_torso_nominal_overlap":sum(any(not c["category"].startswith("torso reference") for c in s["nominal_overlaps"]) for s in samples),
              "samples_with_non_torso_padded_overlap":sum(any(not c["category"].startswith("torso reference") for c in s["padded_overlaps"]) for s in samples),
              "samples_with_torso_boundary_crossing":sum(any(c["category"]=="torso reference boundary" for c in s["nominal_overlaps"]) for s in samples),
              "nominal_overlap_pair_counts":[dict(moving=k[0],obstacle=k[1],category=k[2],samples=v) for k,v in sorted(pairs.items())],
              "padded_overlap_pair_counts":[dict(moving=k[0],obstacle=k[1],category=k[2],samples=v) for k,v in sorted(padded_pairs.items())],
              "sampled_peak_absolute_gravity_torque":{}, "named_poses":named,
              "physical_fit_verified":False,"usable_joint_limits_established":False,
              "samples":samples}
    for key in named["rest"]["gravity_torque"]:
        peak=max(samples,key=lambda s:abs(s["gravity_torque"][key]))
        report["sampled_peak_absolute_gravity_torque"][key]={"absolute_kgf_cm":abs(peak["gravity_torque"][key]),
             "side_lift_deg":peak["side_lift_deg"],"forward_lift_deg":peak["forward_lift_deg"]}
    name = "candidate-results.json" if candidate else "results.json"
    (ROOT/name).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("samples","named_poses")},indent=2))


if __name__ == "__main__":
    generate()
    generate(candidate=True)
