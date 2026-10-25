"""Static support-edge moment sensitivity with an explicit partial mass model."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
G = 9.80665


def shoulder_model():
    path = ROOT.parent / "shoulder/study.py"
    spec = importlib.util.spec_from_file_location("shoulder_for_stability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def moving_moment_bound(shoulder):
    """Triangle-inequality bound for both mirrored arm first moments, kg.m.

    About each parent axis, s = R_parent (a + R_child b). Parent translations
    cancel horizontally for equal mirrored arm masses. Norm preservation gives
    |s_right_xy + s_left_xy| <= 2 (|a| + |b|), for any rotations of this model.
    """
    parts, child_point, _ = shoulder.pose(0, 0, candidate=True)
    parent_point = shoulder.PARENT_AXIS_POINT
    carried_motor, arm = parts[0], parts[1:]
    arm_mass = sum(p["mass_kg"] for p in arm)
    a = (carried_motor["mass_kg"] * (carried_motor["center"] - parent_point)
         + arm_mass * (child_point - parent_point)) / 1000
    b = sum((p["mass_kg"] * (p["center"] - child_point) / 1000 for p in arm), np.zeros(3))
    return {"a_xyz_kg_m": a.tolist(), "b_xyz_kg_m": b.tolist(),
            "two_arm_horizontal_first_moment_bound_kg_m": float(2 * (np.linalg.norm(a) + np.linalg.norm(b))),
            "one_arm_modeled_mass_kg": sum(p["mass_kg"] for p in parts)}


def sampled_arm_pairs(shoulder):
    step = shoulder.P["sample_step_degrees"]
    a0, a1 = shoulder.P["parent_side_lift_degrees"]
    b0, b1 = shoulder.P["child_forward_lift_degrees"]
    angles = [(a, b) for a in range(a0, a1 + 1, step) for b in range(b0, b1 + 1, step)]
    moments = []
    for side, forward in angles:
        parts, _, _ = shoulder.pose(side, forward, candidate=True)
        moments.append(sum((p["mass_kg"] * p["center"][:2] / 1000 for p in parts), np.zeros(2)))
    right = np.array(moments)
    left = right * np.array([-1, 1])
    pair_norm = np.linalg.norm(right[:, None, :] + left[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(pair_norm), pair_norm.shape)
    return {"per_arm_samples": len(angles), "pair_count": int(pair_norm.size),
            "largest_sampled_horizontal_first_moment_kg_m": float(pair_norm[i, j]),
            "right_side_forward_deg": list(angles[i]), "left_side_forward_deg": list(angles[j]),
            "meaning": "Independent mirrored arm combinations, including unresolved collisions; not usable motion limits. Waist yaw can rotate this vector but not change its norm."}


def edge_cases(support, *, mass, fixed_moment_xy, moving_bound, height, force):
    """Lower bounds on restoring moment, and centered added mass for >=0.

    d = n dot center + half_size. The added mass is at the support center,
    hence its lever is half_size. External force acts outward at height.
    """
    if mass <= 0 or height <= 0 or force < 0 or moving_bound < 0:
        raise ValueError("invalid mass, height, force or moment bound")
    center = np.array(support["center_xy_mm"], dtype=float) / 1000
    half = np.array(support["size_xy_mm"], dtype=float) / 2000
    if not np.all(np.isfinite(half)) or np.any(half <= 0):
        raise ValueError("support spans must be positive and finite")
    rows = []
    for name, index, sign in (("right", 0, 1), ("left", 0, -1), ("front", 1, 1), ("rear", 1, -1)):
        d = sign * center[index] + half[index]
        gravity_bound = G * (mass * d - sign * fixed_moment_xy[index] - moving_bound)
        moment_bound = gravity_bound - force * height
        needed = max(0., -moment_bound / (G * half[index]))
        rows.append({"edge": name, "edge_distance_from_waist_axis_m": float(d),
                     "added_centered_mass_lever_m": float(half[index]),
                     "gravity_com_edge_distance_lower_bound_m": float(gravity_bound / (G * mass)),
                     "gravity_restoring_moment_lower_bound_nm": float(gravity_bound),
                     "net_restoring_moment_lower_bound_nm": float(moment_bound),
                     "centered_added_mass_for_nonnegative_bound_kg": float(needed)})
    return rows


def generate(parameters):
    shoulder = shoulder_model()
    body = shoulder.BODY
    bound = moving_moment_bound(shoulder)
    sampled = sampled_arm_pairs(shoulder)
    if sampled["largest_sampled_horizontal_first_moment_kg_m"] > bound["two_arm_horizontal_first_moment_bound_kg_m"] + 1e-10:
        raise AssertionError("sampled moment exceeds analytical bound")
    fixed_names = {"left proximal shoulder", "right proximal shoulder", "head yaw servo", "head pitch servo", "waist servo"}
    fixed_parts = [part for part in shoulder.COMPONENTS if part["name"] in fixed_names]
    if len(fixed_parts) != 5:
        raise ValueError("fixed servo allocation must match the eleven-servo layout")
    fixed_body_moment = sum((np.array(p["center"][:2]) * shoulder.P["servo_mass_kg"] / 1000 for p in fixed_parts), np.zeros(2))
    # Only zero horizontal sum can be omitted from the all-yaw moving bound.
    if not np.allclose(fixed_body_moment, 0, atol=1e-12):
        raise ValueError("off-axis fixed torso moment must be included in waist rotation model")
    jetson_mass = parameters["jetson_mass_kg"]
    jetson_moment = jetson_mass * np.array(body["jetson_on_stand_center"][:2]) / 1000
    total_mass = 2 * bound["one_arm_modeled_mass_kg"] + len(fixed_parts) * shoulder.P["servo_mass_kg"] + jetson_mass
    rows = []
    for support in parameters["support_candidates"]:
        cases = []
        for force in parameters["horizontal_force_examples_n"]:
            edges = edge_cases(support, mass=total_mass, fixed_moment_xy=jetson_moment,
                               moving_bound=bound["two_arm_horizontal_first_moment_bound_kg_m"],
                               height=parameters["force_application_height_above_support_m"], force=force)
            worst = max(edges, key=lambda row: (row["centered_added_mass_for_nonnegative_bound_kg"],
                                               -row["net_restoring_moment_lower_bound_nm"]))
            cases.append({"horizontal_force_n": force,
                          "centered_added_mass_for_nonnegative_bound_kg": worst["centered_added_mass_for_nonnegative_bound_kg"],
                          "governing_edge_by_added_mass": worst["edge"], "edges": edges})
        rows.append({"support": support, "force_cases": cases})
    return {"status": parameters["status"], "modeled_mass_kg": total_mass,
            "mass_ledger_kg": {"two_moving_arms_including_six_servos": 2 * bound["one_arm_modeled_mass_kg"],
                               "five_other_servos": len(fixed_parts) * shoulder.P["servo_mass_kg"], "bare_jetson_kit": jetson_mass},
            "jetson_fixed_horizontal_first_moment_kg_m": jetson_moment.tolist(),
            "analytical_arm_bound": bound, "sampled_pair_crosscheck": sampled,
            "support_sensitivity": rows, "unmodeled_masses": parameters["unmodeled_masses"],
            "scope": "Static, rigid, level contact rectangle; no sliding/structure/dynamics validation. Nonnegative mathematical moment has zero design margin at the boundary. Added mass assumed at support center; unknown masses are not automatically centered ballast.",
            "physical_stability_verified": False, "stand_selected_for_manufacture": False}


def main():
    parameters = json.loads((ROOT / "parameters.json").read_text())
    report = generate(parameters)
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"modeled_mass_kg": report["modeled_mass_kg"], "arm_bound": report["analytical_arm_bound"],
                      "sampled_pairs": report["sampled_pair_crosscheck"],
                      "added_mass_at_5n_kg": {r["support"]["name"]: next(c["centered_added_mass_for_nonnegative_bound_kg"] for c in r["force_cases"] if c["horizontal_force_n"] == 5) for r in report["support_sensitivity"]}}, indent=2))


if __name__ == "__main__":
    main()
