"""Screen assumed straight horizontal arms. Pure calculation; no hardware access.

Run: python3 hardware/little-guy-v2/analysis/arm_loads.py
Lengths are metres, masses kilograms, torques Nm unless labelled otherwise.
"""

import json
from pathlib import Path

G = 9.80665
KGF_CM_TO_NM = G / 100


def moments(mass, center, length=0.0):
    """Gravity torque and rotational inertia of a collinear uniform segment."""
    return mass * G * center, mass * (center**2 + length**2 / 12)


def evaluate(config, scenario):
    upper = scenario["upper_arm_m"]
    fore = scenario["forearm_m"]
    hand = scenario["hand_m"]
    reach = upper + fore + hand
    servo = config["servo_mass_kg"]
    # name, mass, center from proximal shoulder, axial length, carried by elbow
    components = [
        ("second shoulder servo", servo,
         scenario["carried_shoulder_servo_offset_m"], 0, False),
        ("upper structure", scenario["upper_structure_kg"], upper / 2, upper, False),
        ("elbow servo", servo, upper, 0, False),
        ("forearm structure", scenario["forearm_structure_kg"], upper + fore / 2, fore, True),
        ("wrist servo", servo, upper + fore, 0, True),
        ("hand structure", scenario["hand_structure_kg"], upper + fore + hand / 2, hand, True),
        ("tip payload", scenario["payload_kg"], reach, 0, True),
    ]
    alpha = config["angular_acceleration_rad_s2"]
    allowance = config["rated_torque_kgf_cm"] * config["design_allowance_fraction"]
    result = {"name": scenario["name"],
              "acceleration_case": "each row independently accelerates its joint; upstream joint fixed, downstream joints locked",
              "reach_m": reach,
              "moving_mass_kg": sum(c[1] for c in components),
              "screening_allowance_kgf_cm": allowance, "joints": {}}
    for joint, origin in (("proximal_shoulder", 0), ("elbow", upper)):
        rows = []
        for name, mass, center, length, elbow_carried in components:
            if joint == "elbow" and not elbow_carried:
                continue
            gravity, inertia = moments(mass, center - origin, length)
            rows.append({"component": name, "gravity_Nm": gravity,
                         "inertia_kg_m2": inertia})
        gravity = sum(row["gravity_Nm"] for row in rows)
        inertia = sum(row["inertia_kg_m2"] for row in rows)
        total = gravity + inertia * alpha
        remaining = allowance * KGF_CM_TO_NM - total
        radius = reach - origin
        result["joints"][joint] = {
            "gravity_kgf_cm": gravity / KGF_CM_TO_NM,
            "acceleration_kgf_cm": inertia * alpha / KGF_CM_TO_NM,
            "combined_kgf_cm": total / KGF_CM_TO_NM,
            "within_screening_allowance": remaining >= 0,
            # Signed: a negative value means the unloaded assumption already fails.
            "additional_tip_mass_margin_kg": remaining / (G * radius + alpha * radius**2),
            "components": rows,
        }
    return result


if __name__ == "__main__":
    config = json.loads(Path(__file__).with_name("arm-scenarios.json").read_text())
    print(json.dumps({"status": config["status"], "results": [
        evaluate(config, scenario) for scenario in config["scenarios"]
    ]}, indent=2))
