"""Analytical moment checks, independent of any real robot or floor."""
import math
import unittest

import numpy as np

from stability import G, edge_cases, moving_moment_bound, sampled_arm_pairs, shoulder_model


class StabilityTests(unittest.TestCase):
    def test_centered_mass_and_external_force_hand_calculation(self):
        support = {"center_xy_mm": [0, 0], "size_xy_mm": [200, 200]}
        rows = edge_cases(support, mass=1, fixed_moment_xy=[0, 0], moving_bound=0, height=.5, force=2)
        for row in rows:
            self.assertAlmostEqual(row["gravity_restoring_moment_lower_bound_nm"], .1 * G)
            self.assertAlmostEqual(row["net_restoring_moment_lower_bound_nm"], .1 * G - 1)
            self.assertAlmostEqual(row["centered_added_mass_for_nonnegative_bound_kg"], (1 / G - .1) / .1)

    def test_offcenter_support_and_fixed_mass_signs(self):
        support = {"center_xy_mm": [0, -40], "size_xy_mm": [200, 240]}
        rows = {r["edge"]: r for r in edge_cases(support, mass=1, fixed_moment_xy=[0, -.02], moving_bound=0, height=.5, force=0)}
        self.assertAlmostEqual(rows["front"]["gravity_restoring_moment_lower_bound_nm"], G * (.08 + .02))
        self.assertAlmostEqual(rows["rear"]["gravity_restoring_moment_lower_bound_nm"], G * (.16 - .02))

    def test_added_mass_uses_support_center_not_waist_axis(self):
        support = {"center_xy_mm": [0, -40], "size_xy_mm": [200, 240]}
        front = next(r for r in edge_cases(support, mass=1, fixed_moment_xy=[0, 0], moving_bound=.2, height=.5, force=0) if r["edge"] == "front")
        self.assertAlmostEqual(front["centered_added_mass_for_nonnegative_bound_kg"], 1)

    def test_arm_bound_follows_independent_vector_lengths(self):
        result = moving_moment_bound(shoulder_model())
        # Rest-pose masses/offsets: .055 carried motor, .29 distal arm;
        # 3mm lateral / 51mm forward shoulder offset, 12.5mm axis offset.
        a = [.345 * .003, .345 * .051, -.055 * .0125]
        b = [.29 * .04, 0, -(.08 * .05 + .055 * .1125 + .06 * .15 + .055 * .2125 + .04 * .23)]
        self.assertTrue(np.allclose(result["a_xyz_kg_m"], a))
        self.assertTrue(np.allclose(result["b_xyz_kg_m"], b))
        self.assertAlmostEqual(result["two_arm_horizontal_first_moment_bound_kg_m"], 2 * (math.hypot(*a) + math.hypot(*b)))

    def test_all_sampled_pairs_stay_within_rotation_bound(self):
        shoulder = shoulder_model()
        self.assertLessEqual(sampled_arm_pairs(shoulder)["largest_sampled_horizontal_first_moment_kg_m"],
                             moving_moment_bound(shoulder)["two_arm_horizontal_first_moment_bound_kg_m"])


if __name__ == "__main__":
    unittest.main()
