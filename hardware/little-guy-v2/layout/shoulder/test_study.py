"""Analytical checks for the geometric and static-load screening methods."""
import unittest

import numpy as np

import study


class ShoulderStudyTests(unittest.TestCase):
    def test_rotations_preserve_length_and_parent_axis(self):
        for angle in (-30, 0, 45, 90):
            r = study.rotation("y", angle)
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
            np.testing.assert_allclose(r @ [0, 1, 0], [0, 1, 0])
            self.assertAlmostEqual(np.linalg.det(r), 1)

    def test_sat_touch_overlap_and_rotated_separation(self):
        a = study.box("a", [0, 0, 0], [2, 2, 2])
        self.assertFalse(study.intersects(a, study.box("touch", [2, 0, 0], [2, 2, 2])))
        self.assertTrue(study.intersects(a, study.box("overlap", [1.9, 0, 0], [2, 2, 2])))
        # Parallel thin strips whose axis-aligned bounding boxes overlap,
        # but which are separated along their rotated local Z direction.
        r = study.rotation("y", 45)
        thin = study.box("thin", [0, 0, 0], [10, 1, .2], r)
        shifted = study.box("shifted", r @ [0, 0, .3], [10, 1, .2], r)
        self.assertFalse(study.intersects(thin, shifted))
        self.assertTrue(study.intersects(thin, shifted, .06))

    def test_child_rotation_does_not_move_its_motor_case(self):
        arm0, point0, axis0 = study.pose(35, 0)
        arm90, point90, axis90 = study.pose(35, 90)
        np.testing.assert_allclose(arm0[0]["center"], arm90[0]["center"])
        np.testing.assert_allclose(point0, point90)
        np.testing.assert_allclose(axis0, axis90)
        self.assertFalse(np.allclose(arm0[-1]["center"], arm90[-1]["center"]))

    def test_right_angle_positions_and_gravity_moments(self):
        moving, p, axis = study.pose(0, 0)
        self.assertAlmostEqual(sum(x["mass_kg"] for x in moving), .345)
        t = study.gravity_torques(moving, p, axis)
        # 55 g at 3 mm + other 290 g at 38 mm, divided by 10 mm/cm.
        self.assertAlmostEqual(t["parent_roll_signed_kgf_cm"], 1.1185)
        self.assertAlmostEqual(t["child_pitch_signed_kgf_cm"], 0)
        moving, p, axis = study.pose(90, 0)
        np.testing.assert_allclose(moving[0]["center"], [95.5, 50, 175.5], atol=1e-10)
        t = study.gravity_torques(moving, p, axis)
        # Mass-weighted vertical distances in the resting pose become X arms.
        self.assertAlmostEqual(t["parent_roll_signed_kgf_cm"], 4.07625)
        self.assertAlmostEqual(t["child_pitch_signed_kgf_cm"], 0, places=10)
        moving, p, axis = study.pose(0, 90)
        t = study.gravity_torques(moving, p, axis)
        self.assertAlmostEqual(t["child_pitch_signed_kgf_cm"], -4.0075)
        self.assertAlmostEqual(t["parent_roll_signed_kgf_cm"], 1.1185)

    def test_spacing_candidate_changes_gaps_and_rest_load(self):
        moving, point, axis = study.pose(0, 0, candidate=True)
        np.testing.assert_allclose(point, [86, 51, 172.5])
        motor, upper = moving[:2]
        self.assertAlmostEqual((upper["center"][0]-upper["size"][0]/2)
                               -(motor["center"][0]+motor["size"][0]/2), 6.1)
        overlaps=study.conflicts(moving, study.obstacles(), 3)
        self.assertFalse(any(not c["category"].startswith("torso reference") for c in overlaps))
        t=study.gravity_torques(moving, point, axis)
        self.assertAlmostEqual(t["parent_roll_signed_kgf_cm"], 1.2635)

    def test_torso_interior_distinguished_from_boundary(self):
        moving, _, _ = study.pose(0, 0)
        overlap = study.conflicts(moving, study.obstacles())
        motor = next(c for c in overlap if c["moving"] == "right distal shoulder")
        self.assertEqual(motor["category"], "torso reference interior")
        moving, _, _ = study.pose(90, 0)
        overlap = study.conflicts(moving, study.obstacles())
        motor = next(c for c in overlap if c["moving"] == "right distal shoulder")
        self.assertEqual(motor["category"], "torso reference boundary")


if __name__ == "__main__":
    unittest.main()
