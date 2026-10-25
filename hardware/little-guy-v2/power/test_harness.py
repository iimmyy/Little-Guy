"""Deliberate connection mistakes exercise the draft's protection boundaries."""
import json
from pathlib import Path
import unittest

from audit_harness import audit, reachable


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.model = json.loads((Path(__file__).parent / "harness.json").read_text())

    def bridge(self, name, a, b, domain):
        self.model["links"].append({"id": name, "a": a, "b": b, "domain": domain,
                                    "kind": "fault_fixture", "selected_part": None, "current_rating_a": None})

    def test_radial_paths_and_independent_protection(self):
        result = audit(self.model)
        self.assertTrue(result["topology_checks_passed"], result["errors"])
        self.assertEqual(result["load_count"], 11)
        rows = {row["link"]: row for row in result["segment_reference_sums"]}
        self.assertEqual(rows["TRUNK_P"]["rated_sum_reference_a"], 9.9)
        self.assertEqual(rows["TRUNK_R"]["stall_sum_reference_a"], 29.7)
        self.assertEqual(rows["FEED_LEFT_WRIST"]["stall_sum_reference_a"], 2.7)
        self.assertEqual(rows["USB_GROUND"]["stall_sum_reference_a"], 0)
        cuts = {cut["link"]: cut for cut in result["source_cuts"]}
        self.assertEqual(cuts["F_LEFT_WRIST"]["loads_disconnected_from_psu_positive"], ["left_wrist"])
        self.assertEqual(len(cuts["SW_MAIN"]["loads_disconnected_from_psu_positive"]), 11)
        self.assertFalse(result["electrical_design_validated"])
        self.assertFalse(result["all_parts_and_ratings_selected"])

    def test_three_wire_crossfeed_bypasses_two_leaf_fuses(self):
        self.bridge("UNWANTED_VPLUS", "LEFT_WRIST_P", "RIGHT_WRIST_P", "positive")
        result = audit(self.model)
        self.assertFalse(result["topology_checks_passed"])
        cuts = {cut["link"]: cut for cut in result["source_cuts"]}
        self.assertFalse(cuts["F_LEFT_WRIST"]["matches_expected"])
        self.assertFalse(cuts["F_RIGHT_WRIST"]["matches_expected"])

    def test_extra_ground_link_hides_broken_servo_return(self):
        self.bridge("THIN_REFERENCE_LINK", "LEFT_WRIST_R", "RIGHT_WRIST_R", "return")
        result = audit(self.model)
        self.assertFalse(result["topology_checks_passed"])
        edges = [e for e in self.model["links"] if e["domain"] == "return"]
        self.assertIn("LEFT_WRIST_R", reachable(edges, "PSU_R", skip="RETURN_LEFT_WRIST"))

    def test_external_host_ground_bond_can_bypass_open_main_return(self):
        # Hypothetical common negative/earth connection, not a statement that
        # the user's Jetson and servo supplies actually have this connection.
        self.bridge("EXTERNAL_COMMON_RETURN", "JETSON_USB_GND", "PSU_R", "return")
        result = audit(self.model)
        self.assertFalse(result["topology_checks_passed"])
        edges = [e for e in self.model["links"] if e["domain"] == "return"]
        self.assertIn("HUB_R", reachable(edges, "PSU_R", skip="TRUNK_R"))


if __name__ == "__main__":
    unittest.main()
