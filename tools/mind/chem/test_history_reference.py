"""Regression checks for persistent, independent actual-history OX references.

Run with: python -m unittest test_history_reference
"""
import copy
import json
import unittest

import engine as E


class HistoryReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = E.Chem(hour=7.)

    def setUp(self):
        self.body = copy.deepcopy(self.initial)

    def test_json_resume_after_sleep_and_wake(self):
        for awake in (True, False, True):
            self.body.step(1/60, awake=awake, light=awake)
        restored = object.__new__(E.Chem)
        restored.__dict__ = json.loads(json.dumps(vars(self.body), allow_nan=False))
        for awake in (False, True):
            for body in (self.body, restored):
                body.step(1/60, awake=awake, light=awake, threat=.4)
        self.assertEqual(vars(self.body), vars(restored))
        restored.ox_reference['c']['oxa'] += 1.
        self.assertNotEqual(restored.ox_reference['c']['oxa'], self.body.ox_reference['c']['oxa'])

    def test_reference_receives_only_sleep_and_light(self):
        changed = copy.deepcopy(self.body)
        changed.g.update(lc_drive=1.6, ox_cells=1.6, sleep_rise=12.)
        changed.amount['ox_rvlm'] = 1.4
        changed.give('lemborexant', .4)
        for awake in (True, False, True):
            self.body.step(1/60, awake=awake, light=awake)
            changed.step(1/60, awake=awake, light=awake, threat=.7, exercise=.5)
        self.assertEqual(self.body.ox_reference, changed.ox_reference)
        self.assertNotEqual(self.body.eff('ox_rvlm'), changed.eff('ox_rvlm'))
        self.assertNotIn('ox_reference', self.body.ox_reference)

    def test_legacy_readout_then_migration(self):
        for key in ('ox_reference', 'ox_reference_value', 'ox_reference_meta', '_ox_reference_role',
                    'shift', 'light_phase', 'abs_t', 'sleep_start', 'woke_abs_t'):
            del self.body.__dict__[key]
        expected = self.body.eff('a1_symp') + 3*max(0., self.body.eff('ox_rvlm')
                    - self.body.expect_ox[int(self.body.hour*12)%288] - .03)
        self.assertEqual(self.body._sympathetic(), expected)
        self.body.step(1/60, awake=True)
        self.assertEqual(self.body.ox_reference_meta['origin'], 'legacy_unknown_history')
        self.assertAlmostEqual(self.body.abs_t, 1/60, places=12)
        self.assertAlmostEqual(self.body.ox_reference_meta['history_hours'], 1/60, places=12)


if __name__ == '__main__':
    unittest.main()
