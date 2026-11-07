"""Outbox-to-controller tests with synthetic calibration and no serial devices."""
from copy import deepcopy
from dataclasses import replace
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.room import guard
from tools.servo.actions import ActionDispatcher
from tools.servo.bus import BusError
from tools.servo.control import Controller, ControlError, Joint, load_joints
from tools.servo.test_servo import MemoryBus


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.wall = 1000.0
        self.mono = 10.0
        self.bus = MemoryBus()
        self.reads = []
        self.after_read = lambda: None

        def read(servo_id):
            self.reads.append(servo_id)
            sample = MemoryBus.read(self.bus, servo_id)
            self.after_read()
            return replace(sample, sampled_at=self.mono)

        self.bus.read = read
        joint = Joint("head", 1, True, 1234, 4, 1800, 2200, 100, 100, 20,
                      offset_raw=0, hardware_min_ticks=0, hardware_max_ticks=4095)
        self.controller = Controller(self.bus, {"head": joint}, clock=lambda: self.mono)
        self.dispatcher = ActionDispatcher(self.controller, wall_clock=lambda: self.wall)
        self.note = {"kind": "joint_targets", "schema_version": 2,
                     "session_id": self.dispatcher.session_id, "request_id": "test-001",
                     "created_at": 999.5, "expires_at": 1001.5,
                     "targets": {"head": {"position_ticks": 2080, "speed_raw": 50, "acceleration_raw": 10}}}

    def test_guard_default_rejects_movement(self):
        self.assertEqual(guard.act(json.dumps(self.note), 9), ("drop", "movement is disabled", 9))
        self.assertEqual((self.reads, self.bus.writes), ([], []))

    def test_legacy_missing_and_wrong_sessions_fail_before_device_access(self):
        notes = [dict(self.note, schema_version=1)]
        missing = dict(self.note)
        del missing["session_id"]
        notes.append(missing)
        notes.extend(dict(self.note, session_id=value)
                     for value in (None, True, 7, [], {}, "", "old-session", "0" * 64))
        for note in notes:
            with self.subTest(note=note), self.assertRaises(ControlError):
                self.dispatcher.dispatch(note)
        self.assertEqual((self.reads, self.bus.writes), ([], []))
        self.assertEqual(self.dispatcher.used_ids, set())

    def test_restart_rejects_consumed_and_unconsumed_old_notes(self):
        old_session = self.dispatcher.session_id
        self.dispatcher.dispatch(self.note)
        queued = dict(self.note, request_id="never-attempted")
        # New receiver lifetime, same clock and valid calibration: timestamps
        # alone would accept both old requests, including the unused queue entry.
        self.controller = Controller(self.bus, self.controller.joints, clock=lambda: self.mono)
        self.dispatcher = ActionDispatcher(self.controller, wall_clock=lambda: self.wall)
        self.assertNotEqual(self.dispatcher.session_id, old_session)
        self.reads.clear()
        self.bus.writes.clear()
        for note in (self.note, queued):
            with self.assertRaisesRegex(ControlError, "different session"):
                self.dispatcher.dispatch(note)
        self.assertEqual((self.reads, self.bus.writes), ([], []))
        # A newly produced request is accepted only with the new receiver ID.
        fresh = dict(self.note, session_id=self.dispatcher.session_id,
                     created_at=self.wall, expires_at=self.wall + 1)
        self.dispatcher.dispatch(fresh)
        self.assertEqual((len(self.reads), len(self.bus.writes)), (1, 1))
        with self.assertRaisesRegex(ControlError, "already consumed"):
            self.dispatcher.dispatch(fresh)
        self.assertEqual((len(self.reads), len(self.bus.writes)), (1, 1))

    def test_dispatcher_replacement_also_invalidates_pending_queue(self):
        replacement = ActionDispatcher(self.controller, wall_clock=lambda: self.wall)
        with self.assertRaisesRegex(ControlError, "different session"):
            replacement.dispatch(self.note)
        self.assertEqual((self.reads, self.bus.writes), ([], []))

    def test_new_process_rejects_previous_process_note(self):
        script = '''
import json, sys
from tools.servo.actions import ActionDispatcher
from tools.servo.control import ControlError
# No controller or serial device exists: session rejection must precede access.
receiver = ActionDispatcher(None)
try:
    receiver.dispatch(json.loads(sys.stdin.read()))
except ControlError as error:
    assert "different session" in str(error), str(error)
else:
    raise AssertionError("old request accepted")
print(receiver.session_id)
'''
        result = subprocess.run([sys.executable, "-c", script], input=json.dumps(self.note),
                                text=True, capture_output=True, check=True,
                                cwd=Path(__file__).resolve().parents[2], timeout=10)
        self.assertNotEqual(result.stdout.strip(), self.dispatcher.session_id)

    def test_fault_reset_retains_session_and_consumed_ids(self):
        session = self.dispatcher.session_id
        self.bus.fail_write = True
        with self.assertRaises(BusError):
            self.dispatcher.dispatch(self.note)
        self.controller.reset_fault()
        self.bus.fail_write = False
        self.assertEqual(self.dispatcher.session_id, session)
        with self.assertRaisesRegex(ControlError, "already consumed"):
            self.dispatcher.dispatch(self.note)
        with self.assertRaises(AttributeError):
            self.dispatcher.session_id = "restored"

    def test_uncertain_submission_cannot_be_replayed_after_restart(self):
        def uncertain_write(targets):
            self.bus.writes.append(dict(targets))
            raise BusError("transport failed after write attempt")

        self.bus.sync_write = uncertain_write
        with self.assertRaises(BusError):
            self.dispatcher.dispatch(self.note)
        self.controller.reset_fault()
        restarted = ActionDispatcher(self.controller, wall_clock=lambda: self.wall)
        replay = dict(self.note, created_at=self.wall, expires_at=self.wall + 1)
        with self.assertRaisesRegex(ControlError, "different session"):
            restarted.dispatch(replay)
        self.assertEqual((len(self.reads), len(self.bus.writes)), (1, 1))

    def test_injected_handler_uses_feedback_and_keeps_sent_separate(self):
        decision, why, last_say = guard.act(json.dumps(self.note), 9,
                                           motion_handler=self.dispatcher.dispatch)
        self.assertEqual((decision, last_say), ("submitted", 9))
        self.assertIn("unconfirmed", why)
        self.assertEqual(self.reads, [1])
        self.assertEqual(self.bus.writes[0][1].position_ticks, 2080)
        self.assertEqual(self.controller.feedback["head"].position_ticks, 2048)

    def test_schema_and_target_errors_fail_before_device_access(self):
        mutations = [lambda n: n.update(port="/dev/example"),
                     lambda n: n.update(schema_version=True),
                     lambda n: n.update(request_id="../x"),
                     lambda n: n.update(targets={}),
                     lambda n: n.update(targets={"unknown": n["targets"]["head"]})]
        for field, value in [("position_ticks", True), ("position_ticks", 2080.0),
                             ("position_ticks", 2500), ("speed_raw", 0),
                             ("acceleration_raw", 0), ("torque", True)]:
            mutations.append(lambda n, f=field, v=value: n["targets"]["head"].update({f: v}))
        for mutate in mutations:
            note = deepcopy(self.note)
            mutate(note)
            with self.subTest(note=note), self.assertRaises(ControlError):
                self.dispatcher.dispatch(note)
        self.assertEqual((self.reads, self.bus.writes), ([], []))

    def test_stale_future_and_invalid_lifetimes_fail_before_reads(self):
        for created, expires in [(997, 999), (1000.1, 1001), (999, 1002),
                                 (1000, 1000), (999, 1000), (float("nan"), 1001),
                                 (True, 1001), (999, float("inf")), ("999", 1001), (10**400, 1001)]:
            with self.subTest(created=created, expires=expires), self.assertRaises(ControlError):
                self.dispatcher.dispatch(dict(self.note, created_at=created, expires_at=expires))
        self.assertEqual((self.reads, self.bus.writes), ([], []))

    def test_whole_batch_structure_checked_before_reads(self):
        self.controller.joints["elbow"] = replace(self.controller.joints["head"], name="elbow", servo_id=2)
        self.note["targets"]["elbow"] = {"position_ticks": 2080, "speed_raw": 0, "acceleration_raw": 10}
        with self.assertRaises(ControlError):
            self.dispatcher.dispatch(self.note)
        self.assertEqual((self.reads, self.bus.writes), ([], []))

    def test_expiry_is_rechecked_after_reads(self):
        self.note["expires_at"] = 1000.05

        def delay():
            self.wall += .1
            self.mono += .1

        self.after_read = delay
        with self.assertRaises(ControlError):
            self.dispatcher.dispatch(self.note)
        self.assertEqual(self.bus.writes, [])

    def test_backwards_wall_clock_cannot_extend_expiry(self):
        self.note["expires_at"] = 1000.05

        def delay():
            self.wall -= .1
            self.mono += .1

        self.after_read = delay
        with self.assertRaisesRegex(ControlError, "expired during"):
            self.dispatcher.dispatch(self.note)
        self.assertEqual(self.bus.writes, [])

    def test_slow_feedback_does_not_refresh_command_creation_time(self):
        def delay():
            self.wall += .4
            self.mono += .4

        self.after_read = delay
        with self.assertRaisesRegex(ControlError, "command is stale"):
            self.dispatcher.dispatch(self.note)
        self.assertEqual(self.bus.writes, [])

    def test_expiry_between_dispatcher_check_and_controller_submit(self):
        self.note["expires_at"] = 1000.05
        times = iter([10.0, 10.04, 10.06, 10.06])
        self.controller.clock = lambda: next(times)
        with self.assertRaisesRegex(ControlError, "deadline expired"):
            self.dispatcher.dispatch(self.note)
        self.assertEqual(self.bus.writes, [])

    def test_expiry_during_controller_validation(self):
        self.note["expires_at"] = 1000.05
        times = iter([10.0, 10.01, 10.02, 10.06])
        self.controller.clock = lambda: next(times)
        with self.assertRaisesRegex(ControlError, "deadline expired"):
            self.dispatcher.dispatch(self.note)
        self.assertEqual(self.bus.writes, [])

    def test_controller_still_checks_measured_step_and_configuration(self):
        for sample in [replace(self.bus.sample, position_ticks=1900),
                       replace(self.bus.sample, offset_raw=1), replace(self.bus.sample, status=4)]:
            self.bus.sample = sample
            self.note["request_id"] += "x"
            with self.assertRaises(ControlError):
                self.dispatcher.dispatch(self.note)
        self.assertEqual(self.bus.writes, [])

    def test_duplicate_id_remains_consumed_even_with_new_timestamps(self):
        self.dispatcher.dispatch(self.note)
        self.wall += 20
        self.mono += 20
        self.note.update(created_at=self.wall, expires_at=self.wall + 1)
        with self.assertRaisesRegex(ControlError, "already consumed"):
            self.dispatcher.dispatch(self.note)
        self.assertEqual((len(self.reads), len(self.bus.writes)), (1, 1))

    def test_capacity_rejects_without_evicting_ids(self):
        self.dispatcher.max_requests = 1
        self.dispatcher.dispatch(self.note)
        self.note["request_id"] = "different"
        with self.assertRaisesRegex(ControlError, "capacity"):
            self.dispatcher.dispatch(self.note)
        self.assertEqual((len(self.reads), len(self.bus.writes)), (1, 1))

    def test_failed_read_or_write_latches_fault_and_consumes_attempt(self):
        for failing in ("fail_read", "fail_write"):
            self.setUp()
            setattr(self.bus, failing, True)
            with self.subTest(failing=failing), self.assertRaises(BusError):
                self.dispatcher.dispatch(self.note)
            self.assertIsNotNone(self.controller.fault)
            self.controller.reset_fault()
            setattr(self.bus, failing, False)
            with self.assertRaisesRegex(ControlError, "already consumed"):
                self.dispatcher.dispatch(self.note)
            self.assertEqual(self.bus.writes, [])

    def test_real_map_rejects_before_reads(self):
        path = Path(__file__).resolve().parents[2] / "hardware/little-guy-v2/joints.json"
        self.controller.joints = load_joints(path)
        name = next(iter(self.controller.joints))
        self.note["targets"] = {name: self.note["targets"]["head"]}
        with self.assertRaisesRegex(ControlError, "not been calibrated"):
            self.dispatcher.dispatch(self.note)
        self.assertEqual((self.reads, self.bus.writes), ([], []))

    def test_guard_catches_handler_failure_without_claiming_submission(self):
        self.bus.fail_read = True
        decision, why, _ = guard.act(json.dumps(self.note), 9, motion_handler=self.dispatcher.dispatch)
        self.assertEqual(decision, "failed")
        self.assertIn("delivery and motion unconfirmed", why)
        self.assertEqual(self.bus.writes, [])

    def test_uncertain_write_is_not_reported_as_prewrite_rejection(self):
        def uncertain_write(targets):
            self.bus.writes.append(dict(targets))
            raise BusError("transport failed after write attempt")

        self.bus.sync_write = uncertain_write
        decision, why, _ = guard.act(json.dumps(self.note), 9, motion_handler=self.dispatcher.dispatch)
        self.assertEqual(decision, "failed")
        self.assertIn("delivery and motion unconfirmed", why)
        self.assertEqual(len(self.bus.writes), 1)
        self.assertIsNotNone(self.controller.fault)
        with self.assertRaisesRegex(ControlError, "already consumed"):
            self.dispatcher.dispatch(self.note)

    def test_existing_guard_notes_keep_working(self):
        with tempfile.TemporaryDirectory() as temp:
            inbox = Path(temp) / "inbox"
            inbox.mkdir()
            with patch.multiple(guard, STATE=temp, STATUS=str(Path(temp) / "status.json"), INBOX=str(inbox)), \
                    patch.object(guard.time, "time", return_value=1000000000.0):
                self.assertEqual(guard.act('{"kind":"say","text":"hello"}', 0)[0], "said")
                self.assertEqual(guard.act('{"kind":"mood","valence":0,"arousal":0.5,"word":"calm"}', 0)[0], "mood")
                to = "1000000000.000000.json"
                (inbox / to).write_text('{"text":"hi"}')
                note = json.dumps({"kind": "reply", "to": to, "text": "hello"})
                self.assertEqual(guard.act(note, 0)[0], "replied")
                self.assertEqual(guard.act(note, 0)[0], "drop")


if __name__ == "__main__":
    unittest.main()
