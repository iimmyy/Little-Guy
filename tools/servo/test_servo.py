"""Protocol and command-policy tests; all ports are scripted in memory."""
from dataclasses import replace
from pathlib import Path
import unittest

from tools.servo.bus import BusError, STSBus, packet_handler
from tools.servo.control import Controller, ControlError, Feedback, Joint, Target, load_joints


def response(data, servo_id=1, error=0):
    # Protocol status packet; independent fixture builder, not SDK encoder.
    body = bytes([servo_id, len(data) + 2, error]) + bytes(data)
    return b"\xff\xff" + body + bytes([255 - (sum(body) % 256)])


def registers():
    data = [0] * 68
    values = {3: 0xD2, 4: 4, 11: 255, 12: 15, 18: 4, 30: 1, 33: 0, 40: 1,
              56: 0, 57: 8, 58: 7, 59: 128, 60: 125, 61: 4,
              62: 120, 63: 31, 66: 1, 67: 32, 68: 8, 69: 17}
    for address, value in values.items():
        data[address - 3] = value
    return data


class ScriptedPort:
    def __init__(self, replies=(), fragment=3, short_write=False):
        self.replies = list(replies)
        self.fragment = fragment
        self.short_write = short_write
        self.is_using = False
        self.rx = b""
        self.writes = []
        self.budget = 1000

    def clearPort(self):
        self.rx = b""

    def writePort(self, packet):
        self.writes.append(bytes(packet))
        self.rx = self.replies.pop(0) if self.replies else b""
        return len(packet) - int(self.short_write)

    def readPort(self, length):
        if length < 0:
            raise ValueError("negative read length")
        size = min(length, self.fragment)
        data, self.rx = self.rx[:size], self.rx[size:]
        return data

    def setPacketTimeout(self, length):
        self.budget = 1000

    def isPacketTimeout(self):
        self.budget -= 1
        return self.budget < 0


class PacketTests(unittest.TestCase):
    def test_read_wire_bytes_and_direction_bits(self):
        port = ScriptedPort([response(registers())])
        sample = STSBus(packet_handler(port), clock=lambda: 10).read(1)
        self.assertEqual(port.writes, [bytes.fromhex("ff ff 01 04 02 03 44 b1")])
        self.assertEqual((sample.model_number, sample.position_ticks, sample.speed_raw,
                          sample.load_duty_raw, sample.voltage_v, sample.phase_current_raw),
                         (1234, 2048, -7, -125, 12.0, 17))
        self.assertEqual(sample.target_ticks, 2080)
        self.assertEqual(sample.sampled_at, 10)

    def test_rejects_truncated_checksum_valid_reply(self):
        # SDK slicing alone can accidentally count the checksum as the missing data byte.
        port = ScriptedPort([response(registers()[:-1])])
        with self.assertRaises(BusError):
            STSBus(packet_handler(port)).read(1)

    def test_continuous_noise_cannot_defeat_receive_deadline(self):
        class NoisyPort(ScriptedPort):
            reads = 0

            def readPort(self, length):
                self.reads += 1
                if self.reads > 20:
                    raise AssertionError("receive loop ignored deadline")
                self.budget = -1
                return b"\x55" * length

        port = NoisyPort()
        with self.assertRaises(BusError):
            STSBus(packet_handler(port)).read(1)
        self.assertEqual(port.reads, 1)
        self.assertFalse(port.is_using)

    def test_coordinate_offset_and_device_limits_are_preserved(self):
        data = registers()
        data[31 - 3], data[32 - 3] = 0x34, 0x12
        data[9 - 3], data[10 - 3] = 100, 0
        sample = STSBus(packet_handler(ScriptedPort([response(data)]))).read(1)
        self.assertEqual((sample.offset_raw, sample.hardware_min_ticks, sample.hardware_max_ticks),
                         (0x1234, 100, 4095))

    def test_corrupt_checksum_timeout_wrong_id_and_device_errors(self):
        valid = response(registers())
        scenarios = [valid[:-1] + bytes([valid[-1] ^ 1]), b"",
                     response(registers(), servo_id=2), response(registers(), error=4),
                     bytes.fromhex("ff ff 01 00 00 fe")]
        for raw in scenarios:
            with self.subTest(raw=raw.hex()):
                with self.assertRaises(BusError):
                    STSBus(packet_handler(ScriptedPort([raw]))).read(1)

    def test_two_joint_sync_write_matches_fixed_protocol_vector(self):
        port = ScriptedPort()
        bus = STSBus(packet_handler(port), motion_enabled=True)
        bus.sync_write({2: Target(2000, 60, 12), 1: Target(2080, 50, 10)})
        # ID through payload sums to843; ~(843 mod256) =0xb4.
        self.assertEqual(port.writes, [bytes.fromhex(
            "ff ff fe 14 83 29 07 01 0a 20 08 00 00 32 00 02 0c d0 07 00 00 3c 00 b4")])
        self.assertEqual(bus.packet.groupSyncWrite.data_dict, {})

    def test_batch_does_not_leak_between_calls(self):
        port = ScriptedPort()
        bus = STSBus(packet_handler(port), motion_enabled=True)
        bus.sync_write({1: Target(2080, 50, 10), 2: Target(2000, 60, 12)})
        bus.sync_write({2: Target(2000, 60, 12)})
        self.assertEqual(len(port.writes[-1]), 16)
        self.assertEqual(port.writes[-1][7], 2)

    def test_default_bus_cannot_write(self):
        port = ScriptedPort()
        with self.assertRaises(ControlError):
            STSBus(packet_handler(port)).sync_write({1: Target(2000, 50, 10)})
        self.assertEqual(port.writes, [])

    def test_invalid_batch_never_writes_or_queues(self):
        port = ScriptedPort()
        bus = STSBus(packet_handler(port), motion_enabled=True)
        for target in [Target(4096, 50, 10), Target(2000, 0, 10), Target(2000, 50, 0)]:
            with self.assertRaises(ControlError):
                bus.sync_write({1: Target(2000, 50, 10), 2: target})
        self.assertEqual(port.writes, [])
        self.assertEqual(bus.packet.groupSyncWrite.data_dict, {})

    def test_short_write_fails_and_clears_queue(self):
        port = ScriptedPort(short_write=True)
        bus = STSBus(packet_handler(port), motion_enabled=True)
        with self.assertRaises(BusError):
            bus.sync_write({1: Target(2000, 50, 10)})
        self.assertEqual(bus.packet.groupSyncWrite.data_dict, {})


class MemoryBus:
    def __init__(self):
        self.sample = Feedback(1, 10, 1234, 4, 1, 0, 1, 2048, 0, 0, 12, 25, 0, 0, 2048, 0, 0, 0, 4095)
        self.writes = []
        self.fail_read = self.fail_write = False

    def read(self, servo_id):
        if self.fail_read:
            raise BusError("timeout")
        return replace(self.sample, servo_id=servo_id)

    def sync_write(self, targets):
        if self.fail_write:
            raise BusError("short write")
        self.writes.append(dict(targets))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.bus = MemoryBus()
        self.joint = Joint("head", 1, True, 1234, 4, 1800, 2200, 100, 100, 20,
                           offset_raw=0, hardware_min_ticks=0, hardware_max_ticks=4095)
        self.now = 10
        self.control = Controller(self.bus, {"head": self.joint}, clock=lambda: self.now)
        self.control.refresh("head")
        self.target = Target(2080, 50, 10)

    def submit(self):
        return self.control.submit({"head": self.target}, created_at=10)

    def test_sent_target_does_not_become_feedback_or_ack(self):
        receipt = self.submit()
        self.assertFalse(receipt["acknowledged"])
        self.assertEqual(self.control.feedback["head"].position_ticks, 2048)
        self.assertEqual(self.control.last_sent["head"].position_ticks, 2080)

    def test_expired_and_future_commands_never_write(self):
        for timestamp in [9, 11, float("nan"), float("inf")]:
            with self.assertRaises(ControlError):
                self.control.submit({"head": self.target}, created_at=timestamp)
        self.assertEqual(self.bus.writes, [])

    def test_stale_feedback_with_new_command_never_writes(self):
        self.now = 11
        with self.assertRaises(ControlError):
            self.control.submit({"head": self.target}, created_at=11)
        self.assertEqual(self.bus.writes, [])

    def test_configuration_and_status_mismatch_never_write(self):
        for update in [dict(model_number=999), dict(phase=0), dict(resolution=2), dict(mode=1),
                       dict(torque_enabled=0), dict(status=4), dict(servo_id=2),
                       dict(position_ticks=3000), dict(offset_raw=1),
                       dict(hardware_min_ticks=100), dict(hardware_max_ticks=0)]:
            self.control.feedback["head"] = replace(self.bus.sample, **update)
            with self.subTest(update=update), self.assertRaises(ControlError):
                self.submit()
        self.assertEqual(self.bus.writes, [])

    def test_bad_second_joint_prevents_entire_batch(self):
        self.control.joints["elbow"] = replace(self.joint, name="elbow", servo_id=2)
        self.control.refresh("elbow")
        with self.assertRaises(ControlError):
            self.control.submit({"head": self.target, "elbow": Target(2500, 50, 10)}, created_at=10)
        self.assertEqual(self.bus.writes, [])

    def test_mechanical_steps_speed_acceleration_and_types(self):
        for target in [Target(2200, 50, 10), Target(1799, 50, 10), Target(2080, 101, 10),
                       Target(2080, 50, 21), Target(2080, 0, 10), Target(2080, 50, 0),
                       Target(2080.0, 50, 10), Target(2080, True, 10)]:
            self.target = target
            with self.subTest(target=target), self.assertRaises(ControlError):
                self.submit()
        self.assertEqual(self.bus.writes, [])

    def test_missing_calibration_never_writes(self):
        self.control.joints["head"] = replace(self.joint, calibrated=False)
        with self.assertRaises(ControlError):
            self.submit()
        self.assertEqual(self.bus.writes, [])

    def test_failed_read_invalidates_sample_and_latches_fault(self):
        self.bus.fail_read = True
        with self.assertRaises(BusError):
            self.control.refresh("head")
        self.assertNotIn("head", self.control.feedback)
        self.bus.fail_read = False
        self.control.refresh("head")
        with self.assertRaises(ControlError):
            self.submit()
        self.control.reset_fault()
        with self.assertRaises(ControlError):
            self.submit()
        self.control.refresh("head")
        self.submit()

    def test_failed_write_latches_fault_without_claiming_target(self):
        self.bus.fail_write = True
        with self.assertRaises(BusError):
            self.submit()
        self.assertEqual(self.control.last_sent, {})
        self.assertEqual(self.control.feedback, {})
        with self.assertRaises(ControlError):
            self.submit()

    def test_real_joint_map_has_no_invented_calibration(self):
        path = Path(__file__).resolve().parents[2] / "hardware/little-guy-v2/joints.json"
        joints = load_joints(path)
        self.assertEqual(len(joints), 11)
        for joint in joints.values():
            with self.subTest(name=joint.name), self.assertRaises(ControlError):
                joint.check_calibration()


if __name__ == "__main__":
    unittest.main()
