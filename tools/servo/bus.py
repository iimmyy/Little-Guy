"""FEETECH SDK adapter. No device is opened until open_serial() is called."""
from __future__ import annotations

from contextlib import contextmanager
import time

from .control import ControlError, Feedback, integer


class BusError(RuntimeError):
    pass


def signed_magnitude(value, bit):
    return -(value & ~(1 << bit)) if value & (1 << bit) else value


def packet_handler(port):
    """Keep SDK framing, but reject a valid-checksum response of the wrong read length."""
    from .vendor.scservo_sdk import sms_sts, COMM_RX_CORRUPT, COMM_RX_TIMEOUT

    class ReceiveTimeout(Exception):
        pass

    class DeadlinePort:
        # SDK resynchronization branches omit timeout checks. Enforce the deadline
        # on every receive read, even when a broken wire supplies continuous noise.
        def __getattr__(self, name):
            return getattr(port, name)

        @property
        def is_using(self):
            return port.is_using

        @is_using.setter
        def is_using(self, value):
            port.is_using = value

        def readPort(self, length):
            if port.isPacketTimeout():
                raise ReceiveTimeout()
            return port.readPort(length)

    class CheckedSTS(sms_sts):
        expected_response_size = None

        def readTxRx(self, servo_id, address, length):
            self.expected_response_size = length + 6
            try:
                return super().readTxRx(servo_id, address, length)
            finally:
                self.expected_response_size = None

        def rxPacket(self):
            if self.portHandler.isPacketTimeout():
                self.portHandler.is_using = False
                return [], COMM_RX_TIMEOUT
            try:
                packet, result = super().rxPacket()
            except ReceiveTimeout:
                self.portHandler.is_using = False
                return [], COMM_RX_TIMEOUT
            except (IndexError, ValueError):
                self.portHandler.is_using = False
                return [], COMM_RX_CORRUPT
            if result == 0 and self.expected_response_size is not None:
                if len(packet) != self.expected_response_size:
                    return [], COMM_RX_CORRUPT
            return packet, result

    return CheckedSTS(DeadlinePort())


class STSBus:
    def __init__(self, packet_handler, *, clock=time.monotonic, motion_enabled=False):
        self.packet = packet_handler
        self.clock = clock
        self.motion_enabled = motion_enabled

    def read(self, servo_id):
        integer(servo_id, 1, 253, "servo_id")
        started = self.clock()
        # One contiguous register read: model at3 through phase-current high byte at70.
        data, result, error = self.packet.readTxRx(servo_id, 3, 68)
        if result != 0 or error != 0:
            raise BusError(f"servo {servo_id}: communication={result}, status={error:#x}")
        if len(data) != 68 or any(type(x) is not int or not 0 <= x <= 255 for x in data):
            raise BusError(f"servo {servo_id}: invalid register response length or bytes")

        def byte(address):
            return data[address - 3]

        def word(address):
            return byte(address) | byte(address + 1) << 8

        return Feedback(servo_id=servo_id, sampled_at=started, model_number=word(3),
                        phase=byte(18), resolution=byte(30), mode=byte(33), torque_enabled=byte(40),
                        position_ticks=signed_magnitude(word(56), 15),
                        speed_raw=signed_magnitude(word(58), 15),
                        load_duty_raw=signed_magnitude(word(60), 10), voltage_v=byte(62) / 10,
                        temperature_c=byte(63), status=byte(65), moving=byte(66),
                        target_ticks=signed_magnitude(word(67), 15), phase_current_raw=word(69),
                        offset_raw=word(31), hardware_min_ticks=word(9), hardware_max_ticks=word(11))

    def sync_write(self, targets):
        if not self.motion_enabled:
            raise ControlError("this bus is read-only")
        if not 1 <= len(targets) <= 30:
            raise ControlError("SDK sync-write packet supports 1 to 30 targets")
        # Validate even when used without Controller. Mechanical limits belong to Controller.
        for servo_id, target in targets.items():
            integer(servo_id, 1, 253, "servo_id")
            integer(target.position_ticks, 0, 4095, "position_ticks")
            integer(target.speed_raw, 1, 32767, "speed_raw")
            integer(target.acceleration_raw, 1, 254, "acceleration_raw")
        group = self.packet.groupSyncWrite
        group.clearParam()
        try:
            for servo_id, target in sorted(targets.items()):
                if not self.packet.SyncWritePosEx(servo_id, target.position_ticks,
                                                target.speed_raw, target.acceleration_raw):
                    raise BusError(f"could not queue servo {servo_id}")
            result = group.txPacket()
            if result != 0:
                raise BusError(f"sync write communication error: {result}")
        finally:
            group.clearParam()


@contextmanager
def open_serial(device, *, baud=1000000):
    """Explicit, read-only hardware connection. Never auto-discovers or enables torque."""
    from .vendor.scservo_sdk import PortHandler

    class MonotonicPort(PortHandler):
        def getCurrentTime(self):
            return time.monotonic() * 1000

        def setupPort(self, baud):
            result = super().setupPort(baud)
            self.ser.write_timeout = 0.1
            self.ser.timeout = 0.002
            return result

    port = MonotonicPort(device)
    try:
        if not port.setBaudRate(baud):
            raise BusError(f"unsupported baud rate: {baud}")
        yield STSBus(packet_handler(port))
    finally:
        if port.is_open:
            port.closePort()
