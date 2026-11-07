"""Bounded joint commands, with actual feedback kept separate from sent targets."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time


class ControlError(ValueError):
    pass


def integer(value, low, high, field):
    if type(value) is not int or not low <= value <= high:
        raise ControlError(f"{field} must be an integer from {low} to {high}")
    return value


@dataclass(frozen=True)
class Feedback:
    servo_id: int
    sampled_at: float
    model_number: int
    phase: int
    resolution: int
    mode: int
    torque_enabled: int
    position_ticks: int
    speed_raw: int
    load_duty_raw: int
    voltage_v: float
    temperature_c: int
    status: int
    moving: int
    target_ticks: int
    phase_current_raw: int
    offset_raw: int
    hardware_min_ticks: int
    hardware_max_ticks: int


@dataclass(frozen=True)
class Target:
    position_ticks: int
    speed_raw: int
    acceleration_raw: int


@dataclass(frozen=True)
class Joint:
    name: str
    servo_id: int
    calibrated: bool = False
    model_number: int | None = None
    phase: int | None = None
    min_ticks: int | None = None
    max_ticks: int | None = None
    max_step_ticks: int | None = None
    max_speed_raw: int | None = None
    max_acceleration_raw: int | None = None
    offset_raw: int | None = None
    hardware_min_ticks: int | None = None
    hardware_max_ticks: int | None = None

    def check_calibration(self):
        if self.calibrated is not True:
            raise ControlError(f"{self.name}: mechanical limits have not been calibrated")
        integer(self.model_number, 0, 65535, "model_number")
        integer(self.phase, 0, 254, "phase")
        integer(self.min_ticks, 0, 4094, "min_ticks")
        integer(self.max_ticks, self.min_ticks + 1, 4095, "max_ticks")
        integer(self.max_step_ticks, 1, 4095, "max_step_ticks")
        integer(self.max_speed_raw, 1, 32767, "max_speed_raw")
        integer(self.max_acceleration_raw, 1, 254, "max_acceleration_raw")
        integer(self.offset_raw, 0, 8191, "offset_raw")
        integer(self.hardware_min_ticks, 0, 4094, "hardware_min_ticks")
        integer(self.hardware_max_ticks, self.hardware_min_ticks + 1, 4095, "hardware_max_ticks")
        if not self.hardware_min_ticks <= self.min_ticks < self.max_ticks <= self.hardware_max_ticks:
            raise ControlError("mechanical limits must fit within the calibrated device limits")


def load_joints(path: Path) -> dict[str, Joint]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ControlError("unsupported joint-map version")
    joints = {}
    ids = set()
    for row in data["joints"]:
        joint = Joint(**row)
        integer(joint.servo_id, 1, 253, "servo_id")
        if not joint.name or joint.name in joints or joint.servo_id in ids:
            raise ControlError("joint names and servo IDs must be unique")
        if type(joint.calibrated) is not bool:
            raise ControlError("calibrated must be a boolean")
        if joint.calibrated:
            joint.check_calibration()
        joints[joint.name] = joint
        ids.add(joint.servo_id)
    if not joints:
        raise ControlError("joint map is empty")
    return joints


class Controller:
    """Single-owner controller. An in-process freshness check is not a hardware watchdog."""

    def __init__(self, bus, joints, *, clock=time.monotonic, max_age_s=0.3):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ControlError("max_age_s must be positive and finite")
        self.bus, self.joints, self.clock = bus, dict(joints), clock
        self.max_age_s = max_age_s
        self.feedback = {}
        self.last_sent = {}
        self.fault = None

    def refresh(self, name):
        joint = self.joints[name]
        # A failed read must not leave an old sample looking current.
        self.feedback.pop(name, None)
        try:
            sample = self.bus.read(joint.servo_id)
        except Exception as error:
            self.fault = f"feedback failed for {name}: {error}"
            raise
        self.feedback[name] = sample
        return sample

    def reset_fault(self):
        """Explicit operator reset; fresh feedback is still required before a command."""
        self.fault = None
        self.feedback.clear()

    def _fresh(self, timestamp, now, label):
        if not math.isfinite(timestamp) or not 0 <= now - timestamp <= self.max_age_s:
            raise ControlError(f"{label} is stale or from the future")

    def submit(self, targets: dict[str, Target], *, created_at: float, deadline: float | None = None):
        if self.fault:
            raise ControlError(f"controller fault: {self.fault}")
        now = self.clock()
        self._fresh(created_at, now, "command")
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ControlError("command deadline must be finite")
        if not targets:
            raise ControlError("empty command")
        batch = {}
        for name, target in targets.items():
            if name not in self.joints:
                raise ControlError(f"unknown joint: {name}")
            joint = self.joints[name]
            joint.check_calibration()
            sample = self.feedback.get(name)
            if sample is None:
                raise ControlError(f"{name}: fresh feedback required")
            self._fresh(sample.sampled_at, now, f"{name} feedback")
            if sample.servo_id != joint.servo_id:
                raise ControlError(f"{name}: feedback ID mismatch")
            if (sample.model_number != joint.model_number or sample.phase != joint.phase
                    or sample.resolution != 1 or sample.mode != 0 or sample.torque_enabled != 1
                    or sample.offset_raw != joint.offset_raw
                    or sample.hardware_min_ticks != joint.hardware_min_ticks
                    or sample.hardware_max_ticks != joint.hardware_max_ticks):
                raise ControlError(f"{name}: servo configuration differs from calibration")
            if sample.status:
                raise ControlError(f"{name}: servo fault bits {sample.status:#x}")
            integer(sample.position_ticks, joint.min_ticks, joint.max_ticks, "feedback position")
            integer(target.position_ticks, joint.min_ticks, joint.max_ticks, "target position")
            integer(target.speed_raw, 1, joint.max_speed_raw, "speed_raw")
            integer(target.acceleration_raw, 1, joint.max_acceleration_raw, "acceleration_raw")
            if abs(target.position_ticks - sample.position_ticks) > joint.max_step_ticks:
                raise ControlError(f"{name}: target step exceeds calibrated limit")
            if joint.servo_id in batch:
                raise ControlError("duplicate servo ID")
            batch[joint.servo_id] = target
        # All joints pass before any motion write; a sync write is not an acknowledgement.
        # Recheck immediately before initiating transport, including any stricter
        # queue deadline. This cannot retract bytes already handed to the bus.
        now = self.clock()
        self._fresh(created_at, now, "command")
        for name in targets:
            self._fresh(self.feedback[name].sampled_at, now, f"{name} feedback")
        if deadline is not None and now >= deadline:
            raise ControlError("command deadline expired before write")
        try:
            self.bus.sync_write(batch)
        except Exception as error:
            self.fault = f"motion write failed: {error}"
            self.feedback.clear()
            raise
        self.last_sent = dict(targets)
        return {"sent_ids": sorted(batch), "acknowledged": False, "sent_at": self.clock()}
