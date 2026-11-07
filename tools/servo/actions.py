"""Optional, single-owner outbox boundary; importing this module opens no device."""
from __future__ import annotations

import math
import re
import secrets
import time

from .control import ControlError, Target, integer


REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def finite_time(value, field):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ControlError(f"{field} must be a finite timestamp")
    return value


class ActionDispatcher:
    """Dispatch structured requests to an operator-supplied Controller.

    This is a local integration seam, not an enabled guard service. The host
    owns calibration, clocks and the bus. Every new dispatcher starts a new
    session; queued notes from earlier sessions cannot be replayed. Request IDs
    remain consumed for this lifetime, including uncertain transport attempts.
    """

    def __init__(self, controller, *, wall_clock=time.time, max_queue_age_s=2.0,
                 max_requests=4096):
        finite_time(max_queue_age_s, "max_queue_age_s")
        if max_queue_age_s <= 0:
            raise ControlError("max_queue_age_s must be positive")
        integer(max_requests, 1, 1_000_000, "max_requests")
        self.controller = controller
        self.wall_clock = wall_clock
        self.max_queue_age_s = max_queue_age_s
        self.max_requests = max_requests
        self.used_ids = set()
        self._session_id = secrets.token_hex(32)

    @property
    def session_id(self):
        """Receiver lifetime identity for a trusted producer's newly created notes.

        Never stamp this onto existing queued notes or restore it after restart.
        It is a correlation value, not permission to move or authentication.
        """
        return self._session_id

    def _check_time(self, created, expires):
        now = finite_time(self.wall_clock(), "host wall clock")
        if not 0 <= now - created <= self.max_queue_age_s or now >= expires:
            raise ControlError("movement request is stale or from the future")
        return now

    def dispatch(self, note):
        if type(note) is not dict or set(note) != {
            "kind", "schema_version", "session_id", "request_id", "created_at", "expires_at", "targets"
        }:
            raise ControlError("movement request fields do not match the schema")
        if note["kind"] != "joint_targets" or type(note["schema_version"]) is not int or note["schema_version"] != 2:
            raise ControlError("unsupported movement request schema")
        if type(note["session_id"]) is not str or note["session_id"] != self.session_id:
            raise ControlError("movement request belongs to a different session")
        request_id = note["request_id"]
        if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
            raise ControlError("invalid request_id")
        created = finite_time(note["created_at"], "created_at")
        expires = finite_time(note["expires_at"], "expires_at")
        if not 0 < expires - created <= self.max_queue_age_s:
            raise ControlError("movement request lifetime exceeds queue policy")
        # Capture both clocks before reading the bus. Neither a slow read nor a
        # backwards wall-clock change may give an old request a new deadline.
        accepted_at = self.controller.clock()
        remaining = expires - self._check_time(created, expires)
        if request_id in self.used_ids:
            raise ControlError("movement request ID already consumed")
        if len(self.used_ids) >= self.max_requests:
            raise ControlError("movement request capacity reached; operator restart required")
        if self.controller.fault:
            raise ControlError(f"controller fault: {self.controller.fault}")
        rows = note["targets"]
        if type(rows) is not dict or not 1 <= len(rows) <= len(self.controller.joints):
            raise ControlError("targets must contain known joint names")
        targets = {}
        for name, row in rows.items():
            if name not in self.controller.joints:
                raise ControlError(f"unknown joint: {name}")
            if type(row) is not dict or set(row) != {"position_ticks", "speed_raw", "acceleration_raw"}:
                raise ControlError(f"{name}: invalid target fields")
            joint = self.controller.joints[name]
            joint.check_calibration()
            targets[name] = Target(
                integer(row["position_ticks"], joint.min_ticks, joint.max_ticks, "position_ticks"),
                integer(row["speed_raw"], 1, joint.max_speed_raw, "speed_raw"),
                integer(row["acceleration_raw"], 1, joint.max_acceleration_raw, "acceleration_raw"),
            )
        # Consume before any device access; never automatically retry a write
        # whose delivery may be uncertain. No evictions permit replay later.
        self.used_ids.add(request_id)
        for name in targets:
            self.controller.refresh(name)
        self._check_time(created, expires)
        elapsed = self.controller.clock() - accepted_at
        if not math.isfinite(elapsed) or not 0 <= elapsed < remaining:
            raise ControlError("movement request expired during feedback reads")
        return self.controller.submit(targets, created_at=accepted_at, deadline=accepted_at + remaining)
