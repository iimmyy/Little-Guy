"""Inspect ST3215 feedback or exercise the controller with synthetic data."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .bus import open_serial
from .control import Controller, Feedback, Joint, Target, load_joints, integer


def demo():
    # Synthetic values deliberately live here, never in the real joint map.
    class SimulatedBus:
        def read(self, servo_id):
            return Feedback(servo_id, 10.0, 1234, 0, 1, 0, 1, 2048, 0, 0, 12.0, 25, 0, 0, 2048, 0, 0, 0, 4095)

        def sync_write(self, targets):
            self.sent = targets

    bus = SimulatedBus()
    joint = Joint("synthetic_joint", 1, True, 1234, 0, 1800, 2200, 100, 100, 20,
                  offset_raw=0, hardware_min_ticks=0, hardware_max_ticks=4095)
    control = Controller(bus, {joint.name: joint}, clock=lambda: 10.0)
    before = control.refresh(joint.name)
    receipt = control.submit({joint.name: Target(2080, 50, 10)}, created_at=10.0)
    return {"source": "simulation_only", "receipt": receipt,
            "last_sent_target": asdict(control.last_sent[joint.name]),
            "feedback": asdict(before), "physical_motion_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="synthetic data only; opens no device")
    check = commands.add_parser("check-map", help="inspect calibration readiness")
    check.add_argument("path", type=Path)
    status = commands.add_parser("status", help="explicit read-only register request")
    status.add_argument("--port", required=True)
    status.add_argument("--ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            result = demo()
        elif args.command == "check-map":
            joints = load_joints(args.path)
            result = {"joint_count": len(joints), "calibrated": [n for n, j in joints.items() if j.calibrated],
                      "uncalibrated": [n for n, j in joints.items() if not j.calibrated]}
        else:
            for servo_id in args.ids:
                integer(servo_id, 1, 253, "servo_id")
            if len(args.ids) != len(set(args.ids)):
                raise ValueError("duplicate IDs")
            with open_serial(args.port) as bus:
                result = {"source": "servo_registers", "samples": [asdict(bus.read(i)) for i in args.ids]}
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"servo: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
