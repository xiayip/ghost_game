#!/usr/bin/env python3
"""Capture and summarize the live Ghost palm-tracking control chain."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


JOINTS = tuple(f"joint{index}" for index in range(1, 7))


def percentile(values, percent):
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    rank = (len(values) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - rank) + values[upper] * (rank - lower)


def metric(values):
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def frequency(stamps):
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        return None
    return (len(stamps) - 1) / (stamps[-1] - stamps[0])


class PalmProfiler(Node):
    def __init__(self, output):
        super().__init__("ghost_palm_profiler")
        self.output = output
        self.latest_joint_positions = None
        self.reset_capture()

        self.create_subscription(
            String, "/gestures/palm_control", self.on_palm,
            qos_profile_sensor_data)
        self.create_subscription(
            String, "/ghost_game_node/state", self.on_state, 20)
        self.create_subscription(
            JointState, "/joint_states", self.on_joints, 50)
        self.create_subscription(
            JointTrajectoryPoint,
            "/zephyr_arm_impedance_controller/commands",
            self.on_command,
            50,
        )

    def reset_capture(self):
        self.events = []
        self.started = time.monotonic()
        self.palm_times = []
        self.state_times = []
        self.command_times = []
        self.joint_times = []
        self.inference_ms = []
        self.source_age_ms = []
        self.depth_age_ms = []
        self.tracking_error = []
        self.command_delta = []
        self.command_speed = []
        self.command_measured_error = []
        self.yaw_offset = []
        self.yaw_tracking_samples = 0
        self.palm_total = 0
        self.palm_usable = 0
        self.active_state_samples = 0

    def now(self):
        return time.monotonic() - self.started

    def record(self, kind, payload):
        self.events.append({"t": self.now(), "kind": kind, "data": payload})

    @staticmethod
    def finite_value(payload, name):
        value = payload.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def on_palm(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        self.palm_times.append(time.monotonic())
        self.palm_total += 1
        usable = (
            payload.get("palm_open") is True
            and payload.get("distance_valid") is True
        )
        self.palm_usable += int(usable)
        for name, target in (
            ("inference_ms", self.inference_ms),
            ("source_age_ms", self.source_age_ms),
        ):
            value = self.finite_value(payload, name)
            if value is not None:
                target.append(value)
        depth_age = self.finite_value(payload, "depth_age_sec")
        if depth_age is not None:
            self.depth_age_ms.append(depth_age * 1000.0)
        self.record("palm", payload)

    def on_state(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        self.state_times.append(time.monotonic())
        palm = payload.get("palm_interaction") or {}
        if palm.get("active") is True:
            self.active_state_samples += 1
            tracking = self.finite_value(palm, "tracking_error_rad")
            delta = self.finite_value(palm, "command_delta_rad")
            if tracking is not None:
                self.tracking_error.append(tracking)
            if delta is not None:
                self.command_delta.append(delta)
            yaw_offset = self.finite_value(palm, "yaw_offset_rad")
            if yaw_offset is not None:
                self.yaw_offset.append(yaw_offset)
            self.yaw_tracking_samples += int(
                palm.get("yaw_tracking") is True)
        self.record("state", {
            "phase": payload.get("phase"),
            "palm_interaction": palm,
        })

    def on_joints(self, message):
        index = {name: offset for offset, name in enumerate(message.name)}
        if not all(name in index for name in JOINTS):
            return
        positions = [float(message.position[index[name]]) for name in JOINTS]
        if not all(math.isfinite(value) for value in positions):
            return
        self.latest_joint_positions = positions
        self.joint_times.append(time.monotonic())
        self.record("joints", {"positions": positions})

    def on_command(self, message):
        if len(message.positions) < len(JOINTS):
            return
        positions = [float(value) for value in message.positions[:len(JOINTS)]]
        velocities = [
            float(value) for value in message.velocities[:len(JOINTS)]
        ]
        self.command_times.append(time.monotonic())
        if velocities:
            self.command_speed.append(max(abs(value) for value in velocities))
        if self.latest_joint_positions is not None:
            self.command_measured_error.append(max(
                abs(command - measured)
                for command, measured in zip(
                    positions, self.latest_joint_positions, strict=True)
            ))
        self.record("command", {
            "positions": positions,
            "velocities": velocities,
        })

    def summary(self):
        usable_percent = (
            self.palm_usable * 100.0 / self.palm_total
            if self.palm_total else None
        )
        return {
            "duration_sec": self.now(),
            "rates_hz": {
                "palm_control": frequency(self.palm_times),
                "game_state": frequency(self.state_times),
                "impedance_command": frequency(self.command_times),
                "joint_states": frequency(self.joint_times),
            },
            "palm_samples": self.palm_total,
            "palm_usable_percent": usable_percent,
            "active_state_samples": self.active_state_samples,
            "inference_ms": metric(self.inference_ms),
            "source_age_ms": metric(self.source_age_ms),
            "depth_age_ms": metric(self.depth_age_ms),
            "tracking_error_rad": metric(self.tracking_error),
            "command_delta_rad": metric(self.command_delta),
            "command_speed_rad_s": metric(self.command_speed),
            "command_measured_error_rad": metric(
                self.command_measured_error),
            "yaw_tracking_samples": self.yaw_tracking_samples,
            "yaw_offset_rad": metric(self.yaw_offset),
        }

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as stream:
            for event in self.events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--wait-for-palm", action="store_true")
    parser.add_argument("--warmup-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    output = args.output or Path(
        f"/tmp/ghost_palm_profile_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

    rclpy.init()
    node = PalmProfiler(output)
    try:
        if args.wait_for_palm:
            warmup_deadline = time.monotonic() + args.warmup_timeout
            while (rclpy.ok() and node.palm_total == 0
                   and time.monotonic() < warmup_deadline):
                rclpy.spin_once(node, timeout_sec=0.05)
            if node.palm_total == 0:
                raise RuntimeError(
                    "timed out waiting for /gestures/palm_control")
            node.reset_capture()
            print("capture_started=palm_control_matched", flush=True)
        deadline = time.monotonic() + args.duration
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        summary = node.summary()
        node.save()
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"raw_profile={output}")


if __name__ == "__main__":
    main()
