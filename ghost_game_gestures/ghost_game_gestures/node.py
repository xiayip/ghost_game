"""Shared-image ROS adapter with bounded work and fail-closed freshness checks.

Inference owns a worker thread. The ROS timer alone updates semantic latches
and publishes events, so dropping an inference result cannot consume an event.
Imports of ROS, OpenCV and MediaPipe are deferred until they are actually used.
"""
from dataclasses import dataclass
import importlib.util
import json
import math
from threading import Condition, Event, Thread
import time

from .model import DEFAULT_MODEL_PATH, GestureModel, validate_model_file


def frame_age_seconds(stamp_ns, now_ns, max_age):
    """Zero/missing, future and expired camera stamps are not actionable."""
    age = (now_ns - stamp_ns) / 1e9
    return age if stamp_ns > 0 and 0 <= age <= max_age else None


class _LatestMailbox:
    def __init__(self):
        self.condition = Condition()
        self.item = None
        self.overwritten = 0

    def put(self, item):
        with self.condition:
            if self.item is not None:
                self.overwritten += 1
            self.item = item
            self.condition.notify()

    def take(self, timeout=0.):
        with self.condition:
            if timeout and self.item is None:
                self.condition.wait_for(lambda: self.item is not None, timeout)
            item, self.item = self.item, None
            return item


class _LatestWorker:
    """At most one active, one pending and one completed frame."""
    def __init__(self, process, on_error, cleanup, max_fps):
        if not math.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("max_processing_rate must be finite and positive")
        self.process, self.on_error, self.cleanup = process, on_error, cleanup
        self.period = 1. / max_fps
        self.pending, self.completed = _LatestMailbox(), _LatestMailbox()
        self.stop = Event()
        self.thread = Thread(target=self._run, daemon=True, name="gesture_inference")
        self.thread.start()

    def _run(self):
        deadline = 0.
        try:
            while not self.stop.is_set():
                if self.stop.wait(max(0., deadline - time.monotonic())):
                    break
                job = self.pending.take(.05)
                if job is None:
                    continue
                started = time.monotonic()
                deadline = max(deadline + self.period, started + self.period)
                try:
                    result = self.process(job)
                except Exception as exc:
                    result = self.on_error(job, exc)
                if not self.stop.is_set():
                    self.completed.put(result)
        finally:
            self.cleanup()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3.)


def decode_image_message(message, transport):
    """Decode in the worker; honor raw-image row padding and RGB ordering."""
    import cv2
    import numpy as np

    if transport == "compressed":
        frame = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or not frame.size:
            raise ValueError("Could not decode compressed RGB image")
        return frame
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}
    encoding = message.encoding.lower()
    if encoding not in channels:
        raise ValueError(f"Unsupported raw RGB encoding: {encoding}")
    width, height, step = int(message.width), int(message.height), int(message.step)
    count = channels[encoding]
    if min(width, height) <= 0 or step < width * count or len(message.data) < step * height:
        raise ValueError("Invalid raw image dimensions, row stride or buffer size")
    rows = np.frombuffer(message.data, dtype=np.uint8, count=step * height).reshape(height, step)
    frame = rows[:, :width * count].reshape(height, width, count).copy()
    conversions = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
                   "bgra8": cv2.COLOR_BGRA2BGR, "mono8": cv2.COLOR_GRAY2BGR}
    return cv2.cvtColor(frame, conversions[encoding]) if encoding != "bgr8" else frame


@dataclass
class _Frame:
    message: object
    stamp_ns: int
    received: float


def create_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import String
    from .core import GestureEngine

    class GestureNode(Node):
        def __init__(self):
            super().__init__("gesture_recognition")
            defaults = {
                "image_topic": "/camera/color/image_raw/compressed",
                "image_transport": "compressed", "model_path": DEFAULT_MODEL_PATH,
                "num_hands": 2, "max_width": 640, "max_processing_rate": 20.,
                "max_frame_age": .20, "watchdog_timeout": .25, "publish_rate": 60.,
                "min_detection_confidence": .5, "min_presence_confidence": .5,
                "min_tracking_confidence": .5, "score_threshold": .65,
                "hold_seconds": .35, "release_seconds": .25, "cooldown_seconds": 1.5,
                "max_gap": .25, "move_deadzone": .06, "move_scale": .25,
                "state_topic": "/gestures/state", "events_topic": "/gestures/events",
                "palm_control_topic": "/gestures/palm_control",
            }
            for name, value in defaults.items():
                self.declare_parameter(name, value)
            p = lambda name: self.get_parameter(name).value
            self.transport = p("image_transport")
            if self.transport not in ("compressed", "raw"):
                raise ValueError("image_transport must be compressed or raw")
            for name in ("max_processing_rate", "max_frame_age", "watchdog_timeout", "publish_rate"):
                if not math.isfinite(p(name)) or p(name) <= 0:
                    raise ValueError(f"{name} must be finite and positive")
            for name in ("min_detection_confidence", "min_presence_confidence", "min_tracking_confidence"):
                if not math.isfinite(p(name)) or not 0 <= p(name) <= 1:
                    raise ValueError(f"{name} must be in [0, 1]")
            if p("num_hands") < 1 or p("max_width") < 64:
                raise ValueError("Require num_hands >= 1 and max_width >= 64")
            self.max_age = float(p("max_frame_age"))
            self.watchdog_timeout = float(p("watchdog_timeout"))
            self.model_options = {name: p(name) for name in (
                "model_path", "num_hands", "max_width", "min_detection_confidence",
                "min_presence_confidence", "min_tracking_confidence")}
            validate_model_file(p("model_path"))
            missing = [name for name in ("mediapipe", "cv2", "numpy")
                       if importlib.util.find_spec(name) is None]
            if missing:
                raise RuntimeError("Missing gesture dependencies: " + ", ".join(missing)
                                   + ". See GESTURE_INTERACTION.md for installation.")
            self.engine = GestureEngine(
                threshold=p("score_threshold"), hold_seconds=p("hold_seconds"),
                release_seconds=p("release_seconds"), cooldown_seconds=p("cooldown_seconds"),
                max_gap=p("max_gap"), move_deadzone=p("move_deadzone"), move_scale=p("move_scale"))
            latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
            events = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)
            self.states = self.create_publisher(String, p("state_topic"), latest)
            self.events = self.create_publisher(String, p("events_topic"), events)
            self.controls = self.create_publisher(String, p("palm_control_topic"), latest)
            self.model = None
            self.model_failure = None
            self.frames_received = self.frames_processed = self.errors = self.stale = 0
            self.last_source_ns = None
            self.last_fresh_wall = time.monotonic()
            self.last_tick_ns = self.get_clock().now().nanoseconds
            self.next_invalid_wall = self.next_log_wall = 0.
            self.worker = _LatestWorker(self._infer, self._error, self._close_model,
                                        float(p("max_processing_rate")))
            self.create_subscription(CompressedImage if self.transport == "compressed" else Image,
                                     p("image_topic"), self._image, latest)
            # Steady clock guarantees timeout/stop publication even when /clock pauses.
            from rclpy.clock import Clock, ClockType
            self.timer = self.create_timer(1. / p("publish_rate"), self._tick,
                                          clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.get_logger().info("Gesture perception subscribed to " + p("image_topic")
                                   + "; labels/events only, no robot motion commands")

        def _image(self, message):
            self.frames_received += 1
            stamp = message.header.stamp
            self.worker.pending.put(_Frame(message, stamp.sec * 1_000_000_000 + stamp.nanosec,
                                           time.monotonic()))

        def _infer(self, job):
            def expired():
                return (frame_age_seconds(job.stamp_ns, self.get_clock().now().nanoseconds,
                                          self.max_age) is None
                        or time.monotonic() - job.received > self.max_age)
            if expired():
                self.stale += 1
                return {"stamp_ns": job.stamp_ns, "valid": False, "reason": "stale_frame"}
            if self.model_failure:
                return {"stamp_ns": job.stamp_ns, "valid": False, "reason": "model_unavailable"}
            if self.model is None:
                try:
                    self.model = GestureModel(**self.model_options)
                except Exception as exc:
                    self.model_failure = str(exc)
                    raise
            frame = decode_image_message(job.message, self.transport)
            if expired():
                self.stale += 1
                return {"stamp_ns": job.stamp_ns, "valid": False, "reason": "stale_frame"}
            started = time.monotonic()
            hands = self.model.recognize(frame)
            inference_ms = (time.monotonic() - started) * 1000.
            self.frames_processed += 1
            valid = not expired()
            if not valid:
                self.stale += 1
            return {"stamp_ns": job.stamp_ns, "valid": valid,
                    "reason": "ok" if valid else "stale_frame",
                    "hands": hands, "frame_size": (frame.shape[1], frame.shape[0]),
                    "inference_ms": inference_ms, "received": job.received}

        def _error(self, job, error):
            self.errors += 1
            wall = time.monotonic()
            if wall >= self.next_log_wall:
                self.get_logger().error(f"Gesture inference failed: {error}")
                self.next_log_wall = wall + 5.
            return {"stamp_ns": job.stamp_ns, "valid": False, "reason": "inference_error"}

        def _stats(self, age=None, inference_ms=None):
            return {"received": self.frames_received, "processed": self.frames_processed,
                    "dropped_pending": self.worker.pending.overwritten,
                    "dropped_completed": self.worker.completed.overwritten,
                    "stale": self.stale, "errors": self.errors,
                    "source_age_ms": None if age is None else age * 1000.,
                    "inference_ms": inference_ms}

        @staticmethod
        def _send(publisher, value):
            message = String()
            message.data = json.dumps(value, allow_nan=False, separators=(",", ":"))
            publisher.publish(message)

        def _publish(self, state):
            self._send(self.states, state)
            self._send(self.controls, dict(state["control"], stamp=state["stamp"],
                                            hand_id=state["hand_id"], units="dimensionless"))
            for event in state["events"]:
                self._send(self.events, event)

        def _invalid(self, reason, stamp_ns, now_ns):
            # Invalid observations suspend tracking/control but do not count as
            # a witnessed release; a camera gap must not retrigger a held pose.
            state = self.engine.process(None, (stamp_ns or 0) / 1e9, now_ns / 1e9,
                                        max_age=self.max_age)
            state["reason"] = state["control"]["reason"] = reason
            state["stats"] = self._stats()
            self._publish(state)
            self.next_invalid_wall = time.monotonic() + .1

        def _tick(self):
            now_ns = self.get_clock().now().nanoseconds
            wall = time.monotonic()
            if now_ns < self.last_tick_ns:
                self.engine.reset()
                self.worker.pending.take()
                self.worker.completed.take()
                self.last_source_ns = None
                self._invalid("clock_reset", None, now_ns)
            self.last_tick_ns = now_ns
            result = self.worker.completed.take()
            if result is not None:
                stamp_ns = result["stamp_ns"]
                age = frame_age_seconds(stamp_ns, now_ns, self.max_age)
                if (not result["valid"] or age is None
                        or wall - result.get("received", wall) > self.max_age):
                    self._invalid(result["reason"] if not result["valid"] else "stale_frame",
                                  stamp_ns, now_ns)
                    return
                self.last_source_ns, self.last_fresh_wall = stamp_ns, wall
                state = self.engine.process(result["hands"], stamp_ns / 1e9, now_ns / 1e9,
                                            frame_size=result["frame_size"], max_age=self.max_age)
                state["stats"] = self._stats(age, result["inference_ms"])
                self._publish(state)
                return
            expired = (self.last_source_ns is not None
                       and frame_age_seconds(self.last_source_ns, now_ns, self.max_age) is None)
            if wall >= self.next_invalid_wall and (expired or wall - self.last_fresh_wall >= self.watchdog_timeout):
                self._invalid("camera_timeout", self.last_source_ns, now_ns)

        def _close_model(self):
            if self.model is not None:
                self.model.close()

        def destroy_node(self):
            if hasattr(self, "worker"):
                self.worker.close()
            return super().destroy_node()

    return GestureNode()


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = create_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
