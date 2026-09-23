"""One camera subscriber and one latest-frame worker for face or gesture mode."""

from dataclasses import dataclass
import json
import math
from threading import Condition, Event, Thread
import time

import cv2
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .backends import FaceBackend, GestureBackend, hand_boxes
from .mode import FACE, GESTURE, OFF, ModeState, normalize_mode


def frame_age_seconds(stamp_ns, now_ns, max_age):
    age = (now_ns - stamp_ns) / 1e9
    return age if stamp_ns > 0 and 0 <= age <= max_age else None


def decode_image_message(message, compressed):
    """Decode JPEG or common raw encodings without cv_bridge."""
    if compressed:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        if encoded.size == 0:
            raise ValueError("compressed image data is empty")
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("failed to decode compressed image")
        return image

    encoding = message.encoding.lower()
    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1, "8uc1": 1}
    if encoding not in channels:
        raise ValueError(f"unsupported raw image encoding: {message.encoding}")
    count = channels[encoding]
    width, height, step = int(message.width), int(message.height), int(message.step)
    if min(width, height) <= 0 or step < width * count or len(message.data) < step * height:
        raise ValueError("invalid raw image dimensions, stride, or buffer")
    rows = np.frombuffer(message.data, dtype=np.uint8, count=step * height).reshape(height, step)
    pixels = rows[:, : width * count]
    if count == 1:
        return cv2.cvtColor(pixels.reshape(height, width), cv2.COLOR_GRAY2BGR)
    image = pixels.reshape(height, width, count)
    if encoding == "rgb8":
        return np.ascontiguousarray(image[:, :, ::-1])
    return np.ascontiguousarray(image)


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

    def take(self, timeout=0.0):
        with self.condition:
            if timeout and self.item is None:
                self.condition.wait_for(lambda: self.item is not None, timeout)
            item, self.item = self.item, None
            return item


class _LatestWorker:
    """Bound work to one active, one pending, and one completed frame."""

    def __init__(self, process, on_error, cleanup):
        self.process = process
        self.on_error = on_error
        self.cleanup = cleanup
        self.pending = _LatestMailbox()
        self.completed = _LatestMailbox()
        self.stop = Event()
        self.thread = Thread(target=self._run, daemon=True, name="ghost_perception")
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                job = self.pending.take(0.05)
                if job is None:
                    continue
                try:
                    result = self.process(job)
                except Exception as error:
                    result = self.on_error(job, error)
                if not self.stop.is_set():
                    self.completed.put(result)
        finally:
            self.cleanup()

    def clear(self):
        self.pending.take()
        self.completed.take()

    def close(self):
        self.stop.set()
        with self.pending.condition:
            self.pending.condition.notify()
        self.thread.join(timeout=5.0)


@dataclass
class _Frame:
    message: object
    stamp_ns: int
    received: float
    mode: str
    generation: int


class GhostGamePerceptionNode(Node):
    """Run exactly one phase-selected perception backend for each RGB frame."""

    def __init__(self):
        super().__init__("ghost_game_perception")
        defaults = {
            "image_topic": "/camera/color/image_raw/compressed",
            "input_compressed": True,
            "mode_topic": "/ghost/perception/mode",
            "status_topic": "/ghost/perception/status",
            "initial_mode": "OFF",
            "face_enabled": True,
            "gesture_enabled": False,
            "face_model_path": "",
            "gesture_model_path": "~/.local/share/ghost_game/gesture_recognizer.task",
            "face_max_processing_rate": 30.0,
            "gesture_max_processing_rate": 20.0,
            "face_max_frame_age": 0.25,
            "gesture_max_frame_age": 0.20,
            "watchdog_timeout": 0.30,
            "publish_rate": 60.0,
            "face_detection_topic": "/nearest_face/detection",
            "face_head_crop_topic": "/nearest_face/head_crop",
            "face_publish_head_crop": True,
            "face_debug_image_topic": "/nearest_face/debug_image",
            "face_debug_compressed_topic": "/nearest_face/debug_image/compressed",
            "face_publish_debug_image": True,
            "face_score_threshold": 0.70,
            "face_nms_threshold": 0.30,
            "face_top_k": 5000,
            "face_detector_input_width": 424,
            "face_bbox_expand_left_ratio": 0.50,
            "face_bbox_expand_right_ratio": 0.50,
            "face_bbox_expand_top_ratio": 0.75,
            "face_bbox_expand_bottom_ratio": 0.50,
            "face_jpeg_quality": 85,
            "gesture_num_hands": 2,
            "gesture_max_width": 640,
            "gesture_min_detection_confidence": 0.5,
            "gesture_min_presence_confidence": 0.5,
            "gesture_min_tracking_confidence": 0.5,
            "gesture_score_threshold": 0.65,
            "gesture_hold_seconds": 0.35,
            "gesture_release_seconds": 0.25,
            "gesture_cooldown_seconds": 1.5,
            "gesture_max_gap": 0.25,
            "gesture_move_deadzone": 0.06,
            "gesture_move_scale": 0.25,
            "gesture_bbox_padding_ratio": 0.08,
            "gesture_state_topic": "/gestures/state",
            "gesture_events_topic": "/gestures/events",
            "gesture_palm_control_topic": "/gestures/palm_control",
            "gesture_detection_topic": "/gestures/detections",
            "gesture_debug_compressed_topic": "/gestures/debug_image/compressed",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value

        self.image_topic = str(p("image_topic"))
        self.input_compressed = bool(p("input_compressed"))
        self.face_enabled = bool(p("face_enabled"))
        self.gesture_enabled = bool(p("gesture_enabled"))
        self.rates = {
            FACE: float(p("face_max_processing_rate")),
            GESTURE: float(p("gesture_max_processing_rate")),
        }
        self.max_ages = {
            FACE: float(p("face_max_frame_age")),
            GESTURE: float(p("gesture_max_frame_age")),
        }
        self.watchdog_timeout = float(p("watchdog_timeout"))
        self.publish_rate = float(p("publish_rate"))
        if any(not math.isfinite(value) or value <= 0 for value in (
            *self.rates.values(), *self.max_ages.values(),
            self.watchdog_timeout, self.publish_rate,
        )):
            raise ValueError("perception rates and timeouts must be finite and positive")

        self.jpeg_quality = int(p("face_jpeg_quality"))
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("face_jpeg_quality must be in [1, 100]")
        self.gesture_bbox_padding = float(p("gesture_bbox_padding_ratio"))
        if not math.isfinite(self.gesture_bbox_padding) or self.gesture_bbox_padding < 0:
            raise ValueError("gesture_bbox_padding_ratio must be finite and non-negative")

        self.face_options = {
            "model_path": str(p("face_model_path")),
            "score_threshold": float(p("face_score_threshold")),
            "nms_threshold": float(p("face_nms_threshold")),
            "top_k": int(p("face_top_k")),
            "input_width": int(p("face_detector_input_width")),
            "expansion": (
                float(p("face_bbox_expand_left_ratio")),
                float(p("face_bbox_expand_right_ratio")),
                float(p("face_bbox_expand_top_ratio")),
                float(p("face_bbox_expand_bottom_ratio")),
            ),
        }
        self.gesture_model_options = {
            "model_path": str(p("gesture_model_path")),
            "num_hands": int(p("gesture_num_hands")),
            "max_width": int(p("gesture_max_width")),
            "min_detection_confidence": float(p("gesture_min_detection_confidence")),
            "min_presence_confidence": float(p("gesture_min_presence_confidence")),
            "min_tracking_confidence": float(p("gesture_min_tracking_confidence")),
        }
        self.gesture_engine_options = {
            "threshold": float(p("gesture_score_threshold")),
            "hold_seconds": float(p("gesture_hold_seconds")),
            "release_seconds": float(p("gesture_release_seconds")),
            "cooldown_seconds": float(p("gesture_cooldown_seconds")),
            "max_gap": float(p("gesture_max_gap")),
            "move_deadzone": float(p("gesture_move_deadzone")),
            "move_scale": float(p("gesture_move_scale")),
        }

        latest = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.face_detection_pub = self.create_publisher(
            Detection2DArray, str(p("face_detection_topic")), reliable
        )
        self.face_crop_pub = (
            self.create_publisher(Image, str(p("face_head_crop_topic")), latest)
            if bool(p("face_publish_head_crop"))
            else None
        )
        self.face_debug_pub = None
        self.face_debug_compressed_pub = None
        if bool(p("face_publish_debug_image")):
            self.face_debug_pub = self.create_publisher(
                Image, str(p("face_debug_image_topic")), latest
            )
            self.face_debug_compressed_pub = self.create_publisher(
                CompressedImage, str(p("face_debug_compressed_topic")), latest
            )

        self.gesture_state_pub = self.create_publisher(
            String, str(p("gesture_state_topic")), latest
        )
        self.gesture_events_pub = self.create_publisher(
            String, str(p("gesture_events_topic")), reliable
        )
        self.gesture_control_pub = self.create_publisher(
            String, str(p("gesture_palm_control_topic")), latest
        )
        self.gesture_detection_pub = self.create_publisher(
            Detection2DArray, str(p("gesture_detection_topic")), reliable
        )
        self.gesture_debug_pub = self.create_publisher(
            CompressedImage, str(p("gesture_debug_compressed_topic")), latest
        )
        self.status_pub = self.create_publisher(String, str(p("status_topic")), latched)

        self.mode = ModeState(OFF)
        self.last_mode_command = {
            "requested": OFF,
            "effective": OFF,
            "source": "startup",
            "phase": "",
            "reason": "startup",
        }
        self.face_backend = None
        self.gesture_backend = None
        self.last_header = None
        self.last_enqueued_at = 0.0
        self.last_fresh_at = time.monotonic()
        self.next_timeout_publish = 0.0
        self.next_error_log = 0.0
        self.next_status_publish = 0.0
        self.stats = {
            "received": 0,
            "processed": 0,
            "face_processed": 0,
            "gesture_processed": 0,
            "stale": 0,
            "errors": 0,
            "generation_discarded": 0,
            "last_inference_ms": None,
            "last_source_age_ms": None,
        }

        self.worker = _LatestWorker(self._process, self._on_worker_error, self._close_backends)
        input_type = CompressedImage if self.input_compressed else Image
        self.image_sub = self.create_subscription(
            input_type, self.image_topic, self._on_image, latest
        )
        self.mode_sub = self.create_subscription(
            String, str(p("mode_topic")), self._on_mode, latched
        )
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self._tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

        self._apply_mode(str(p("initial_mode")), source="initial_mode")
        self.get_logger().info(
            "Unified perception ready: input=%s, face=%s, gesture=%s, mode=%s"
            % (
                self.image_topic,
                self.face_enabled,
                self.gesture_enabled,
                self.mode.snapshot()[0],
            )
        )

    def _mode_allowed(self, requested):
        if requested == FACE and not self.face_enabled:
            return OFF, "face_disabled"
        if requested == GESTURE and not self.gesture_enabled:
            return OFF, "gesture_disabled"
        return requested, "accepted"

    def _on_mode(self, message):
        try:
            raw = json.loads(message.data) if message.data.lstrip().startswith("{") else message.data
            requested = normalize_mode(raw)
            source = raw.get("source", "mode_topic") if isinstance(raw, dict) else "mode_topic"
            phase = raw.get("phase", "") if isinstance(raw, dict) else ""
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.get_logger().error(f"Rejected perception mode command: {error}")
            self._publish_status(reason="invalid_mode_command")
            return
        self._apply_mode(requested, source=source, phase=phase)

    def _apply_mode(self, requested, source, phase=""):
        requested = normalize_mode(requested)
        effective, reason = self._mode_allowed(requested)
        previous, _ = self.mode.snapshot()
        mode, generation, changed = self.mode.switch(effective)
        self.last_mode_command = {
            "requested": requested,
            "effective": effective,
            "source": str(source),
            "phase": str(phase),
            "reason": reason,
        }
        if changed:
            self.worker.clear()
            self.last_enqueued_at = 0.0
            self.last_fresh_at = time.monotonic()
            if self.gesture_backend is not None:
                self.gesture_backend.reset()
            if previous == FACE:
                self._publish_empty_face()
            if previous == GESTURE:
                self._publish_inactive_gesture("mode_inactive")
            self.get_logger().info(
                f"Perception mode {previous} -> {mode}, generation={generation}, source={source}"
            )
        self._publish_status(
            reason=reason,
            requested=requested,
            source=source,
            changed=changed,
        )

    def _on_image(self, message):
        mode, generation = self.mode.snapshot()
        if mode == OFF:
            return
        now = time.monotonic()
        if now - self.last_enqueued_at < 0.90 / self.rates[mode]:
            return
        self.last_enqueued_at = now
        self.last_header = message.header
        stamp = message.header.stamp
        self.stats["received"] += 1
        self.worker.pending.put(
            _Frame(
                message=message,
                stamp_ns=stamp.sec * 1_000_000_000 + stamp.nanosec,
                received=now,
                mode=mode,
                generation=generation,
            )
        )

    def _process(self, job):
        mode, generation = self.mode.snapshot()
        if generation != job.generation or mode != job.mode:
            return {"valid": False, "reason": "generation_changed", "job": job}
        max_age = self.max_ages[job.mode]
        if (
            frame_age_seconds(job.stamp_ns, self.get_clock().now().nanoseconds, max_age) is None
            or time.monotonic() - job.received > max_age
        ):
            self.stats["stale"] += 1
            return {"valid": False, "reason": "stale_frame", "job": job}
        bgr = decode_image_message(job.message, self.input_compressed)
        mode, generation = self.mode.snapshot()
        if generation != job.generation or mode != job.mode:
            return {"valid": False, "reason": "generation_changed", "job": job}
        started = time.monotonic()
        if job.mode == FACE:
            if self.face_backend is None:
                self.face_backend = FaceBackend(**self.face_options)
            observation = self.face_backend.process(bgr)
        else:
            if self.gesture_backend is None:
                self.gesture_backend = GestureBackend(
                    self.gesture_model_options, self.gesture_engine_options
                )
            observation = self.gesture_backend.process(bgr)
        inference_ms = (time.monotonic() - started) * 1000.0
        self.stats["processed"] += 1
        self.stats["face_processed" if job.mode == FACE else "gesture_processed"] += 1
        return {
            "valid": True,
            "reason": "ok",
            "job": job,
            "bgr": bgr,
            "observation": observation,
            "inference_ms": inference_ms,
        }

    def _on_worker_error(self, job, error):
        self.stats["errors"] += 1
        now = time.monotonic()
        if now >= self.next_error_log:
            self.get_logger().error(f"{job.mode} perception failed: {error}")
            self.next_error_log = now + 2.0
        return {
            "valid": False,
            "reason": "inference_error",
            "job": job,
            "detail": str(error),
        }

    def _tick(self):
        wall = time.monotonic()
        result = self.worker.completed.take()
        if result is not None:
            job = result["job"]
            mode, generation = self.mode.snapshot()
            if mode != job.mode or generation != job.generation:
                self.stats["generation_discarded"] += 1
            elif result["valid"]:
                age = frame_age_seconds(
                    job.stamp_ns,
                    self.get_clock().now().nanoseconds,
                    self.max_ages[job.mode],
                )
                if age is None or wall - job.received > self.max_ages[job.mode]:
                    self.stats["stale"] += 1
                    self._publish_timeout(job.mode, "stale_frame")
                else:
                    self.last_fresh_at = wall
                    self.stats["last_inference_ms"] = result["inference_ms"]
                    self.stats["last_source_age_ms"] = age * 1000.0
                    if job.mode == FACE:
                        self._publish_face(result)
                    else:
                        self._publish_gesture(result)
            elif result["reason"] != "generation_changed":
                self._publish_timeout(job.mode, result["reason"])

        mode, _ = self.mode.snapshot()
        if (
            mode != OFF
            and wall - self.last_fresh_at >= self.watchdog_timeout
            and wall >= self.next_timeout_publish
        ):
            self._publish_timeout(mode, "camera_timeout")
            self.next_timeout_publish = wall + 0.10
        if wall >= self.next_status_publish:
            self._publish_status(reason="running")
            self.next_status_publish = wall + 1.0

    @staticmethod
    def _bgr_to_image(pixels, header):
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        output = Image()
        output.header = header
        output.height, output.width = pixels.shape[:2]
        output.encoding = "bgr8"
        output.is_bigendian = 0
        output.step = output.width * 3
        output.data = pixels.tobytes()
        return output

    @staticmethod
    def _send_json(publisher, value):
        message = String()
        message.data = json.dumps(value, allow_nan=False, separators=(",", ":"))
        publisher.publish(message)

    def _detection_array(self, header, boxes, kind):
        output = Detection2DArray()
        output.header = header
        for index, box in enumerate(boxes):
            detection = Detection2D()
            detection.header = header
            detection.id = "nearest_face" if kind == "face" else f"hand_{index}"
            detection.bbox.center.position.x = box.x + box.width / 2.0
            detection.bbox.center.position.y = box.y + box.height / 2.0
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = float(box.width)
            detection.bbox.size_y = float(box.height)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = kind if kind == "face" else box.label
            hypothesis.hypothesis.score = float(box.score)
            detection.results.append(hypothesis)
            output.detections.append(detection)
        return output

    def _publish_face(self, result):
        job, bgr, selected = result["job"], result["bgr"], result["observation"]
        boxes = [] if selected is None else [selected]
        self.face_detection_pub.publish(
            self._detection_array(job.message.header, boxes, "face")
        )
        if selected is not None and self.face_crop_pub is not None:
            crop = bgr[
                selected.y : selected.y + selected.height,
                selected.x : selected.x + selected.width,
            ]
            self.face_crop_pub.publish(self._bgr_to_image(crop, job.message.header))

        raw_wanted = self.face_debug_pub is not None and self.face_debug_pub.get_subscription_count() > 0
        jpeg_wanted = (
            self.face_debug_compressed_pub is not None
            and self.face_debug_compressed_pub.get_subscription_count() > 0
        )
        if not raw_wanted and not jpeg_wanted:
            return
        canvas = bgr.copy()
        if selected is not None:
            cv2.rectangle(
                canvas,
                (selected.x, selected.y),
                (selected.x + selected.width, selected.y + selected.height),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                canvas,
                f"nearest head crop {selected.score:.2f}",
                (selected.x, max(20, selected.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
        if raw_wanted:
            self.face_debug_pub.publish(self._bgr_to_image(canvas, job.message.header))
        if jpeg_wanted:
            self._publish_jpeg(self.face_debug_compressed_pub, canvas, job.message.header)

    def _publish_gesture(self, result):
        job, bgr, hands = result["job"], result["bgr"], result["observation"]
        now_ns = self.get_clock().now().nanoseconds
        state = self.gesture_backend.engine.process(
            hands,
            job.stamp_ns / 1e9,
            now_ns / 1e9,
            frame_size=(bgr.shape[1], bgr.shape[0]),
            max_age=self.max_ages[GESTURE],
        )
        mode, generation = self.mode.snapshot()
        state["mode"] = mode
        state["generation"] = generation
        state["stats"] = self._stats_snapshot()
        self._send_json(self.gesture_state_pub, state)
        self._send_json(
            self.gesture_control_pub,
            dict(
                state["control"],
                stamp=state["stamp"],
                hand_id=state["hand_id"],
                units="dimensionless",
                generation=generation,
            ),
        )
        for event in state["events"]:
            self._send_json(self.gesture_events_pub, dict(event, generation=generation))

        boxes = hand_boxes(
            hands, (bgr.shape[1], bgr.shape[0]), self.gesture_bbox_padding
        )
        self.gesture_detection_pub.publish(
            self._detection_array(job.message.header, boxes, "gesture")
        )
        if self.gesture_debug_pub.get_subscription_count() > 0:
            canvas = bgr.copy()
            for box in boxes:
                cv2.rectangle(
                    canvas,
                    (box.x, box.y),
                    (box.x + box.width, box.y + box.height),
                    (255, 180, 0),
                    2,
                )
                cv2.putText(
                    canvas,
                    f"{box.label} {box.score:.2f}",
                    (box.x, max(20, box.y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 180, 0),
                    2,
                )
            self._publish_jpeg(self.gesture_debug_pub, canvas, job.message.header)

    def _publish_jpeg(self, publisher, canvas, header):
        ok, encoded = cv2.imencode(
            ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            self.get_logger().error("Failed to encode perception debug JPEG")
            return
        message = CompressedImage()
        message.header = header
        message.format = "jpeg"
        message.data = encoded.tobytes()
        publisher.publish(message)

    def _publish_empty_face(self):
        output = Detection2DArray()
        if self.last_header is not None:
            output.header = self.last_header
        self.face_detection_pub.publish(output)

    def _empty_gesture_state(self, reason):
        mode, generation = self.mode.snapshot()
        return {
            "valid": False,
            "reason": reason,
            "stamp": self.get_clock().now().nanoseconds / 1e9,
            "hand_id": None,
            "label": "unknown",
            "score": 0.0,
            "source": "none",
            "center": None,
            "center_px": None,
            "pointing": None,
            "events": [],
            "control": {
                "active": False,
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "depth_source": "none",
                "reason": reason,
            },
            "mode": mode,
            "generation": generation,
            "stats": self._stats_snapshot(),
        }

    def _publish_inactive_gesture(self, reason):
        state = self._empty_gesture_state(reason)
        self._send_json(self.gesture_state_pub, state)
        self._send_json(
            self.gesture_control_pub,
            dict(
                state["control"],
                stamp=state["stamp"],
                hand_id=None,
                units="dimensionless",
                generation=state["generation"],
            ),
        )
        output = Detection2DArray()
        if self.last_header is not None:
            output.header = self.last_header
        self.gesture_detection_pub.publish(output)

    def _publish_timeout(self, mode, reason):
        if mode == FACE:
            self._publish_empty_face()
        elif mode == GESTURE:
            self._publish_inactive_gesture(reason)

    def _stats_snapshot(self):
        return dict(
            self.stats,
            dropped_pending=self.worker.pending.overwritten,
            dropped_completed=self.worker.completed.overwritten,
        )

    def _publish_status(self, reason, **extra):
        mode, generation = self.mode.snapshot()
        payload = {
            "mode": mode,
            "generation": generation,
            "reason": reason,
            "face_enabled": self.face_enabled,
            "gesture_enabled": self.gesture_enabled,
            "last_command": dict(self.last_mode_command),
            "stats": self._stats_snapshot(),
        }
        payload.update(extra)
        self._send_json(self.status_pub, payload)

    def _close_backends(self):
        if self.gesture_backend is not None:
            self.gesture_backend.close()

    def destroy_node(self):
        if hasattr(self, "worker"):
            self.worker.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GhostGamePerceptionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
