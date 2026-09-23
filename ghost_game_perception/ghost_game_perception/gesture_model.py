"""MediaPipe Tasks adapter, imported lazily so core tests need no ML runtime."""
from pathlib import Path
import math
import time


DEFAULT_MODEL_PATH = "~/.local/share/ghost_game/gesture_recognizer.task"


def validate_model_file(model_path):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Gesture model not found: {path}. Run 'ros2 run ghost_game_perception "
            "download_gesture_model' or set model_path to a local .task bundle."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Gesture model is empty: {path}")
    return path


class GestureModel:
    """Single-thread owner. VIDEO mode reuses hand tracking between frames.

    ``recognize`` accepts BGR uint8 images and returns normalized image and
    hand-local world landmarks. World landmarks are NOT camera-frame depth.
    The source camera timestamp remains the caller's responsibility; an
    independent strictly increasing clock satisfies MediaPipe VIDEO ordering.
    """

    def __init__(
        self, model_path=DEFAULT_MODEL_PATH, num_hands=2, max_width=640,
        min_detection_confidence=.5, min_presence_confidence=.5,
        min_tracking_confidence=.5,
    ):
        path = validate_model_file(model_path)
        if isinstance(num_hands, bool) or not isinstance(num_hands, int) or num_hands < 1:
            raise ValueError("num_hands must be a positive integer")
        if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 64:
            raise ValueError("max_width must be an integer >= 64")
        for value in (min_detection_confidence, min_presence_confidence, min_tracking_confidence):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Model confidence thresholds must be in [0, 1]")
        try:
            import cv2
            import numpy as np
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "Gesture inference requires MediaPipe Tasks, NumPy and OpenCV in "
                "the same Python environment as ROS2. Install the dependencies "
                "described in GESTURE_INTERACTION.md for your target platform."
            ) from exc
        self._cv2, self._np, self._mp = cv2, np, mp
        self.max_width = max_width
        self._last_timestamp_ms = -1
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(path), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize(self, bgr):
        from .gesture_core import HandObservation

        if self._recognizer is None:
            raise RuntimeError("Gesture model is closed")
        if (getattr(bgr, "ndim", 0) != 3 or bgr.shape[2] != 3
                or bgr.shape[0] <= 0 or bgr.shape[1] <= 0
                or bgr.dtype != self._np.uint8):
            raise ValueError("recognize expects a non-empty uint8 BGR image")
        height, width = bgr.shape[:2]
        if width > self.max_width:
            bgr = self._cv2.resize(
                bgr, (self.max_width, max(1, round(height * self.max_width / width))),
                interpolation=self._cv2.INTER_AREA)
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(self._last_timestamp_ms + 1, time.perf_counter_ns() // 1_000_000)
        self._last_timestamp_ms = timestamp_ms
        result = self._recognizer.recognize_for_video(image, timestamp_ms)
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            categories = result.gestures[index] if index < len(result.gestures) else []
            category = max(categories, key=lambda item: item.score) if categories else None
            handed = result.handedness[index] if index < len(result.handedness) else []
            world = (result.hand_world_landmarks[index]
                     if index < len(result.hand_world_landmarks) else None)
            hands.append(HandObservation(
                landmarks=[(float(p.x), float(p.y), float(p.z)) for p in landmarks],
                label=category.category_name if category else "None",
                score=float(category.score) if category else 0.,
                handedness=handed[0].category_name if handed else "",
                world_landmarks=([(float(p.x), float(p.y), float(p.z)) for p in world]
                                 if world else None),
            ))
        return hands

    def close(self):
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
