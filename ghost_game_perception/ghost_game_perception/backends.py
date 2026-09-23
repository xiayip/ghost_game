"""ROS-independent adapters for the existing YuNet and MediaPipe backends."""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from .face_core import (
    expand_face_box,
    scale_face_box,
    select_largest_face,
)
from .gesture_core import GestureEngine
from .gesture_model import GestureModel, validate_model_file


class FaceBackend:
    """Own one YuNet detector and return the expanded nearest-head box."""

    def __init__(
        self,
        model_path,
        score_threshold=0.70,
        nms_threshold=0.30,
        top_k=5000,
        input_width=424,
        expansion=(0.50, 0.50, 0.75, 0.50),
    ):
        if not 0.0 < score_threshold <= 1.0:
            raise ValueError("face_score_threshold must be in (0, 1]")
        if not 0.0 < nms_threshold <= 1.0:
            raise ValueError("face_nms_threshold must be in (0, 1]")
        if top_k <= 0 or input_width <= 0:
            raise ValueError("face_top_k and face_detector_input_width must be positive")
        if len(expansion) != 4 or any(
            not math.isfinite(value) or value < 0.0 for value in expansion
        ):
            raise ValueError("face bbox expansion must contain four non-negative values")

        default_model = (
            "yunet_2026may.onnx"
            if int(cv2.__version__.split(".")[0]) >= 5
            else "yunet_2023mar.onnx"
        )
        path = Path(str(model_path)).expanduser()
        if path.is_dir():
            path = path / default_model
        if not path.is_file():
            raise FileNotFoundError(f"YuNet model not found: {path}")

        self.model_path = path
        self.score_threshold = float(score_threshold)
        self.input_width = int(input_width)
        self.expansion = tuple(float(value) for value in expansion)
        self.detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), self.score_threshold,
            float(nms_threshold), int(top_k)
        )
        self.detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

    def process(self, bgr):
        height, width = bgr.shape[:2]
        if width > self.input_width:
            detector_width = self.input_width
            detector_height = max(1, round(height * detector_width / width))
            detector_image = cv2.resize(
                bgr, (detector_width, detector_height), interpolation=cv2.INTER_AREA
            )
        else:
            detector_image = bgr
            detector_height, detector_width = height, width
        self.detector.setInputSize((detector_width, detector_height))
        _, faces = self.detector.detect(detector_image)
        face_box = select_largest_face(
            faces, detector_image.shape, self.score_threshold
        )
        face_box = scale_face_box(
            face_box,
            bgr.shape,
            width / detector_width,
            height / detector_height,
        )
        return (
            expand_face_box(face_box, bgr.shape, *self.expansion)
            if face_box is not None
            else None
        )


class GestureBackend:
    """Own one MediaPipe recognizer plus the project's temporal intent engine."""

    def __init__(self, model_options, engine_options):
        validate_model_file(model_options["model_path"])
        self.model = GestureModel(**model_options)
        self.engine = GestureEngine(**engine_options)

    def process(self, bgr):
        return self.model.recognize(bgr)

    def reset(self):
        self.engine.reset()

    def close(self):
        self.model.close()


@dataclass(frozen=True)
class HandBox:
    x: int
    y: int
    width: int
    height: int
    label: str
    score: float
    handedness: str


def hand_boxes(hands, frame_size, padding_ratio=0.08):
    """Convert normalized MediaPipe landmarks into clipped pixel boxes."""
    width, height = frame_size
    if width <= 0 or height <= 0 or not math.isfinite(padding_ratio) or padding_ratio < 0:
        raise ValueError("invalid frame size or hand bbox padding")
    boxes = []
    for hand in hands:
        points = getattr(hand, "landmarks", None)
        if not points or len(points) != 21:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        if not all(math.isfinite(value) for value in xs + ys):
            continue
        left = min(xs) * width
        right = max(xs) * width
        top = min(ys) * height
        bottom = max(ys) * height
        padding = max(right - left, bottom - top) * padding_ratio
        left = max(0, math.floor(left - padding))
        top = max(0, math.floor(top - padding))
        right = min(width, math.ceil(right + padding))
        bottom = min(height, math.ceil(bottom + padding))
        if right <= left or bottom <= top:
            continue
        boxes.append(
            HandBox(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                label=str(getattr(hand, "label", "hand") or "hand"),
                score=float(getattr(hand, "score", 0.0)),
                handedness=str(getattr(hand, "handedness", "")),
            )
        )
    return boxes
