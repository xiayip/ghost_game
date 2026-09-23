"""ROS-independent face-box validation and nearest-face selection."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def area(self):
        return self.width * self.height


def scale_face_box(box, source_shape, scale_x, scale_y):
    """Map a detector-space box into source-image pixel coordinates."""
    if box is None:
        return None
    if (not math.isfinite(scale_x) or not math.isfinite(scale_y) or
            scale_x <= 0.0 or scale_y <= 0.0):
        raise ValueError('face-box scale factors must be finite and positive')
    height, width = source_shape[:2]
    left = max(0, int(math.floor(box.x * scale_x)))
    top = max(0, int(math.floor(box.y * scale_y)))
    right = min(
        width, int(math.ceil((box.x + box.width) * scale_x)))
    bottom = min(
        height, int(math.ceil((box.y + box.height) * scale_y)))
    return FaceBox(
        x=left,
        y=top,
        width=max(0, right - left),
        height=max(0, bottom - top),
        score=box.score,
    )


def expand_face_box(
        box, image_shape, left_ratio=0.0, right_ratio=0.0,
        top_ratio=0.0, bottom_ratio=0.0):
    """Expand a detected face into a head crop, clipped to the image.

    Ratios are relative to the detected face width/height. Keeping this step
    separate from face selection prevents padding and image-edge clipping
    from changing which person is considered nearest.
    """
    ratios = (left_ratio, right_ratio, top_ratio, bottom_ratio)
    if any(not math.isfinite(value) or value < 0.0 for value in ratios):
        raise ValueError('face-box expansion ratios must be finite and non-negative')

    image_height, image_width = image_shape[:2]
    left = max(0, int(math.floor(box.x - box.width * left_ratio)))
    top = max(0, int(math.floor(box.y - box.height * top_ratio)))
    right = min(
        image_width,
        int(math.ceil(box.x + box.width + box.width * right_ratio)))
    bottom = min(
        image_height,
        int(math.ceil(box.y + box.height + box.height * bottom_ratio)))

    return FaceBox(
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
        score=box.score,
    )


def select_largest_face(faces, image_shape, score_threshold=0.70):
    """Return the largest valid visible face box, or ``None``.

    With RGB only, projected face area is a distance proxy, not a metric
    depth measurement. Every frame is evaluated independently. Boxes are
    clipped to the actual image before comparison and publication.
    """
    height, width = image_shape[:2]
    candidates = []
    for face in [] if faces is None else faces:
        values = np.asarray(face, dtype=float).reshape(-1)
        if values.size < 15 or not np.isfinite(values[:15]).all():
            continue
        x, y, box_width, box_height = values[:4]
        score = float(values[14])
        if box_width <= 0.0 or box_height <= 0.0 or score < score_threshold:
            continue

        left = max(0, int(math.floor(x)))
        top = max(0, int(math.floor(y)))
        right = min(width, int(math.ceil(x + box_width)))
        bottom = min(height, int(math.ceil(y + box_height)))
        if right <= left or bottom <= top:
            continue
        candidates.append(FaceBox(
            x=left, y=top, width=right - left, height=bottom - top,
            score=score))

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda box: (box.area, box.score, -box.x, -box.y))
