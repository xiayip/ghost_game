"""Small, dependency-light helpers for sampling an aligned ROS depth image."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthSample:
    distance_m: float
    valid_count: int
    center_px: tuple
    radius_px: int


def match_depth_frame_stamps(
    stamps_ns,
    color_stamp_ns,
    now_ns,
    max_age_sec,
    max_skew_sec,
):
    """Select the buffered depth frame closest to one RGB frame.

    Returning the reason with the timing measurements keeps the ROS node's
    failure output useful without coupling this helper to ROS message types.
    """
    if not stamps_ns:
        return None, None, None, "no_depth_frame"
    if not all(
        math.isfinite(float(value)) and float(value) > 0
        for value in (color_stamp_ns, now_ns, max_age_sec, max_skew_sec)
    ):
        raise ValueError("depth matching times and limits must be positive")

    recent = []
    for index, stamp_ns in enumerate(stamps_ns):
        if not isinstance(stamp_ns, (int, float)) or stamp_ns <= 0:
            continue
        age_sec = (float(now_ns) - float(stamp_ns)) / 1e9
        if 0.0 <= age_sec <= float(max_age_sec):
            skew_sec = abs(float(stamp_ns) - float(color_stamp_ns)) / 1e9
            recent.append((skew_sec, -float(stamp_ns), index, age_sec))
    if not recent:
        return None, None, None, "stale_depth"

    skew_sec, _, index, age_sec = min(recent)
    if skew_sec > float(max_skew_sec):
        return None, age_sec, skew_sec, "rgb_depth_skew"
    return index, age_sec, skew_sec, "ok"


def _encoding_dtype(encoding, is_bigendian):
    encoding = str(encoding).lower()
    byte_order = ">" if is_bigendian else "<"
    if encoding in ("16uc1", "mono16"):
        return np.dtype(byte_order + "u2"), 0.001
    if encoding == "32fc1":
        return np.dtype(byte_order + "f4"), 1.0
    raise ValueError(f"unsupported depth encoding: {encoding}")


def _sample_depth_pixels(
    pixels,
    center_normalized,
    scale,
    radius_ratio=0.035,
    min_radius_px=4,
    max_radius_px=24,
    min_distance_m=0.15,
    max_distance_m=1.50,
    min_valid_pixels=8,
):
    """Return the median metric depth around a normalized palm center."""
    if not isinstance(pixels, np.ndarray) or pixels.ndim != 2:
        raise ValueError("depth pixels must be a two-dimensional array")
    height, width = pixels.shape
    if width <= 0 or height <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid depth image dimensions")
    if (
        not isinstance(center_normalized, (list, tuple))
        or len(center_normalized) != 2
        or not all(math.isfinite(float(value)) for value in center_normalized)
    ):
        raise ValueError("invalid normalized palm center")
    x_norm, y_norm = (float(value) for value in center_normalized)
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        raise ValueError("normalized palm center is outside the image")
    if not (0.0 < min_distance_m < max_distance_m):
        raise ValueError("invalid depth distance range")
    if min_valid_pixels <= 0:
        raise ValueError("min_valid_pixels must be positive")

    center_x = min(width - 1, max(0, int(round(x_norm * (width - 1)))))
    center_y = min(height - 1, max(0, int(round(y_norm * (height - 1)))))
    radius = int(round(min(width, height) * float(radius_ratio)))
    radius = min(int(max_radius_px), max(int(min_radius_px), radius))
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)

    roi = pixels[y0:y1, x0:x1].astype(np.float32) * float(scale)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    valid = roi[circle]
    valid = valid[
        np.isfinite(valid)
        & (valid >= float(min_distance_m))
        & (valid <= float(max_distance_m))
    ]
    if valid.size < int(min_valid_pixels):
        return None
    return DepthSample(
        distance_m=float(np.median(valid)),
        valid_count=int(valid.size),
        center_px=(center_x, center_y),
        radius_px=radius,
    )


def sample_aligned_depth(
    message,
    center_normalized,
    radius_ratio=0.035,
    min_radius_px=4,
    max_radius_px=24,
    min_distance_m=0.15,
    max_distance_m=1.50,
    min_valid_pixels=8,
):
    """Sample a raw depth Image registered to the RGB image."""
    width, height, step = int(message.width), int(message.height), int(message.step)
    dtype, scale = _encoding_dtype(message.encoding, bool(message.is_bigendian))
    bytes_per_pixel = dtype.itemsize
    if width <= 0 or height <= 0:
        raise ValueError("invalid depth image dimensions")
    if step < width * bytes_per_pixel or len(message.data) < step * height:
        raise ValueError("invalid depth image stride or buffer")
    pixels = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=message.data,
        strides=(step, bytes_per_pixel),
    )
    return _sample_depth_pixels(
        pixels,
        center_normalized,
        scale,
        radius_ratio,
        min_radius_px,
        max_radius_px,
        min_distance_m,
        max_distance_m,
        min_valid_pixels,
    )


def sample_compressed_depth(
    message,
    center_normalized,
    radius_ratio=0.035,
    min_radius_px=4,
    max_radius_px=24,
    min_distance_m=0.15,
    max_distance_m=1.50,
    min_valid_pixels=8,
):
    """Decode and sample a 16UC1 compressedDepth PNG message.

    compressed_depth_image_transport prepends a small transport header before
    the PNG payload. Finding the PNG signature keeps this compatible across
    plugin versions without depending on the C++ header layout.
    """
    image_format = str(message.format).lower()
    if "16uc1" not in image_format or "compresseddepth" not in image_format:
        raise ValueError(f"unsupported compressed depth format: {message.format}")
    payload = bytes(message.data)
    signature = b"\x89PNG\r\n\x1a\n"
    offset = payload.find(signature)
    if offset < 0:
        raise ValueError("compressed depth message contains no PNG payload")
    pixels = cv2.imdecode(
        np.frombuffer(payload[offset:], dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if pixels is None or pixels.ndim != 2 or pixels.dtype != np.uint16:
        raise ValueError("compressed depth PNG is not a 16-bit image")
    return _sample_depth_pixels(
        pixels,
        center_normalized,
        0.001,
        radius_ratio,
        min_radius_px,
        max_radius_px,
        min_distance_m,
        max_distance_m,
        min_valid_pixels,
    )
