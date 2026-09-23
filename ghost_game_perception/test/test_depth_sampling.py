import numpy as np
import pytest
import cv2
from sensor_msgs.msg import CompressedImage, Image

from ghost_game_perception.depth_sampling import (
    match_depth_frame_stamps,
    sample_aligned_depth,
    sample_compressed_depth,
)


def depth_message(values, encoding="16UC1", pad=0):
    values = np.asarray(values)
    message = Image()
    message.height, message.width = values.shape
    message.encoding = encoding
    message.is_bigendian = 0
    row_bytes = values.dtype.itemsize * message.width
    message.step = row_bytes + pad
    rows = bytearray(message.step * message.height)
    for row in range(message.height):
        start = row * message.step
        rows[start : start + row_bytes] = values[row].tobytes()
    message.data = bytes(rows)
    return message


def test_uint16_depth_uses_meters_median_and_ignores_invalid_pixels():
    pixels = np.full((40, 60), 720, dtype=np.uint16)
    pixels[18:23, 28:33] = 510
    pixels[20, 30] = 0
    result = sample_aligned_depth(
        depth_message(pixels, pad=8), (.5, .5), radius_ratio=.08,
        min_radius_px=3, max_radius_px=3, min_valid_pixels=5,
    )
    assert result.distance_m == pytest.approx(.510)
    assert result.valid_count >= 5
    assert result.center_px == (30, 20)


def test_float_depth_and_big_endian_are_supported():
    floats = np.full((20, 20), .82, dtype=np.float32)
    assert sample_aligned_depth(
        depth_message(floats, encoding="32FC1"), (.5, .5),
        min_radius_px=2, max_radius_px=2,
    ).distance_m == pytest.approx(.82)

    values = np.full((20, 20), 640, dtype=">u2")
    message = depth_message(values)
    message.is_bigendian = 1
    assert sample_aligned_depth(
        message, (.5, .5), min_radius_px=2, max_radius_px=2,
    ).distance_m == pytest.approx(.64)


def test_insufficient_or_out_of_range_depth_returns_none():
    pixels = np.zeros((20, 20), dtype=np.uint16)
    pixels[10, 10] = 600
    assert sample_aligned_depth(
        depth_message(pixels), (.5, .5), min_radius_px=2,
        max_radius_px=2, min_valid_pixels=2,
    ) is None


def test_bad_encoding_and_center_are_rejected():
    message = depth_message(np.ones((20, 20), dtype=np.uint16))
    message.encoding = "8UC1"
    with pytest.raises(ValueError, match="unsupported depth encoding"):
        sample_aligned_depth(message, (.5, .5))
    message.encoding = "16UC1"
    with pytest.raises(ValueError, match="outside"):
        sample_aligned_depth(message, (1.1, .5))


def test_compressed_depth_png_uses_transport_header_and_millimeters():
    pixels = np.full((40, 60), 740, dtype=np.uint16)
    pixels[18:23, 28:33] = 525
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    message = CompressedImage()
    message.format = "16UC1; compressedDepth png"
    message.data = b"transport123" + encoded.tobytes()
    result = sample_compressed_depth(
        message, (.5, .5), radius_ratio=.08,
        min_radius_px=3, max_radius_px=3, min_valid_pixels=5,
    )
    assert result.distance_m == pytest.approx(.525)
    assert result.center_px == (30, 20)


def test_compressed_depth_rejects_wrong_format_and_missing_png():
    message = CompressedImage()
    message.format = "jpeg"
    message.data = b"not-a-depth-image"
    with pytest.raises(ValueError, match="unsupported compressed depth"):
        sample_compressed_depth(message, (.5, .5))
    message.format = "16UC1; compressedDepth png"
    with pytest.raises(ValueError, match="no PNG"):
        sample_compressed_depth(message, (.5, .5))


def test_depth_buffer_selects_frame_closest_to_rgb_not_latest_arrival():
    stamps = [9_900_000_000, 10_000_000_000, 10_100_000_000]
    index, age, skew, reason = match_depth_frame_stamps(
        stamps, color_stamp_ns=10_015_000_000,
        now_ns=10_150_000_000, max_age_sec=.3, max_skew_sec=.1,
    )
    assert index == 1
    assert age == pytest.approx(.15)
    assert skew == pytest.approx(.015)
    assert reason == "ok"


def test_depth_buffer_reports_stale_and_skewed_frames():
    assert match_depth_frame_stamps(
        [9_000_000_000], 10_000_000_000, 10_100_000_000, .3, .1,
    )[-1] == "stale_depth"
    index, age, skew, reason = match_depth_frame_stamps(
        [9_900_000_000], 10_100_000_000, 10_150_000_000, .3, .1,
    )
    assert index is None
    assert age == pytest.approx(.25)
    assert skew == pytest.approx(.2)
    assert reason == "rgb_depth_skew"
