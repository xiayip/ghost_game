from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sensor_msgs.msg import Image


@dataclass(frozen=True)
class EncodedImage:
    data: bytes
    width: int
    height: int


class ImageEncodingError(RuntimeError):
    pass


def _image_to_bgr(message: Image) -> np.ndarray:
    """Decode the 8-bit ROS image formats emitted by FLUX without cv_bridge.

    The container currently uses NumPy 2 while the Jazzy cv_bridge extension
    was built against NumPy 1. Importing it can segfault. Reading the explicit
    Image layout also keeps this small cloud-upload node independent of that
    binary ABI.
    """
    encoding = str(message.encoding).lower()
    channels_by_encoding = {
        "bgr8": 3,
        "8uc3": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ImageEncodingError(f"不支持的 ROS 图像编码: {message.encoding}")
    width = int(message.width)
    height = int(message.height)
    if width <= 0 or height <= 0:
        raise ImageEncodingError("ROS 图像尺寸无效")
    row_bytes = width * channels
    step = int(message.step) or row_bytes
    if step < row_bytes:
        raise ImageEncodingError("ROS 图像 step 小于单行像素大小")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise ImageEncodingError(
            f"ROS 图像数据不完整: {raw.size} < {required} 字节")
    rows = raw[:required].reshape(height, step)[:, :row_bytes]
    if channels == 1:
        frame = rows.reshape(height, width)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = rows.reshape(height, width, channels)
    if encoding in ("rgb8",):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(frame)


def encode_ros_image(
    message: Image,
    *,
    max_long_edge: int,
    jpeg_quality: int,
    max_input_bytes: int,
) -> EncodedImage:
    try:
        frame = _image_to_bgr(message)
    except ImageEncodingError:
        raise
    except Exception as exc:  # defensive boundary around OpenCV/NumPy decoding
        raise ImageEncodingError(f"ROS 图像转换失败: {exc}") from exc

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ImageEncodingError("ROS 图像为空")

    height, width = frame.shape[:2]
    longest = max(width, height)
    if max_long_edge > 0 and longest > max_long_edge:
        scale = max_long_edge / float(longest)
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        raise ImageEncodingError("JPEG 编码失败")
    data = encoded.tobytes()
    if len(data) > max_input_bytes:
        raise ImageEncodingError(
            f"JPEG 大小 {len(data)} 字节超过上限 {max_input_bytes} 字节"
        )
    return EncodedImage(data=data, width=width, height=height)
