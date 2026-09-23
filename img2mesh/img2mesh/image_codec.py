"""Convert common 8-bit ROS images without the cv_bridge NumPy ABI."""

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image


_CHANNELS = {
    'mono8': 1,
    '8uc1': 1,
    'bgr8': 3,
    'rgb8': 3,
    'bgra8': 4,
    'rgba8': 4,
}


def image_message_to_array(message):
    """Return contiguous OpenCV channel order while respecting row padding."""
    encoding = message.encoding.lower()
    channels = _CHANNELS.get(encoding)
    if channels is None:
        raise ValueError(f'不支持的图像编码 {message.encoding}')
    width = int(message.width)
    height = int(message.height)
    minimum_step = width * channels
    if width <= 0 or height <= 0 or int(message.step) < minimum_step:
        raise ValueError(
            f'无效图像尺寸/步长: {width}x{height} step={message.step}')
    expected_bytes = int(message.step) * height
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < expected_bytes:
        raise ValueError(f'图像数据被截断: {raw.size} < {expected_bytes}')
    rows = raw[:expected_bytes].reshape(height, int(message.step))
    pixels = rows[:, :minimum_step].reshape(height, width, channels)
    if encoding in ('mono8', '8uc1'):
        return np.ascontiguousarray(pixels[:, :, 0])
    if encoding == 'rgb8':
        return np.ascontiguousarray(pixels[:, :, ::-1])
    if encoding == 'rgba8':
        return np.ascontiguousarray(pixels[:, :, [2, 1, 0, 3]])
    return np.ascontiguousarray(pixels)


def array_to_image_message(pixels, encoding):
    """Build a sensor_msgs/Image from a contiguous uint8 NumPy array."""
    encoding = encoding.lower()
    channels = _CHANNELS.get(encoding)
    if channels is None:
        raise ValueError(f'不支持的图像编码 {encoding}')
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if channels == 1:
        if pixels.ndim != 2:
            raise ValueError('mono8 图像必须是二维数组')
        height, width = pixels.shape
    else:
        if pixels.ndim != 3 or pixels.shape[2] != channels:
            raise ValueError(f'{encoding} 图像必须有 {channels} 个通道')
        height, width = pixels.shape[:2]
    message = Image()
    message.height = height
    message.width = width
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = width * channels
    message.data = pixels.tobytes()
    return message


def image_to_png(message, max_bytes=20_000_000):
    pixels = image_message_to_array(message)
    if pixels.size == 0:
        raise ValueError('收到空图像')
    encoded, png = cv2.imencode('.png', pixels)
    if not encoded:
        raise ValueError('PNG 编码失败')
    data = png.tobytes()
    if len(data) > max_bytes:
        raise ValueError(f'PNG 超过 Tripo 单图 20 MB 限制: {len(data)} 字节')
    return data


def compressed_image_to_png(message, max_bytes=20_000_000):
    """Return the exact PNG payload from a CompressedImage after validation.

    FLUX publishes its generated PNG on a compressed topic. Passing those
    bytes through unchanged makes the image uploaded to Tripo auditable and
    avoids a decode/re-encode round trip. Non-PNG compressed images are
    decoded and normalized to PNG for compatibility with the Tripo uploader.
    """
    if not isinstance(message, CompressedImage):
        raise ValueError('期望 sensor_msgs/msg/CompressedImage')
    data = bytes(message.data)
    if not data:
        raise ValueError('收到空压缩图像')
    if len(data) > max_bytes:
        raise ValueError(f'压缩图像超过 Tripo 单图 20 MB 限制: {len(data)} 字节')
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.size == 0:
        raise ValueError(f'无法解码压缩图像: {message.format!r}')
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return data
    encoded, png = cv2.imencode('.png', pixels)
    if not encoded:
        raise ValueError('压缩图像转 PNG 失败')
    normalized = png.tobytes()
    if len(normalized) > max_bytes:
        raise ValueError(
            f'PNG 超过 Tripo 单图 20 MB 限制: {len(normalized)} 字节')
    return normalized
