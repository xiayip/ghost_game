"""Convert between ROS Image messages and exact PNG bytes."""

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image


def image_message_to_bgr(message):
    """Decode common ROS 8-bit images without the cv_bridge NumPy ABI."""
    encoding = message.encoding.lower()
    channels_by_encoding = {
        'mono8': 1,
        '8uc1': 1,
        'bgr8': 3,
        'rgb8': 3,
        'bgra8': 4,
        'rgba8': 4,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f'不支持的图像编码 {message.encoding}')
    minimum_step = int(message.width) * channels
    if message.width <= 0 or message.height <= 0 or message.step < minimum_step:
        raise ValueError(
            f'无效图像尺寸/步长: {message.width}x{message.height} step={message.step}')
    expected_bytes = int(message.step) * int(message.height)
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < expected_bytes:
        raise ValueError(f'图像数据被截断: {raw.size} < {expected_bytes}')
    rows = raw[:expected_bytes].reshape(int(message.height), int(message.step))
    pixels = rows[:, :minimum_step].reshape(
        int(message.height), int(message.width), channels)
    if encoding in ('mono8', '8uc1'):
        return cv2.cvtColor(pixels[:, :, 0], cv2.COLOR_GRAY2BGR)
    if encoding == 'rgb8':
        return np.ascontiguousarray(pixels[:, :, ::-1])
    if encoding == 'bgra8':
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if encoding == 'rgba8':
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(pixels)


def bgr_to_image_message(pixels, header):
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.size == 0:
        raise ValueError('推理服务返回的图像不是有效 BGR 图像')
    raw = Image()
    raw.header = header
    raw.height = pixels.shape[0]
    raw.width = pixels.shape[1]
    raw.encoding = 'bgr8'
    raw.is_bigendian = 0
    raw.step = pixels.shape[1] * 3
    raw.data = pixels.tobytes()
    return raw


def image_message_to_png(message, max_bytes=20_000_000):
    pixels = image_message_to_bgr(message)
    if pixels is None or pixels.size == 0:
        raise ValueError('收到空图像')
    ok, encoded = cv2.imencode('.png', pixels)
    if not ok:
        raise ValueError('PNG 编码失败')
    data = encoded.tobytes()
    if len(data) > max_bytes:
        raise ValueError(f'输入 PNG 超过限制: {len(data)} > {max_bytes} 字节')
    return data


def png_to_messages(png, header):
    pixels = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or pixels.size == 0:
        raise ValueError('推理服务返回的内容不是有效 PNG')
    raw = bgr_to_image_message(pixels, header)
    compressed = CompressedImage()
    compressed.header = header
    compressed.format = 'png'
    compressed.data = png
    return raw, compressed, pixels.shape[1], pixels.shape[0]
