import cv2
import numpy as np
from sensor_msgs.msg import Image

from img2doc.image_codec import encode_ros_image


def test_encode_ros_image_resizes_long_edge():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    message = Image()
    message.height = frame.shape[0]
    message.width = frame.shape[1]
    message.encoding = "bgr8"
    message.step = frame.shape[1] * frame.shape[2]
    message.data = frame.tobytes()
    result = encode_ros_image(
        message, max_long_edge=100, jpeg_quality=90, max_input_bytes=1_000_000
    )
    decoded = cv2.imdecode(np.frombuffer(result.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert (result.width, result.height) == (100, 50)
    assert decoded.shape[:2] == (50, 100)


def test_encode_rgb_image_with_padded_rows():
    # Verify ROS row padding and RGB-to-BGR conversion without cv_bridge.
    rgb = np.zeros((8, 16, 3), dtype=np.uint8)
    rgb[:, :8] = (255, 0, 0)
    rgb[:, 8:] = (0, 255, 0)
    padded = b"".join(row.tobytes() + b"\x63\x63" for row in rgb)
    message = Image()
    message.height = 8
    message.width = 16
    message.encoding = "rgb8"
    message.step = 16 * 3 + 2
    message.data = padded
    result = encode_ros_image(
        message, max_long_edge=100, jpeg_quality=100, max_input_bytes=1_000_000
    )
    decoded = cv2.imdecode(np.frombuffer(result.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded[4, 3, 2] > decoded[4, 3, 1]
    assert decoded[4, 12, 1] > decoded[4, 12, 2]
