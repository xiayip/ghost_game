import numpy as np
from std_msgs.msg import Header

from ghost_game_perception.node import GhostGamePerceptionNode


def test_bgr_crop_message_is_contiguous_and_preserves_header():
    source = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    non_contiguous_crop = source[1:7:2, 2:9:2]
    header = Header(frame_id='camera_color_optical_frame')

    message = GhostGamePerceptionNode._bgr_to_image(non_contiguous_crop, header)

    assert message.header.frame_id == 'camera_color_optical_frame'
    assert message.encoding == 'bgr8'
    assert (message.height, message.width, message.step) == (3, 4, 12)
    decoded = np.frombuffer(message.data, dtype=np.uint8).reshape(3, 4, 3)
    assert np.array_equal(decoded, non_contiguous_crop)

