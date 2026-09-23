from ghost_game_perception.gesture_core import HandObservation

from ghost_game_perception.backends import hand_boxes


def hand(points, label="Open_Palm", score=0.9):
    return HandObservation(points, label, score, "Right")


def test_hand_boxes_clip_landmarks_and_preserve_label():
    points = [(0.25, 0.20, 0.0)] * 21
    points[1] = (0.75, 0.80, 0.0)
    box = hand_boxes([hand(points)], (640, 480), padding_ratio=0.0)[0]
    assert (box.x, box.y, box.width, box.height) == (160, 96, 320, 288)
    assert box.label == "Open_Palm"
    assert box.score == 0.9


def test_hand_boxes_ignore_invalid_and_clip_to_frame():
    invalid = hand([(0.1, 0.1, 0.0)] * 2)
    points = [(-0.1, -0.1, 0.0)] * 21
    points[1] = (1.1, 1.1, 0.0)
    boxes = hand_boxes([invalid, hand(points)], (100, 50), padding_ratio=0.1)
    assert len(boxes) == 1
    assert (boxes[0].x, boxes[0].y, boxes[0].width, boxes[0].height) == (0, 0, 100, 50)
