from ghost_game_perception.gesture_core import HandObservation

from ghost_game_perception.backends import hand_boxes, select_largest_hand


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


def test_select_largest_hand_uses_visible_bbox_area():
    small_points = [(0.40, 0.40, 0.0)] * 21
    small_points[1] = (0.60, 0.60, 0.0)
    small = hand(small_points, label="small")
    large_points = [(0.15, 0.20, 0.0)] * 21
    large_points[1] = (0.80, 0.85, 0.0)
    large = hand(large_points, label="large")

    assert select_largest_hand([small, large], (640, 480), 0.0) is large


def test_select_largest_hand_ignores_invalid_detections():
    invalid = hand([(0.1, 0.1, 0.0)] * 2)
    valid_points = [(0.25, 0.25, 0.0)] * 21
    valid_points[1] = (0.75, 0.75, 0.0)
    valid = hand(valid_points)

    assert select_largest_hand([invalid, valid], (320, 240)) is valid
    assert select_largest_hand([invalid], (320, 240)) is None
