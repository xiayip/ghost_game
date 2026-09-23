import unittest

import numpy as np

from ghost_game_perception.face_core import (
    FaceBox,
    expand_face_box,
    scale_face_box,
    select_largest_face,
)


def face(x, y, width, height, score=0.95):
    output = np.zeros(15, dtype=float)
    output[:4] = [x, y, width, height]
    output[14] = score
    return output


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.shape = (480, 640, 3)

    def test_selects_largest_visible_box(self):
        small = face(20, 30, 80, 80)
        large = face(300, 100, 120, 100)
        self.assertEqual(
            select_largest_face([small, large], self.shape),
            FaceBox(300, 100, 120, 100, 0.95))

    def test_clips_box_to_image(self):
        selected = select_largest_face(
            [face(-10, -20, 100, 80)], self.shape)
        self.assertEqual(selected, FaceBox(0, 0, 90, 60, 0.95))

    def test_rejects_low_score_invalid_and_offscreen_boxes(self):
        invalid = face(10, 10, -20, 30)
        low_score = face(10, 10, 200, 200, 0.5)
        offscreen = face(800, 10, 100, 100)
        self.assertIsNone(select_largest_face(
            [invalid, low_score, offscreen], self.shape))

    def test_none_or_empty_input_returns_none(self):
        self.assertIsNone(select_largest_face(None, self.shape))
        self.assertIsNone(select_largest_face([], self.shape))

    def test_tie_break_is_deterministic(self):
        right = face(300, 100, 100, 100, 0.95)
        left = face(20, 100, 100, 100, 0.95)
        self.assertEqual(
            select_largest_face([right, left], self.shape).x, 20)


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.shape = (480, 640, 3)
        self.face = FaceBox(100, 100, 100, 80, 0.95)

    def test_expands_face_into_asymmetric_head_crop(self):
        expanded = expand_face_box(
            self.face, self.shape,
            left_ratio=0.50,
            right_ratio=0.50,
            top_ratio=0.75,
            bottom_ratio=0.50,
        )
        self.assertEqual(expanded, FaceBox(50, 40, 200, 180, 0.95))

    def test_expansion_is_clipped_to_image(self):
        edge_face = FaceBox(5, 10, 100, 80, 0.95)
        expanded = expand_face_box(
            edge_face, self.shape,
            left_ratio=0.50,
            right_ratio=0.50,
            top_ratio=0.75,
            bottom_ratio=0.50,
        )
        self.assertEqual(expanded, FaceBox(0, 0, 155, 130, 0.95))

    def test_zero_expansion_preserves_face_box(self):
        self.assertEqual(
            expand_face_box(self.face, self.shape),
            self.face,
        )

    def test_invalid_expansion_is_rejected(self):
        with self.assertRaises(ValueError):
            expand_face_box(self.face, self.shape, left_ratio=-0.1)


class ScalingTests(unittest.TestCase):
    def test_maps_detector_box_back_to_source_pixels(self):
        detector_box = FaceBox(50, 25, 100, 80, 0.95)

        self.assertEqual(
            scale_face_box(detector_box, (530, 848, 3), 2.0, 2.0),
            FaceBox(100, 50, 200, 160, 0.95),
        )

    def test_none_is_preserved_and_invalid_scale_is_rejected(self):
        shape = (480, 640, 3)
        box = FaceBox(100, 100, 100, 80, 0.95)
        self.assertIsNone(scale_face_box(None, shape, 2.0, 2.0))
        with self.assertRaises(ValueError):
            scale_face_box(box, shape, 0.0, 2.0)


if __name__ == '__main__':
    unittest.main()
