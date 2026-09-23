import numpy as np
import pytest

from ghost_game_orchestrator.palm_interaction import (
    CameraAxisServo,
    PalmDepthFilter,
    bounded_joint_servo_target,
    camera_pose_and_jacobian,
    palm_center_error_x,
)


ANCHOR = [-1.5137020349502563, .596736490726471, 1.2563692331314087,
          -.2887808680534363, .012655341997742653, -.02569492720067501]
LOWER = [-2.8, 0.0, 0.0, -1.57, -1.57, -3.14]
UPPER = [2.8, 3.14, 3.14, 1.57, 1.57, 3.14]


def payload(distance, hand_id="right:1", **extra):
    return {
        "active": True,
        "label": "open_palm",
        "distance_valid": True,
        "distance_m": distance,
        "hand_id": hand_id,
        **extra,
    }


def test_depth_filter_sign_deadzone_saturation_and_reacquisition():
    tracker = PalmDepthFilter(deadzone=.02, gain=1, max_offset=.08,
                              smoothing_time=.001, sample_timeout=.2)
    assert tracker.update(payload(.60), 0, 0) == 0
    # Farther hand -> camera/arm reaches forward (+ optical Z).
    assert tracker.update(payload(.70), .01, .01) == pytest.approx(.08, abs=1e-4)
    # A new hand cannot inherit the old hand's displacement.
    assert tracker.update(payload(.45, "left:2"), .02, .02) == 0
    assert tracker.update(payload(.46, "left:2"), .03, .03) == 0
    assert tracker.update(payload(.30, "left:2"), .04, .04) < 0
    # A stale sample freezes motion, but a quick return from the same hand
    # keeps the original neutral instead of silently zeroing the command.
    assert tracker.update(payload(.30, "left:2"), .04, .30) is None
    assert tracker.update(payload(.30, "left:2"), .31, .31) < 0


def test_brief_invalid_palm_holds_then_fails_closed():
    tracker = PalmDepthFilter()
    assert tracker.update(payload(.5), 0, 0) == 0
    bad = payload(.4)
    bad["active"] = False
    assert tracker.update(bad, .1, .1) == 0
    assert tracker.last_reason == "holding_open_palm_arming"
    assert tracker.update(bad, .21, .21) is None
    # Motion freezes immediately, while the same hand may reacquire its
    # original neutral point within the grace period.
    assert tracker.baseline == pytest.approx(.5)
    assert tracker.update(payload(.4), .22, .22) < 0


def test_landmark_palm_is_accepted_when_semantic_label_is_unknown():
    tracker = PalmDepthFilter(smoothing_time=.001)
    first = payload(.5)
    first.update(label="unknown", palm_open=True, palm_source="landmarks")
    moved = payload(.6)
    moved.update(label="unknown", palm_open=True, palm_source="landmarks")

    assert tracker.update(first, 0, 0) == 0
    assert tracker.update(moved, .1, .1) > 0


def test_neutral_is_forgotten_after_reacquisition_timeout():
    tracker = PalmDepthFilter(sample_timeout=.2, reacquire_timeout=.5)
    assert tracker.update(payload(.5), 0, 0) == 0
    bad = payload(.4)
    bad["distance_valid"] = False
    assert tracker.update(bad, .1, .1) == 0
    assert tracker.last_reason == "holding_invalid_depth"
    assert tracker.update(bad, .7, .7) is None
    assert tracker.baseline is None
    assert tracker.update(payload(.4), .71, .71) == 0
    assert tracker.baseline == pytest.approx(.4)


def test_camera_jacobian_matches_finite_difference_translation():
    position, _, jacobian = camera_pose_and_jacobian(ANCHOR)
    epsilon = 1e-6
    for index in range(6):
        shifted = list(ANCHOR)
        shifted[index] += epsilon
        moved, _, _ = camera_pose_and_jacobian(shifted)
        assert np.asarray((moved - position) / epsilon) == pytest.approx(
            jacobian[:3, index], abs=2e-6)


def test_servo_moves_in_requested_optical_direction_and_respects_bounds():
    servo = CameraAxisServo(
        ANCHOR, LOWER, UPPER, [.12, .35, .35, .35, .18, .18],
        max_linear_speed=.06, max_joint_speed=.35,
    )
    command = list(ANCHOR)
    start, rotation, _ = camera_pose_and_jacobian(command)
    for _ in range(150):
        step = servo.step(command, .08, .02)
        command = step.positions
    end, _, _ = camera_pose_and_jacobian(command)
    projection = float(np.dot(end - start, rotation[:, 2]))
    assert projection > .065
    assert all(abs(command[i] - ANCHOR[i]) <= [ .12, .35, .35, .35, .18, .18][i] + 1e-8
               for i in range(6))
    assert max(abs(value) for value in step.velocities) <= .35 + 1e-8

    command = list(ANCHOR)
    for _ in range(150):
        step = servo.step(command, -.08, .02)
        command = step.positions
    retreat, _, _ = camera_pose_and_jacobian(command)
    assert float(np.dot(retreat - start, rotation[:, 2])) < -.065


def test_palm_center_error_and_bounded_yaw_target():
    assert palm_center_error_x({"palm_open": True, "center": [.5, .4]}) == 0
    assert palm_center_error_x({"palm_open": True, "center": [.75, .4]}) == .5
    assert palm_center_error_x({"palm_open": False, "center": [.75, .4]}) is None
    target = 0.0
    for _ in range(100):
        target = bounded_joint_servo_target(
            target, 0.0, -1.57, 1.57, error=.8, gain=1.5,
            max_speed=.7, dt=.02, direction=-1.0, max_offset=.45,
            deadband=.08, joint_margin=.05,
        )
    assert target == pytest.approx(-.45)


def test_cartesian_servo_tracks_yawed_reference_and_depth_together():
    limits = [.12, .35, .35, .35, .45, .18]
    servo = CameraAxisServo(
        ANCHOR, LOWER, UPPER, limits,
        max_linear_speed=.20, max_joint_speed=.8,
    )
    reference = list(ANCHOR)
    reference[4] += .30
    command = list(ANCHOR)
    for _ in range(200):
        command = servo.step(
            command, .05, .01, reference=reference,
            controlled_joint_index=4).positions
    desired_origin, desired_rotation, _ = camera_pose_and_jacobian(reference)
    actual_position, actual_rotation, _ = camera_pose_and_jacobian(command)
    desired_position = desired_origin + desired_rotation[:, 2] * .05
    assert command[4] == pytest.approx(reference[4], abs=.01)
    assert np.linalg.norm(actual_position - desired_position) < .02
    assert np.linalg.norm(actual_rotation - desired_rotation) < .08
