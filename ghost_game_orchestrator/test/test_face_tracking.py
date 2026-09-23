import pytest

from ghost_game_orchestrator.face_tracking import (
    FaceStabilityGate,
    face_matches_track,
    face_is_acceptable,
    make_face_sample,
    scan_target,
    servo_target,
)


def sample(
        received_at=10.0, center_x=320.0, center_y=240.0,
        width=160.0, height=160.0):
    return make_face_sample(
        received_at, center_x, center_y, width, height,
        640, 480, 0.9)


def test_face_sample_reports_normalized_center_error_and_area():
    target = sample(center_x=480.0, center_y=120.0)

    assert target.error_x == pytest.approx(0.5)
    assert target.error_y == pytest.approx(-0.5)
    assert target.area_ratio == pytest.approx(1.0 / 12.0)


def test_far_or_stale_face_is_rejected():
    far = sample(width=30.0, height=30.0)
    close = sample()

    assert not face_is_acceptable(far, 10.1, 0.5, 0.015, 0.65)
    assert not face_is_acceptable(close, 11.0, 0.5, 0.015, 0.65)
    assert face_is_acceptable(close, 10.1, 0.5, 0.015, 0.65)


def test_stability_gate_requires_dwell_and_resets_after_jump():
    gate = FaceStabilityGate(2.0, 0.08, 0.35)
    centered = sample()

    assert not gate.update(centered, 10.0)
    assert not gate.update(centered, 11.9)
    assert gate.update(centered, 12.0)

    jumped = sample(center_x=400.0)
    assert not gate.update(jumped, 12.1)
    assert gate.stable_for == 0.0


def test_face_track_rejects_detector_identity_switch():
    target = sample(center_x=240.0, center_y=220.0, width=150.0, height=170.0)
    correction = sample(
        center_x=270.0, center_y=225.0, width=155.0, height=175.0)
    different_person = sample(
        center_x=500.0, center_y=100.0, width=80.0, height=90.0)

    assert face_matches_track(target, correction, 0.25, 0.75)
    assert not face_matches_track(target, different_person, 0.25, 0.75)


def test_scan_target_offsets_only_camera_yaw_and_pitch_joints():
    reference = [0.0] * 6
    target = scan_target(
        reference, [-1.0] * 6, [1.0] * 6,
        yaw_index=4, pitch_index=3,
        yaw_offset=0.3, pitch_offset=-0.2)

    assert target == [0.0, 0.0, 0.0, -0.2, 0.3, 0.0]


def test_servo_target_integrates_velocity_at_control_period():
    target = servo_target(
        current_command=[0.0] * 6,
        reference=[0.0] * 6,
        lower_limits=[-1.0] * 6,
        upper_limits=[1.0] * 6,
        yaw_index=4,
        pitch_index=3,
        error_x=0.8,
        error_y=-0.5,
        yaw_gain=0.35,
        pitch_gain=0.35,
        max_speed=0.25,
        dt=0.04,
        yaw_direction=-1.0,
        pitch_direction=-1.0,
        yaw_max_offset=0.5,
        pitch_max_offset=0.3,
        deadband=0.08,
    )

    assert target[4] == pytest.approx(-0.25 * 0.04)
    assert target[3] == pytest.approx(0.35 * 0.5 * 0.04)


def test_servo_target_holds_deadband_axis_and_clamps_local_limit():
    target = servo_target(
        current_command=[0.0, 0.0, 0.0, 0.0, 0.499, 0.0],
        reference=[0.0] * 6,
        lower_limits=[-1.0] * 6,
        upper_limits=[1.0] * 6,
        yaw_index=4,
        pitch_index=3,
        error_x=-1.0,
        error_y=0.05,
        yaw_gain=1.0,
        pitch_gain=1.0,
        max_speed=0.25,
        dt=0.1,
        yaw_direction=-1.0,
        pitch_direction=-1.0,
        yaw_max_offset=0.5,
        pitch_max_offset=0.3,
        deadband=0.08,
    )

    assert target[4] == pytest.approx(0.5)
    assert target[3] == pytest.approx(0.0)
