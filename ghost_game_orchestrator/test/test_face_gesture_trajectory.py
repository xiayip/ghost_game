import pytest

from ghost_game_orchestrator.ghost_game_node import GhostGameNode


def make_node():
    node = object.__new__(GhostGameNode)
    node.joints = [f'joint{i}' for i in range(1, 7)]
    node.joint_lower_limits = [-2.8, 0.0, 0.0, -1.57, -1.57, -3.14]
    node.joint_upper_limits = [2.8, 3.14, 3.14, 1.57, 1.57, 3.14]
    node.face_yaw_index = 4
    node.face_pitch_index = 3
    node.face_gesture_roll_index = 5
    node.face_joint_margin = 0.05
    node.face_orbit_pitch_amplitude = 0.22
    node.face_orbit_yaw_amplitude = 0.36
    node.face_orbit_max_joint_speed = 1.20
    node.face_orbit_min_motion_time = 0.18
    node.face_gesture_pitch_amplitude = 0.18
    node.face_gesture_roll_amplitude = 0.30
    node.face_gesture_max_joint_speed = 0.80
    node.face_gesture_nod_max_joint_speed = 1.25
    node.face_gesture_nod_min_motion_time = 0.18
    node.face_gesture_min_motion_time = 0.32
    node.face_gesture_trajectory_rate_hz = 40.0
    return node


def test_face_gesture_is_one_dense_bounded_trajectory_returning_to_anchor():
    node = make_node()
    anchor = [0.0] * 6

    points, times, velocities = node._build_face_inspection_trajectory(anchor)

    assert len(points) == len(times) == len(velocities)
    assert len(points) > 200
    assert all(after > before for before, after in zip(times, times[1:]))
    assert points[0] == anchor
    assert points[-1] == pytest.approx(anchor)
    assert max(point[4] for point in points) == pytest.approx(0.36, abs=1e-3)
    assert min(point[4] for point in points) == pytest.approx(-0.36, abs=1e-3)
    assert max(point[3] for point in points) >= 0.219
    assert min(point[3] for point in points) <= -0.219
    assert max(point[5] for point in points) == pytest.approx(0.30)
    assert max(abs(value) for velocity in velocities for value in velocity) <= 1.251
    assert times[-1] < 10.0
