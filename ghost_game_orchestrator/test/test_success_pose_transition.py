import threading
from types import SimpleNamespace

import ghost_game_orchestrator.ghost_game_node as game_node_module
from ghost_game_orchestrator.ghost_game_node import GhostGameNode


def make_node():
    node = object.__new__(GhostGameNode)
    node.joints = [f'joint{i}' for i in range(1, 7)]
    node._state_lock = threading.Lock()
    node._phase = 'idle'
    node.success_stiffness = [1.0] * 6
    node.locked_damping = [0.5] * 6
    node.impedance_controller = 'impedance'
    node.impedance_trajectory_controller = 'impedance_jtc'
    node.position_adapter_controller = 'position_adapter'
    node.position_trajectory_controller = 'position_jtc'
    node.success_seed_time = 0.2
    node.success_time_from_start = 8.0
    node.success_positions = [1.0] * 6
    node.success_velocities = [0.0] * 6
    node.success_accelerations = [0.0] * 6
    node._impedance_traj_client = object()
    node.get_logger = lambda: SimpleNamespace(
        info=lambda _message: None,
        error=lambda _message: None,
    )
    return node


def test_success_pose_validates_live_state_without_waiting_for_action_result():
    node = make_node()
    calls = []
    node._ramp_stiffness = lambda stiffness, damping: calls.append(
        ('ramp', stiffness, damping))
    node._switch_controllers = lambda **kwargs: calls.append(
        ('switch', kwargs)) or True
    node._positions_snapshot = lambda: [0.0] * 6

    def send_trajectory(client, positions, times, **kwargs):
        calls.append(('trajectory', positions, times, kwargs))
        return True

    node._send_trajectory = send_trajectory
    node._wait_for_success_pose_settled = lambda: calls.append(
        ('validate',)) or True

    assert node._move_to_success_pose() is True

    trajectory_calls = [call for call in calls if call[0] == 'trajectory']
    assert trajectory_calls[0][3]['wait_result'] is True
    assert trajectory_calls[1][3]['wait_result'] is False
    assert calls[-1] == ('validate',)
    assert node._phase == 'success_move'


def test_success_validation_window_includes_nominal_motion(monkeypatch):
    node = make_node()
    node.success_validation_timeout = 6.0
    node.success_settle_time = 0.5
    node.success_settle_movement = 0.01
    node.success_position_tolerance = [0.05] * 6
    node._positions_snapshot = lambda: [0.0] * 6
    errors = []
    node.get_logger = lambda: SimpleNamespace(
        info=lambda _message: None,
        error=errors.append,
    )
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        game_node_module.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(
        game_node_module.time, 'sleep',
        lambda duration: setattr(clock, 'now', clock.now + duration))

    assert node._wait_for_success_pose_settled() is False

    assert clock.now >= 114.0
    assert clock.now < 114.1
    assert len(errors) == 1
