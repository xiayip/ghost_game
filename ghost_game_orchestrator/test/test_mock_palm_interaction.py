from types import SimpleNamespace
import threading

import ghost_game_orchestrator.ghost_game_node as game_node_module
from ghost_game_orchestrator.ghost_game_node import GhostGameNode


class FakeLogger:

    def info(self, _message):
        pass

    def error(self, _message, **_kwargs):
        pass


class DeferredThread:
    instances = []

    def __init__(self, target, daemon):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.instances.append(self)

    def start(self):
        self.started = True


def make_node():
    node = object.__new__(GhostGameNode)
    node._state_lock = threading.Lock()
    node._running = False
    node._phase = 'idle'
    node._abort_requested = False
    node._go_home_requested = False
    node._mock_solve_requested = True
    node._mock_solve_active = True
    node._mock_palm_transition_pending = False
    node._camera_ready = False
    node._cancel_active_trajectory = lambda: None
    node.get_logger = lambda: FakeLogger()
    return node


def test_mock_palm_service_starts_from_idle(monkeypatch):
    node = make_node()
    response = SimpleNamespace(success=None, message='')
    DeferredThread.instances.clear()
    monkeypatch.setattr(
        game_node_module, 'threading', SimpleNamespace(Thread=DeferredThread))

    result = node._on_mock_palm_interaction(None, response)

    assert result.success is True
    assert node._running is True
    assert node._camera_ready is False
    assert node._mock_solve_requested is False
    assert node._mock_solve_active is False
    assert len(DeferredThread.instances) == 1
    assert DeferredThread.instances[0].started is True
    assert DeferredThread.instances[0].daemon is True


def test_mock_palm_service_preempts_an_active_round(monkeypatch):
    node = make_node()
    node._running = True
    node._phase = 'searching'
    response = SimpleNamespace(success=None, message='')
    DeferredThread.instances.clear()
    monkeypatch.setattr(
        game_node_module, 'threading', SimpleNamespace(Thread=DeferredThread))

    result = node._on_mock_palm_interaction(None, response)

    assert result.success is True
    assert 'queued' in result.message
    assert node._abort_requested is True
    assert node._mock_palm_transition_pending is True
    assert len(DeferredThread.instances) == 1
    assert DeferredThread.instances[0].started is True


def test_mock_palm_runner_ignores_pipeline_terminal_state():
    node = make_node()
    node._running = True
    calls = []
    node._set_impedance_params = (
        lambda **kwargs: calls.append(('gravity', kwargs)) or True)
    node._move_to_success_pose = (
        lambda: calls.append(('success_pose', None)) or True)
    node._face_control_request = lambda: None
    node._run_palm_interaction = (
        lambda **kwargs: calls.append(('palm', kwargs)) or 'ok')

    node._start_mock_palm_interaction()

    assert calls == [
        ('gravity', {'gravity_compensation.factor': 1.0}),
        ('success_pose', None),
        ('palm', {'finish_on_mesh_ready': False}),
    ]
    assert node._phase == 'done'
    assert node._camera_ready is False
    assert node._running is False


def test_mock_palm_runner_aborts_if_success_pose_fails():
    node = make_node()
    node._running = True
    calls = []
    node._set_impedance_params = lambda **_kwargs: True
    node._move_to_success_pose = lambda: False
    node._face_control_request = lambda: None
    node._safe_abort = lambda: calls.append('abort')
    node._run_palm_interaction = lambda **_kwargs: calls.append('palm')

    node._start_mock_palm_interaction()

    assert calls == ['abort']
    assert node._running is False


def test_palm_interaction_opens_and_settles_gripper_before_start(monkeypatch):
    node = make_node()
    node.gripper_open_position = 0.0
    node.gripper_settle_time = 0.75
    calls = []
    node._open_gripper = lambda position: calls.append(('open', position))
    node._face_control_request = lambda: calls.append(('control', None))
    monkeypatch.setattr(
        game_node_module.time, 'sleep',
        lambda duration: calls.append(('settle', duration)))

    result = node._prepare_gripper_for_palm_interaction()

    assert result is None
    assert calls == [
        ('open', 0.0),
        ('settle', 0.75),
        ('control', None),
    ]
