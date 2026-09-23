import threading

from ghost_game_orchestrator.web_monitor import _ControlBridge


def test_control_bridge_round_trip():
    bridge = _ControlBridge()
    result = {}

    def request():
        result['value'] = bridge.request('start', timeout=1.0)

    thread = threading.Thread(target=request)
    thread.start()
    call = bridge.take(timeout=1.0)
    assert call.command == 'start'
    call.finish(200, True, 'round started')
    thread.join(timeout=1.0)

    assert result['value'] == (
        200,
        {'ok': True, 'command': 'start', 'message': 'round started'},
    )


def test_control_bridge_rejects_unknown_command():
    bridge = _ControlBridge()

    status, payload = bridge.request('dance', timeout=0.01)

    assert status == 404
    assert payload['ok'] is False
    assert payload['command'] == 'dance'


def _assert_control_bridge_accepts(command):
    bridge = _ControlBridge()
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault(
            'value', bridge.request(command, timeout=1.0)))
    thread.start()
    call = bridge.take(timeout=1.0)
    assert call.command == command
    call.finish(200, True, 'mock command accepted')
    thread.join(timeout=1.0)

    assert result['value'][0] == 200
    assert result['value'][1]['ok'] is True


def test_control_bridge_accepts_mock_solve_command():
    _assert_control_bridge_accepts('mock_solve')


def test_control_bridge_accepts_mock_palm_interaction_command():
    _assert_control_bridge_accepts('mock_palm_interaction')


def test_control_bridge_reports_timeout():
    bridge = _ControlBridge()

    status, payload = bridge.request('abort', timeout=0.01)
    expired_call = bridge.take(timeout=1.0)

    assert status == 504
    assert payload['ok'] is False
    assert 'timed out' in payload['message']
    assert expired_call.expired is True
