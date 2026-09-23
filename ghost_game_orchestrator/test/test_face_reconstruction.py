import json
import threading

from std_msgs.msg import String

from ghost_game_orchestrator.ghost_game_node import GhostGameNode


class RecordingPublisher:

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeLogger:

    def info(self, _message):
        pass


def make_node():
    node = object.__new__(GhostGameNode)
    node.enable_face_reconstruction = True
    node.face_reconstruction_prompt = 'Preserve identity for 3D reconstruction.'
    node._reconstruction_lock = threading.Lock()
    node._reconstruction_waiting_for_request = False
    node._reconstruction_request_id = ''
    node._reconstruction_state = node._new_reconstruction_state()
    node._reconstruction_prompt_pub = RecordingPublisher()
    node._mesh_lock = threading.Lock()
    node._mesh_state = {
        'status': 'idle', 'progress': 0, 'request_id': '',
        'received_at': 0.0,
    }
    node.get_logger = lambda: FakeLogger()
    return node


def status_message(request_id, status, **extra):
    return String(data=json.dumps({
        'request_id': request_id,
        'status': status,
        **extra,
    }))


def test_submission_publishes_once_and_waits_for_accepted_request():
    node = make_node()

    node._submit_face_reconstruction()

    assert [message.data for message in node._reconstruction_prompt_pub.messages] == [
        'Preserve identity for 3D reconstruction.'
    ]
    assert node._reconstruction_state['status'] == 'submitted'
    assert node._reconstruction_waiting_for_request is True


def test_status_tracks_only_the_request_accepted_for_this_round():
    node = make_node()
    node._submit_face_reconstruction()

    # A terminal status from an older round must not claim this submission.
    node._face_reconstruction_status_cb(status_message('old', 'success'))
    assert node._reconstruction_state['status'] == 'submitted'

    node._face_reconstruction_status_cb(status_message('new', 'accepted'))
    node._face_reconstruction_status_cb(status_message(
        'new', 'success', output_path='/tmp/result.png', width=768,
        height=1024, total_sec=4.2))
    node._face_reconstruction_status_cb(status_message('other', 'error'))

    assert node._reconstruction_state == {
        'enabled': True,
        'status': 'success',
        'request_id': 'new',
        'input_path': '',
        'latest_input_path': '',
        'output_path': '/tmp/result.png',
        'width': 768,
        'height': 1024,
        'total_sec': 4.2,
        'error': '',
    }


def test_reset_exposes_enabled_idle_state():
    node = make_node()
    node._submit_face_reconstruction()

    node._reset_reconstruction_state()

    assert node._reconstruction_state['enabled'] is True
    assert node._reconstruction_state['status'] == 'idle'
    assert node._reconstruction_request_id == ''
