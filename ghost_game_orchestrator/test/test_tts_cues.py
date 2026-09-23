from types import SimpleNamespace
import threading
import time

from ghost_game_orchestrator.ghost_game_node import GhostGameNode


class RecordingPublisher:

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeStopClient:

    def __init__(self, ready):
        self.ready = ready
        self.requests = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return SimpleNamespace()


def make_node(tts_enabled=True, stop_ready=False):
    node = object.__new__(GhostGameNode)
    node.tts_enabled = tts_enabled
    node._tts_pub = RecordingPublisher()
    node._tts_stop_client = FakeStopClient(stop_ready)
    node._wait_for_future = lambda future, timeout: True
    return node


def eventually(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError('timed out waiting for speech helper')


def test_speak_strips_and_publishes_plain_text():
    node = make_node()

    node._speak('  意识端口已经接入。  ')

    assert [message.data for message in node._tts_pub.messages] == [
        '意识端口已经接入。'
    ]


def test_disabled_or_empty_speech_is_ignored():
    disabled = make_node(tts_enabled=False)
    enabled = make_node()

    disabled._speak('不会播放')
    enabled._speak('   ')

    assert disabled._tts_pub.messages == []
    assert enabled._tts_pub.messages == []


def test_interrupt_cancels_queue_before_new_announcement():
    node = make_node(stop_ready=True)
    release_wait = threading.Event()
    waits = []

    def wait_for_stop(future, timeout):
        waits.append(timeout)
        release_wait.wait(timeout=1.0)

    node._wait_for_future = wait_for_stop

    node._interrupt_speech('安全协议已启动。')

    assert len(node._tts_stop_client.requests) == 1
    eventually(lambda: waits == [0.75])
    assert node._tts_pub.messages == []
    release_wait.set()
    eventually(lambda: len(node._tts_pub.messages) == 1)
    assert waits == [0.75]
    assert [message.data for message in node._tts_pub.messages] == [
        '安全协议已启动。'
    ]


def test_interrupt_still_announces_when_stop_service_is_starting():
    node = make_node(stop_ready=False)

    node._interrupt_speech('返回初始位置。')

    assert node._tts_stop_client.requests == []
    assert [message.data for message in node._tts_pub.messages] == [
        '返回初始位置。'
    ]
