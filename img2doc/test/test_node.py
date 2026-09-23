from types import SimpleNamespace
import threading

from img2doc.node import Img2DocNode


class FakeLogger:
    def info(self, _message):
        pass


def test_auto_submit_does_not_reenter_image_lock():
    node = object.__new__(Img2DocNode)
    node._lock = threading.Lock()
    node._latest_image = None
    node._image_count = 0
    node._last_auto_submit = 0.0
    node._values = {
        "auto_submit": True,
        "auto_interval_sec": 0.01,
        "log_each_image": False,
    }
    node.get_logger = lambda: FakeLogger()
    calls = []

    def start_request(source):
        assert node._lock.acquire(blocking=False)
        node._lock.release()
        calls.append(source)
        return True, "accepted"

    node._start_request = start_request
    message = SimpleNamespace(width=640, height=480, encoding="bgr8")

    node._on_image(message)

    assert calls == ["auto"]
    assert node._latest_image is message
