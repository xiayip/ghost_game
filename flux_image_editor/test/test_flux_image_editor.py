import json
import time

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import String

from flux_image_editor.client import EditResponse, EditorClient, EditorError
from flux_image_editor.config import load_ros_parameters
from flux_image_editor.image_codec import image_message_to_png, png_to_messages
from flux_image_editor.ros_node import FluxImageEditorNode, validate_settings


def make_png(width=96, height=80, color=(30, 80, 190)):
    pixels = np.full((height, width, 3), color, dtype=np.uint8)
    ok, data = cv2.imencode('.png', pixels)
    assert ok
    return data.tobytes()


def make_image(pixels, encoding='bgr8', row_padding=0):
    pixels = np.asarray(pixels, dtype=np.uint8)
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    message = Image()
    message.height = pixels.shape[0]
    message.width = pixels.shape[1]
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = message.width * channels + row_padding
    if row_padding:
        rows = np.zeros((message.height, message.step), dtype=np.uint8)
        rows[:, :message.width * channels] = pixels.reshape(message.height, -1)
        message.data = rows.tobytes()
    else:
        message.data = np.ascontiguousarray(pixels).tobytes()
    return message


def test_yaml_and_validation():
    values = load_ros_parameters()
    assert values['server_url'] == 'http://127.0.0.1:8090'
    assert values['auto_submit_on_prompt'] is True
    assert not validate_settings(values)
    values['server_url'] = 'file:///tmp/socket'
    assert 'server_url' in validate_settings(values)


def test_ros_image_png_round_trip():
    pixels = np.zeros((80, 96, 3), dtype=np.uint8)
    pixels[:, :, 1] = 177
    source = make_image(pixels, encoding='bgr8', row_padding=7)
    source.header.frame_id = 'camera'
    png = image_message_to_png(source)
    raw, compressed, width, height = png_to_messages(png, source.header)
    assert png.startswith(b'\x89PNG')
    assert bytes(compressed.data) == png
    assert compressed.format == 'png'
    assert (width, height) == (96, 80)
    assert raw.header.frame_id == 'camera'
    decoded = np.frombuffer(raw.data, dtype=np.uint8).reshape(80, 96, 3)
    assert np.array_equal(decoded, pixels)


def test_http_client_parses_png_and_timing_headers():
    class Response:
        status_code = 200
        content = make_png()
        text = ''
        headers = {
            'content-type': 'image/png',
            'x-inference-sec': '1.234',
            'x-output-width': '96',
            'x-output-height': '80',
        }

    class Session:
        def post(self, url, **kwargs):
            assert url == 'http://127.0.0.1:8090/v1/edit'
            assert kwargs['files']['image'][2] == 'image/png'
            assert kwargs['data']['prompt'] == '赛博朋克风格'
            assert kwargs['data']['request_id'] == 'request-1'
            return Response()

        def close(self):
            pass

    client = EditorClient('http://127.0.0.1:8090/', 10, Session())
    result = client.edit(make_png(), '赛博朋克风格', 'request-1')
    assert result.width == 96
    assert result.height == 80
    assert result.inference_sec == pytest.approx(1.234)


def test_http_error_is_readable():
    class Response:
        status_code = 503
        content = b''
        text = '{"detail":"model is loading"}'
        headers = {'content-type': 'application/json'}

    class Session:
        def post(self, *_args, **_kwargs):
            return Response()

        def close(self):
            pass

    with pytest.raises(EditorError, match='503'):
        EditorClient('http://localhost:8090', 10, Session()).edit(make_png(), 'test')


def test_node_one_prompt_runs_once_and_publishes_both_outputs(monkeypatch, tmp_path):
    calls = []
    output_png = make_png(128, 96, (180, 40, 110))

    class StubClient:
        def __init__(self, base_url, timeout):
            assert base_url == 'http://127.0.0.1:8090'
            assert timeout == 300.0

        def edit(self, png, prompt, request_id):
            assert png.startswith(b'\x89PNG')
            assert len(request_id) == 32
            calls.append(prompt)
            return EditResponse(output_png, 2.5, 128, 96, 18.0, 42, request_id, 2.7)

        def close(self):
            pass

    monkeypatch.setattr('flux_image_editor.ros_node.EditorClient', StubClient)
    rclpy.init()
    node = None
    try:
        node = FluxImageEditorNode()
        assert node.set_parameters([
            Parameter('input_directory', value=str(tmp_path / 'inputs')),
            Parameter('output_directory', value=str(tmp_path)),
        ])[0].successful
        raw_messages = []
        png_messages = []
        status_messages = []
        node._raw_publisher = type('Publisher', (), {'publish': raw_messages.append})()
        node._png_publisher = type('Publisher', (), {'publish': png_messages.append})()
        node._status_publisher = type('Publisher', (), {'publish': status_messages.append})()

        image = make_image(np.zeros((120, 160, 3), dtype=np.uint8))
        image.header.frame_id = 'portrait_camera'
        node._on_image(image)
        node._on_prompt(String(data='将人物转换成赛博朋克风格'))
        node._worker.join(timeout=5)
        node._drain_updates()

        assert calls == ['将人物转换成赛博朋克风格']
        assert len(raw_messages) == 1
        assert len(png_messages) == 1
        assert bytes(png_messages[0].data) == output_png
        assert raw_messages[0].header.frame_id == 'portrait_camera'
        terminal = json.loads(status_messages[-1].data)
        assert terminal['status'] == 'success'
        assert terminal['input_path']
        assert terminal['latest_input_path']
        assert (tmp_path / 'inputs' / 'latest_head_crop.png').read_bytes().startswith(
            b'\x89PNG')
        assert terminal['inference_sec'] == pytest.approx(2.5)
        assert terminal['model_load_sec'] == pytest.approx(18.0)
        assert terminal['server_total_sec'] == pytest.approx(2.7)
        assert terminal['seed'] == 42
        assert terminal['server_request_id'] == terminal['request_id']
        assert terminal['pair_ready_at'].endswith('Z')
        assert terminal['request_sent_at'].endswith('Z')
        assert terminal['png_received_at'].endswith('Z')
        assert terminal['published_at'].endswith('Z')
        assert terminal['push_to_png_received_sec'] >= 0
        assert terminal['push_to_publish_sec'] >= terminal['push_to_png_received_sec']
        assert terminal['width'] == 128 and terminal['height'] == 96
        assert terminal['output_path']
        assert (tmp_path / (terminal['request_id'] + '.png')).read_bytes() == output_png

        # Camera frames only update the cache; they do not repeat a consumed prompt.
        node._on_image(image)
        time.sleep(0.05)
        assert calls == ['将人物转换成赛博朋克风格']
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
