import json
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ghost_game_interfaces.msg import MeshResult
from flux_image_editor.client import EditResponse, EditorError

from img2mesh.image_codec import array_to_image_message, image_to_png
from img2mesh.config import load_common_parameters, load_local_api_key
from img2mesh.node import Img2MeshNode, validate_settings
from img2mesh.tripo_client import TripoClient


def test_ros_image_is_encoded_as_png():
    image = np.zeros((260, 300, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    message = array_to_image_message(image, 'bgr8')
    png = image_to_png(message)
    assert png.startswith(b'\x89PNG\r\n\x1a\n')
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, image)


def test_png_keeps_alpha_channel():
    image = np.zeros((260, 300, 4), dtype=np.uint8)
    image[:, :, 3] = 128
    message = array_to_image_message(image, 'bgra8')
    png = image_to_png(message)
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (260, 300, 4)
    assert np.array_equal(decoded, image)


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(('POST', url, kwargs))
        if url.endswith('/files/presign'):
            return FakeResponse({'code': 0, 'data': {
                'presigned_url': 'https://storage.example/upload', 'file_token': 'file_123',
            }})
        return FakeResponse({'code': 0, 'data': {'task_id': 'task_123'}})

    def put(self, url, **kwargs):
        self.calls.append(('PUT', url, kwargs))
        return FakeResponse(status_code=200)

    def get(self, url, **kwargs):
        self.calls.append(('GET', url, kwargs))
        return FakeResponse({'code': 0, 'data': {
            'status': 'success', 'progress': 100,
            'output': {'model_url': 'https://cdn.example/model.glb'},
        }})


def test_tripo_upload_submit_and_poll():
    session = FakeSession()
    client = TripoClient('test-key', 'https://api.example/v3', 5.0, session=session)
    token = client.upload_png(b'PNG-BYTES')
    task_id = client.submit_image(token, 'P1-20260311', 5000, True, True, False)
    output = client.wait_for_task(task_id, 1.0, 5.0, threading.Event(), lambda *_: None)
    assert output['model_url'] == 'https://cdn.example/model.glb'
    assert [call[0] for call in session.calls] == ['POST', 'PUT', 'POST', 'GET']
    assert session.calls[1][2]['data'] == b'PNG-BYTES'
    assert session.calls[2][2]['json']['input'] == 'file_123'
    assert session.calls[2][2]['json']['model'] == 'P1-20260311'


def test_invalid_mode_and_p1_quad_rejected():
    values = Img2MeshNode.PARAM_DEFAULTS.copy()
    assert values['face_limit'] == 8_000
    values['model'] = 'P1-20260311'
    values['mode'] = 'bad'
    assert 'mode' in validate_settings(values)
    values['mode'] = 'single'
    values['quad'] = True
    assert 'quad' in validate_settings(values)
    values['quad'] = False


def test_yaml_values_and_private_key_can_be_edited(monkeypatch, tmp_path):
    values = load_common_parameters()
    assert values['api_base_url'] == 'https://openapi.tripo3d.ai/v3'
    values.update({'model': 'P2-20260801', 'face_limit': 24000,
                   'image_topic': '/camera/front/image_raw'})
    (tmp_path / 'img2mesh.yaml').write_text(
        yaml.safe_dump({'img2mesh': {'ros__parameters': values}}), encoding='utf-8',
    )
    (tmp_path / 'local_api.yaml').write_text(
        yaml.safe_dump({'tripo': {'api_key': 'local-test-key'}}), encoding='utf-8',
    )
    monkeypatch.setenv('IMG2MESH_CONFIG_DIR', str(tmp_path))
    loaded = load_common_parameters()
    assert loaded['model'] == 'P2-20260801'
    assert loaded['face_limit'] == 24000
    assert loaded['image_topic'] == '/camera/front/image_raw'
    assert not validate_settings({'api_key': '', **loaded})
    assert load_local_api_key() == 'local-test-key'


def test_node_submits_once_and_publishes_model_url(monkeypatch):
    class StubSession:
        def close(self):
            pass

    class StubClient:
        def __init__(self, *_):
            self.session = StubSession()

        def upload_png(self, png):
            assert png.startswith(b'\x89PNG')
            return 'file_123'

        def submit_image(self, *_):
            return 'task_123'

        def wait_for_task(self, _task_id, _interval, _timeout, _stop, callback):
            callback('running', 50)
            return {'model_url': 'https://cdn.example/head.glb',
                    'rendered_image_url': 'https://cdn.example/preview.png'}

    monkeypatch.setattr('img2mesh.node.TripoClient', StubClient)
    rclpy.init()
    node = None
    try:
        node = Img2MeshNode()
        assert node.set_parameters([Parameter('api_key', value='test-key')])[0].successful
        message = array_to_image_message(
            np.zeros((260, 300, 3), dtype=np.uint8), 'bgr8')
        node._on_image(message)
        published = []
        statuses = []
        model_urls = []
        node._publisher = type('PublisherStub', (), {'publish': published.append})()
        node._status_publisher = type(
            'PublisherStub', (), {'publish': statuses.append})()
        node._model_url_publisher = type(
            'PublisherStub', (), {'publish': model_urls.append})()
        accepted, _ = node._try_submit()
        assert accepted
        node._worker.join(timeout=5)
        node._drain_updates()
        assert [item.status for item in published] == [
            'uploading', 'queued', 'running', 'success',
        ]
        assert published[-1].model_url == 'https://cdn.example/head.glb'
        assert model_urls[-1].data == 'https://cdn.example/head.glb'
        final_status = json.loads(statuses[-1].data)
        assert final_status['status'] == 'success'
        assert final_status['model_ready'] is True
        assert 'cdn.example' not in statuses[-1].data
        assert published[-1].task_id == 'task_123'
        assert node.set_parameters([Parameter('mode', value='multi')])[0].successful
        accepted, message = node._try_submit()
        assert not accepted and '尚未实现' in message
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


def test_non_style_auto_submit_runs_once_per_new_image(monkeypatch):
    submitted = []

    class StubSession:
        def close(self):
            pass

    class StubClient:
        def __init__(self, *_):
            self.session = StubSession()

        def upload_png(self, _png):
            return 'file_123'

        def submit_image(self, *_):
            submitted.append('mesh')
            return f'task_{len(submitted)}'

        def wait_for_task(self, *_):
            return {'model_url': 'https://cdn.example/result.glb'}

    monkeypatch.setattr('img2mesh.node.TripoClient', StubClient)
    defaults = Img2MeshNode.PARAM_DEFAULTS.copy()
    defaults.update({
        'api_key': 'test-key',
        'auto_submit': True,
        'style_enabled': False,
    })
    monkeypatch.setattr(Img2MeshNode, 'PARAM_DEFAULTS', defaults)
    rclpy.init()
    node = None
    try:
        node = Img2MeshNode()
        image = array_to_image_message(
            np.zeros((260, 300, 3), dtype=np.uint8), 'bgr8')
        node._on_image(image)
        node._worker.join(timeout=5)
        node._retry_pending_auto_submission()
        assert submitted == ['mesh']

        node._on_image(image)
        node._worker.join(timeout=5)
        assert submitted == ['mesh', 'mesh']
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


def test_local_api_key_is_used_without_export_or_ros_parameter(monkeypatch):
    keys = []

    class StubSession:
        def close(self):
            pass

    class StubClient:
        def __init__(self, key, *_):
            keys.append(key)
            self.session = StubSession()

        def upload_png(self, _png):
            return 'file_123'

        def submit_image(self, *_):
            return 'task_123'

        def wait_for_task(self, *_):
            return {'model_url': 'https://cdn.example/model.glb'}

    monkeypatch.setattr('img2mesh.node.TripoClient', StubClient)
    monkeypatch.setattr('img2mesh.node.load_local_api_key', lambda: 'local-test-key')
    monkeypatch.delenv('TRIPO_API_KEY', raising=False)
    rclpy.init()
    node = None
    try:
        node = Img2MeshNode()
        assert node.get_parameter('api_key').value == ''
        node._on_image(array_to_image_message(
            np.zeros((260, 300, 3), dtype=np.uint8), 'bgr8'))
        accepted, _ = node._try_submit()
        assert accepted
        node._worker.join(timeout=5)
        assert keys == ['local-test-key']
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


def test_ros_topics_and_submit_service(monkeypatch):
    class StubSession:
        def close(self):
            pass

    class StubClient:
        def __init__(self, *_):
            self.session = StubSession()

        def upload_png(self, _png):
            return 'file_123'

        def submit_image(self, *_):
            return 'task_123'

        def wait_for_task(self, _task_id, _interval, _timeout, _stop, callback):
            callback('running', 70)
            return {'model_url': 'https://cdn.example/result.glb'}

    monkeypatch.setattr('img2mesh.node.TripoClient', StubClient)
    rclpy.init()
    bridge = driver = executor = None
    try:
        bridge = Img2MeshNode()
        bridge.set_parameters([Parameter('api_key', value='test-key')])
        driver = Node('img2mesh_test_driver')
        publisher = driver.create_publisher(Image, '/sensor_msgs/image_raw', 10)
        client = driver.create_client(Trigger, '/img2mesh/submit')
        results = []
        statuses = []
        model_urls = []
        driver.create_subscription(
            MeshResult, '/img2mesh/mesh_result', results.append, 10,
        )
        driver.create_subscription(
            String, '/img2mesh/status', statuses.append, 10)
        driver.create_subscription(
            String, '/img2mesh/model_url', model_urls.append, 10)
        executor = SingleThreadedExecutor()
        executor.add_node(bridge)
        executor.add_node(driver)
        image = array_to_image_message(
            np.zeros((260, 300, 3), dtype=np.uint8), 'bgr8')
        deadline = time.monotonic() + 5
        while bridge._latest_image is None and time.monotonic() < deadline:
            publisher.publish(image)
            executor.spin_once(timeout_sec=0.1)
        assert bridge._latest_image is not None
        assert client.wait_for_service(timeout_sec=1)
        future = client.call_async(Trigger.Request())
        while not future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert future.done() and future.result().success
        while (not any(item.status == 'success' for item in results)
               or not model_urls
               or not any(json.loads(item.data).get('model_ready')
                          for item in statuses)) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert results[-1].model_url == 'https://cdn.example/result.glb'
        assert model_urls[-1].data == 'https://cdn.example/result.glb'
        assert json.loads(statuses[-1].data)['model_ready'] is True
        assert 'cdn.example' not in statuses[-1].data
    finally:
        if executor is not None:
            executor.shutdown()
        if driver is not None:
            driver.destroy_node()
        if bridge is not None:
            bridge.destroy_node()
        rclpy.try_shutdown()


def test_file_mode_reads_path_without_image_subscription(monkeypatch, tmp_path):
    image_path = tmp_path / 'portrait.jpg'
    assert cv2.imwrite(str(image_path), np.zeros((260, 300, 3), dtype=np.uint8))

    class StubSession:
        def close(self):
            pass

    class StubClient:
        def __init__(self, *_):
            self.session = StubSession()

        def upload_png(self, png):
            assert png.startswith(b'\x89PNG')
            return 'file_123'

        def submit_image(self, *_):
            return 'task_123'

        def wait_for_task(self, _task_id, _interval, _timeout, _stop, _callback):
            return {'model_url': 'https://cdn.example/file-result.glb'}

    monkeypatch.setattr('img2mesh.node.TripoClient', StubClient)
    settings = Img2MeshNode.PARAM_DEFAULTS.copy()
    settings.update({
        'test_mode': True,
        'test_image_path': str(image_path),
        'test_exit_on_complete': False,
        'api_key': 'test-key',
    })
    monkeypatch.setattr(Img2MeshNode, 'PARAM_DEFAULTS', settings)
    rclpy.init()
    node = None
    try:
        node = Img2MeshNode()
        assert node._subscriber is None
        published = []
        node._publisher = type('PublisherStub', (), {'publish': published.append})()
        node._start_file_test()
        node._worker.join(timeout=5)
        node._drain_updates()
        assert published[-1].status == 'success'
        assert published[-1].model_url == 'https://cdn.example/file-result.glb'
        assert not node._test_failed
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


def test_style_prompt_topic_and_image_topic_to_mesh_result(monkeypatch, tmp_path):
    calls = []

    class StubSession:
        def close(self):
            pass

    class StubClient:
        def __init__(self, *_):
            self.session = StubSession()

        def upload_png(self, png):
            assert png == b'\x89PNG\r\n\x1a\nLOCAL-STYLE'
            calls.append(('upload', 'styled_png'))
            return 'file_123'

        def wait_for_task(self, task_id, _interval, _timeout, _stop, callback):
            callback('running', 50)
            assert task_id == 'task_mesh'
            return {'model_url': 'https://cdn.example/styled.glb'}

        def submit_image(self, image_input, *_):
            calls.append(('mesh', image_input))
            return 'task_mesh'

    class StubStyleClient:
        def __init__(self, server_url, timeout):
            assert server_url == 'http://127.0.0.1:8090'
            assert timeout == 600.0

        def ready(self):
            return True

        def edit(self, png, prompt, request_id):
            assert png.startswith(b'\x89PNG')
            calls.append(('style', prompt, request_id))
            return EditResponse(
                png=b'\x89PNG\r\n\x1a\nLOCAL-STYLE', inference_sec=0.7,
                width=256, height=384, server_request_id='local_style_request',
                server_total_sec=0.72,
            )

        def close(self):
            pass

    monkeypatch.setattr('img2mesh.node.TripoClient', StubClient)
    monkeypatch.setattr('img2mesh.node.EditorClient', StubStyleClient)
    rclpy.init()
    bridge = driver = executor = None
    try:
        bridge = Img2MeshNode()
        assert bridge.set_parameters([
            Parameter('api_key', value='test-key'),
            Parameter('style_enabled', value=True),
            Parameter('style_output_directory', value=str(tmp_path)),
        ])[0].successful
        driver = Node('img2mesh_style_test_driver')
        image_pub = driver.create_publisher(Image, '/sensor_msgs/image_raw', 10)
        prompt_pub = driver.create_publisher(String, '/img2mesh/style_prompt', 10)
        client = driver.create_client(Trigger, '/img2mesh/submit')
        results = []
        driver.create_subscription(MeshResult, '/img2mesh/mesh_result', results.append, 10)
        executor = SingleThreadedExecutor()
        executor.add_node(bridge)
        executor.add_node(driver)
        deadline = time.monotonic() + 5
        while (image_pub.get_subscription_count() == 0
               or prompt_pub.get_subscription_count() == 0) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        image = array_to_image_message(
            np.zeros((260, 300, 3), dtype=np.uint8), 'bgr8')
        image.header.frame_id = 'style_test_camera'
        image_pub.publish(image)
        while bridge._latest_image is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert bridge._latest_image is not None
        assert client.wait_for_service(timeout_sec=1)
        missing_prompt = client.call_async(Trigger.Request())
        while not missing_prompt.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert missing_prompt.done()
        assert not missing_prompt.result().success
        assert not calls

        prompt_pub.publish(String(data='戴上太阳镜'))
        while bridge._latest_prompt != '戴上太阳镜' and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert bridge._latest_prompt == '戴上太阳镜'
        submitted = client.call_async(Trigger.Request())
        while not submitted.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert submitted.done() and submitted.result().success
        bridge._worker.join(timeout=5)
        while not any(item.status == 'success' for item in results) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert calls[0][0:2] == ('style', '戴上太阳镜')
        assert calls[1:] == [('upload', 'styled_png'), ('mesh', 'file_123')]
        assert any(item.status == 'styling' for item in results)
        assert results[-1].style_task_id == 'local_style_request'
        style_path = Path(urlparse(results[-1].style_image_url).path)
        assert style_path.parent == tmp_path
        assert style_path.read_bytes() == b'\x89PNG\r\n\x1a\nLOCAL-STYLE'
        assert results[-1].task_id == 'task_mesh'
        assert results[-1].model_url == 'https://cdn.example/styled.glb'
        assert results[-1].source_header.frame_id == 'style_test_camera'
    finally:
        if executor is not None:
            executor.shutdown()
        if driver is not None:
            driver.destroy_node()
        if bridge is not None:
            bridge.destroy_node()
        rclpy.try_shutdown()


def test_style_launch_auto_submits_once_per_prompt(monkeypatch):
    calls = []

    class StubSession:
        def close(self):
            pass

    class StubClient:
        def __init__(self, *_):
            self.session = StubSession()

        def upload_png(self, _png):
            calls.append('upload')
            return 'file_123'

        def submit_image(self, image_input, model, face_limit, *_):
            assert image_input == 'file_123'
            assert model == 'P1-20260311'
            assert face_limit == 8_000
            calls.append('mesh')
            return 'mesh_task'

        def wait_for_task(self, _task_id, _interval, _timeout, _stop, _callback):
            return {'model_url': 'https://cdn.example/styled.glb'}

    class StubStyleClient:
        def __init__(self, *_):
            pass

        def ready(self):
            return True

        def edit(self, _png, prompt, request_id):
            calls.append(('style', prompt))
            return EditResponse(
                png=b'\x89PNG\r\n\x1a\nLOCAL-STYLE', inference_sec=0.7,
                width=256, height=384, server_request_id=request_id,
                server_total_sec=0.72,
            )

        def close(self):
            pass

    monkeypatch.setattr('img2mesh.node.TripoClient', StubClient)
    monkeypatch.setattr('img2mesh.node.EditorClient', StubStyleClient)
    defaults = Img2MeshNode.PARAM_DEFAULTS.copy()
    defaults.update({'api_key': 'test-key', 'style_enabled': True,
                     'style_save_output': False, 'auto_submit': True})
    monkeypatch.setattr(Img2MeshNode, 'PARAM_DEFAULTS', defaults)
    rclpy.init()
    node = None
    try:
        node = Img2MeshNode()
        info_logs = []
        logger = type('LoggerStub', (), {
            'info': lambda _self, message: info_logs.append(message),
            'warning': lambda _self, _message: None,
            'error': lambda _self, _message: None,
        })()
        monkeypatch.setattr(node, 'get_logger', lambda: logger)
        results = []
        node._publisher = type('PublisherStub', (), {'publish': results.append})()
        image = array_to_image_message(
            np.zeros((260, 300, 3), dtype=np.uint8), 'bgr8')
        node._on_prompt(String(data='赛博朋克风'))
        assert calls == []  # Prompt alone is not enough.
        node._on_image(image)
        node._worker.join(timeout=5)
        node._drain_updates()
        assert calls == [('style', '赛博朋克风'), 'upload', 'mesh']
        assert results[-1].status == 'success'
        assert any('已收到本地风格化 PNG' in item
                   and 'local_style_total_sec=' in item for item in info_logs)

        for _ in range(4):
            node._on_image(image)
            node._retry_pending_auto_submission()
        assert calls == [('style', '赛博朋克风'), 'upload', 'mesh']

        node._on_prompt(String(data='赛博朋克风'))  # A new publication is a new request.
        node._worker.join(timeout=5)
        node._drain_updates()
        assert calls == [
            ('style', '赛博朋克风'), 'upload', 'mesh',
            ('style', '赛博朋克风'), 'upload', 'mesh',
        ]
        assert sum('收到 /sensor_msgs/image_raw' in item for item in info_logs) == 1
        assert sum('收到 /img2mesh/style_prompt' in item for item in info_logs) == 2
        assert sum('开始本地 FLUX 图像编辑' in item for item in info_logs) == 2
        assert sum('三维生成任务已提交' in item for item in info_logs) == 2
        assert sum('已收到模型 URL' in item for item in info_logs) == 2
        assert not any('test-key' in item for item in info_logs)

        assert node.set_parameters([Parameter('log_each_image', value=True)])[0].successful
        node._on_image(image)
        assert sum('收到 /sensor_msgs/image_raw' in item for item in info_logs) == 2
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


def test_failed_local_style_does_not_submit_mesh(monkeypatch):
    calls = []

    class StubSession:
        def close(self):
            pass

    class StubClient:
        def __init__(self, *_):
            self.session = StubSession()

    class StubStyleClient:
        def __init__(self, *_):
            pass

        def ready(self):
            return True

        def edit(self, *_):
            calls.append('style')
            raise EditorError('本地 FLUX 编辑失败')

        def close(self):
            pass

    monkeypatch.setattr('img2mesh.node.TripoClient', StubClient)
    monkeypatch.setattr('img2mesh.node.EditorClient', StubStyleClient)
    rclpy.init()
    node = None
    try:
        node = Img2MeshNode()
        assert node.set_parameters([
            Parameter('api_key', value='test-key'),
            Parameter('style_enabled', value=True),
        ])[0].successful
        node._on_prompt(String(data='赛博朋克风'))
        node._on_image(array_to_image_message(
            np.zeros((260, 300, 3), dtype=np.uint8), 'bgr8'))
        published = []
        node._publisher = type('PublisherStub', (), {'publish': published.append})()
        accepted, _ = node._try_submit()
        assert accepted
        node._worker.join(timeout=5)
        node._drain_updates()
        assert calls == ['style']
        assert published[-1].status == 'error'
        assert published[-1].style_task_id
        assert published[-1].task_id == ''
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
