import json
import struct
import tempfile
from pathlib import Path

from ghost_game_orchestrator.web_monitor import _MeshStore


def make_glb():
    document = b'{"asset":{"version":"2.0"}}'
    document += b' ' * ((4 - len(document) % 4) % 4)
    chunk = struct.pack('<II', len(document), 0x4E4F534A) + document
    return struct.pack('<4sII', b'glTF', 2, 12 + len(chunk)) + chunk


def test_mesh_store_loads_local_glb_and_exposes_same_origin_url():
    with tempfile.TemporaryDirectory(prefix='ghost-mesh-') as directory:
        model = Path(directory) / 'sample.glb'
        model.write_bytes(make_glb())

        store = _MeshStore(model)
        info = json.loads(store.get_json())

        assert info['status'] == 'ready'
        assert info['source'] == 'local'
        assert info['version'] == 1
        assert info['bytes'] == model.stat().st_size
        assert info['model_url'] == '/api/mesh/model.glb?v=1'
        assert store.get_model() == model.read_bytes()


def test_mesh_store_rejects_invalid_local_file():
    with tempfile.TemporaryDirectory(prefix='ghost-mesh-') as directory:
        model = Path(directory) / 'not-a-model.glb'
        model.write_bytes(b'not gltf')

        store = _MeshStore(model)
        info = json.loads(store.get_json())

        assert info['status'] == 'error'
        assert info['model_url'] is None
        assert info['message'].startswith('local_model_error:')


def test_mesh_store_downloads_published_url_without_exposing_it():
    payload = make_glb()

    class Response:
        headers = {'Content-Length': str(len(payload))}

        def __init__(self):
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, amount):
            chunk = payload[self.offset:self.offset + amount]
            self.offset += len(chunk)
            return chunk

    def opener(request, timeout):
        assert request.full_url == 'https://models.example/ghost.glb?token=secret'
        assert timeout == 60.0
        return Response()

    store = _MeshStore()
    assert store.update_from_url(
        'https://models.example/ghost.glb?token=secret', opener=opener)
    info_text = store.get_json()
    info = json.loads(info_text)

    assert info['status'] == 'ready'
    assert info['source'] == 'published_url'
    assert info['model_url'] == '/api/mesh/model.glb?v=1'
    assert 'models.example' not in info_text
    assert 'secret' not in info_text


def test_mesh_store_rejects_non_http_url():
    store = _MeshStore()

    assert not store.update_from_url('file:///etc/passwd')
    assert json.loads(store.get_json())['message'] == 'invalid_model_url'
