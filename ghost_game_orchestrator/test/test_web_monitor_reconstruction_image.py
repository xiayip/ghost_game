import json

import pytest

from ghost_game_orchestrator.web_monitor import _ReconstructionImageStore


def png_bytes(payload=b'ghost'):
    return b'\x89PNG\r\n\x1a\n' + payload


def test_reconstruction_image_store_exposes_versioned_same_origin_url():
    store = _ReconstructionImageStore(max_bytes=1024)

    assert store.set(png_bytes()) is True
    info = json.loads(store.get_json())

    assert info['ready'] is True
    assert info['version'] == 1
    assert info['bytes'] == len(png_bytes())
    assert info['image_url'] == '/api/reconstruction/image.png?v=1'
    assert store.get() == png_bytes()


def test_reconstruction_image_store_clears_between_rounds():
    store = _ReconstructionImageStore()
    store.set(png_bytes())

    store.clear()

    info = json.loads(store.get_json())
    assert info['ready'] is False
    assert info['version'] == 1
    assert info['bytes'] == 0
    assert info['image_url'] is None
    assert store.get() is None


def test_reconstruction_image_store_rejects_invalid_or_oversized_data():
    store = _ReconstructionImageStore(max_bytes=16)

    assert store.set(b'not-png') is False
    assert store.set(png_bytes(b'x' * 20)) is False
    assert json.loads(store.get_json())['ready'] is False


def test_reconstruction_image_store_requires_positive_limit():
    with pytest.raises(ValueError):
        _ReconstructionImageStore(max_bytes=0)
