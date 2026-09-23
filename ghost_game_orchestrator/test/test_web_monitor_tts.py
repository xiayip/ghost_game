import json

from ghost_game_orchestrator.web_monitor import _TtsStore


def test_tts_store_exposes_unicode_caption_and_monotonic_sequence():
    store = _TtsStore()

    initial = json.loads(store.get_json())
    assert initial == {'text': '', 'sequence': 0, 'received_at': None}

    store.set('  正在潜入深网。  ')
    first = json.loads(store.get_json())
    assert first['text'] == '正在潜入深网。'
    assert first['sequence'] == 1
    assert isinstance(first['received_at'], float)

    store.set('特工档案生成中。')
    second = json.loads(store.get_json())
    assert second['text'] == '特工档案生成中。'
    assert second['sequence'] == 2
    assert second['received_at'] >= first['received_at']


def test_tts_store_ignores_empty_caption():
    store = _TtsStore()
    store.set('   ')

    assert json.loads(store.get_json())['sequence'] == 0
