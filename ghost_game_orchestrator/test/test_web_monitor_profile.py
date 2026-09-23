import json

from ghost_game_orchestrator.web_monitor import _CyberProfileStore


def valid_profile():
    return {
        'request_id': 'profile-7',
        'status': 'success',
        'character_name': '镜面幽灵',
        'codename': '007',
        'character_gender': '未知',
        'role': '网络侦察员',
        'cyberware_level': {'code': 'C2', 'name': '增强级'},
        'introduction': '游走在城市网络边缘的侦察员。',
    }


def test_profile_store_tracks_generation_and_success():
    store = _CyberProfileStore()
    store.begin()
    assert json.loads(store.get_json())['status'] == 'generating'

    assert store.set_json(json.dumps(valid_profile(), ensure_ascii=False))
    result = json.loads(store.get_json())
    assert result['status'] == 'success'
    assert result['character_name'] == '镜面幽灵'
    assert result['cyberware_level'] == {'code': 'C2', 'name': '增强级'}
    assert result['sequence'] == 2


def test_profile_store_reports_error_and_resets():
    store = _CyberProfileStore()
    assert store.set_json(json.dumps({
        'request_id': '',
        'status': 'error',
        'error_message': 'DeepSeek API Key 或客户端未配置',
    }, ensure_ascii=False))
    assert 'API Key' in json.loads(store.get_json())['error_message']

    store.reset()
    assert json.loads(store.get_json())['status'] == 'idle'


def test_profile_store_rejects_malformed_success():
    store = _CyberProfileStore()
    malformed = valid_profile()
    malformed.pop('introduction')

    assert not store.set_json(json.dumps(malformed, ensure_ascii=False))
    assert json.loads(store.get_json())['status'] == 'idle'
