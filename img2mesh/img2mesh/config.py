"""Load editable ROS parameters and the private local Tripo key from YAML."""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


COMMON_PARAMETER_NAMES = {
    'image_topic', 'style_prompt_topic', 'result_topic', 'status_topic',
    'model_url_topic', 'submit_service',
    'mode', 'api_base_url', 'model', 'face_limit', 'texture', 'pbr', 'quad',
    'auto_submit', 'auto_interval_sec', 'log_each_image', 'style_enabled',
    'style_server_url', 'style_request_timeout_sec', 'style_save_output',
    'style_output_directory', 'poll_interval_sec', 'request_timeout_sec',
    'task_timeout_sec', 'test_mode', 'test_image_path', 'test_exit_on_complete',
}


def config_dir():
    override = os.environ.get('IMG2MESH_CONFIG_DIR')
    if override:
        return Path(override).expanduser()
    source_dir = Path(__file__).resolve().parent.parent / 'config'
    if source_dir.is_dir():
        return source_dir
    return Path(get_package_share_directory('img2mesh')) / 'config'


def _read_yaml(path):
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'无法读取配置文件 {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError(f'配置文件 {path} 的顶层必须是映射')
    return data


def load_common_parameters():
    path = config_dir() / 'img2mesh.yaml'
    data = _read_yaml(path)
    node = data.get('img2mesh')
    parameters = node.get('ros__parameters') if isinstance(node, dict) else None
    if not isinstance(parameters, dict):
        raise ValueError(f'{path} 缺少 img2mesh.ros__parameters')
    missing = COMMON_PARAMETER_NAMES - parameters.keys()
    unknown = parameters.keys() - COMMON_PARAMETER_NAMES
    if missing or unknown:
        raise ValueError(f'{path} 参数名不匹配: 缺少 {sorted(missing)}，未知 {sorted(unknown)}')
    return dict(parameters)


def load_local_api_key():
    path = config_dir() / 'local_api.yaml'
    if not path.is_file():
        return ''
    data = _read_yaml(path)
    tripo = data.get('tripo')
    if not isinstance(tripo, dict) or not isinstance(tripo.get('api_key'), str):
        raise ValueError(f'{path} 必须包含 tripo.api_key 字符串')
    return tripo['api_key'].strip()
