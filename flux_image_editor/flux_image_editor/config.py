"""Configuration helpers shared by the ROS node and inference service."""

import os
from pathlib import Path

import yaml


ROS_PARAMETER_NAMES = {
    'image_topic', 'prompt_topic', 'output_image_topic', 'output_png_topic',
    'status_topic', 'submit_service', 'server_url', 'request_timeout_sec',
    'auto_submit_on_prompt', 'input_directory', 'save_input',
    'output_directory', 'save_output',
    'max_input_bytes', 'log_timing',
}


def config_dir():
    override = os.environ.get('FLUX_IMAGE_EDITOR_CONFIG_DIR')
    if override:
        return Path(override).expanduser()
    source_dir = Path(__file__).resolve().parent.parent / 'config'
    if source_dir.is_dir():
        return source_dir
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('flux_image_editor')) / 'config'


def read_yaml(path):
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'无法读取配置文件 {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError(f'配置文件 {path} 的顶层必须是映射')
    return data


def load_ros_parameters():
    path = config_dir() / 'flux_image_editor.yaml'
    data = read_yaml(path)
    node = data.get('flux_image_editor')
    parameters = node.get('ros__parameters') if isinstance(node, dict) else None
    if not isinstance(parameters, dict):
        raise ValueError(f'{path} 缺少 flux_image_editor.ros__parameters')
    missing = ROS_PARAMETER_NAMES - parameters.keys()
    unknown = parameters.keys() - ROS_PARAMETER_NAMES
    if missing or unknown:
        raise ValueError(f'{path} 参数名不匹配: 缺少 {sorted(missing)}，未知 {sorted(unknown)}')
    return dict(parameters)


def load_server_settings(path=None):
    configured = path or os.environ.get('FLUX_EDITOR_SERVER_CONFIG')
    path = Path(configured).expanduser() if configured else config_dir() / 'server.yaml'
    data = read_yaml(path)
    server = data.get('server')
    if not isinstance(server, dict):
        raise ValueError(f'{path} 缺少 server 映射')
    return dict(server)
