"""HTTP client for the local FLUX.2 inference service."""

from dataclasses import dataclass

import requests


class EditorError(RuntimeError):
    pass


@dataclass(frozen=True)
class EditResponse:
    png: bytes
    inference_sec: float
    width: int
    height: int
    model_load_sec: float = 0.0
    seed: int = -1
    server_request_id: str = ''
    server_total_sec: float = 0.0


class EditorClient:
    def __init__(self, base_url, timeout_sec, session=None):
        self.base_url = base_url.rstrip('/')
        self.timeout_sec = timeout_sec
        self.session = session or requests.Session()

    def ready(self):
        try:
            response = self.session.get(
                f'{self.base_url}/ready', timeout=min(self.timeout_sec, 5.0),
            )
            return response.status_code == 200 and response.json().get('ready') is True
        except (requests.RequestException, ValueError):
            return False

    def edit(self, png, prompt, request_id=''):
        try:
            response = self.session.post(
                f'{self.base_url}/v1/edit',
                files={'image': ('input.png', png, 'image/png')},
                data={'prompt': prompt, 'request_id': request_id}, timeout=self.timeout_sec,
            )
        except requests.RequestException as exc:
            raise EditorError(f'连接 FLUX 推理服务失败: {exc}') from exc
        if response.status_code != 200:
            detail = response.text[:500].strip()
            raise EditorError(f'FLUX 推理服务返回 HTTP {response.status_code}: {detail}')
        if response.headers.get('content-type', '').split(';')[0] != 'image/png':
            raise EditorError('FLUX 推理服务没有返回 image/png')
        try:
            inference_sec = float(response.headers.get('x-inference-sec', '0'))
            width = int(response.headers.get('x-output-width', '0'))
            height = int(response.headers.get('x-output-height', '0'))
            model_load_sec = float(response.headers.get('x-model-load-sec', '0'))
            seed = int(response.headers.get('x-seed', '-1'))
            server_request_id = response.headers.get('x-request-id', '')
            server_total_sec = float(response.headers.get('x-server-total-sec', '0'))
        except ValueError as exc:
            raise EditorError('FLUX 推理服务返回了无效的计时或尺寸响应头') from exc
        if not response.content:
            raise EditorError('FLUX 推理服务返回空 PNG')
        return EditResponse(
            response.content, inference_sec, width, height, model_load_sec, seed,
            server_request_id, server_total_sec,
        )

    def close(self):
        self.session.close()
