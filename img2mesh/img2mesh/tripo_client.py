"""Small, synchronous Tripo V3 client used by the node's worker thread."""

import time

import requests


class TripoError(RuntimeError):
    """An upload, API, or generation task failed."""


class TripoTaskError(TripoError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class TripoClient:
    def __init__(self, api_key, base_url, request_timeout, session=None):
        self.base_url = base_url.rstrip('/')
        self.request_timeout = request_timeout
        self.session = session or requests.Session()
        self.headers = {'Authorization': f'Bearer {api_key}'}

    @staticmethod
    def _data(response, stage):
        try:
            payload = response.json()
        except ValueError as exc:
            raise TripoError(f'{stage}: API 返回的不是 JSON') from exc
        if not isinstance(payload, dict):
            raise TripoError(f'{stage}: API 响应格式错误')
        if not response.ok or payload.get('code') != 0:
            code = payload.get('code', response.status_code)
            message = str(payload.get('message') or payload.get('error') or '')[:200]
            raise TripoError(f'{stage}: API 错误 {code} {message}'.strip())
        data = payload.get('data')
        if not isinstance(data, dict):
            raise TripoError(f'{stage}: API 响应缺少 data')
        return data

    def upload_png(self, png_bytes):
        try:
            response = self.session.post(
                f'{self.base_url}/files/presign',
                headers=self.headers,
                json={'format': 'png'},
                timeout=self.request_timeout,
            )
            data = self._data(response, '申请上传地址')
            upload_url = data.get('presigned_url')
            file_token = data.get('file_token')
            if not upload_url or not file_token:
                raise TripoError('申请上传地址: 响应缺少 presigned_url 或 file_token')
            response = self.session.put(
                upload_url,
                data=png_bytes,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=self.request_timeout,
            )
            if not response.ok:
                raise TripoError(f'上传 PNG 失败: HTTP {response.status_code}')
            return file_token
        except requests.RequestException as exc:
            # A presigned URL contains a signature. Do not copy its URL into ROS logs.
            raise TripoError('上传 PNG 网络请求失败') from exc

    def submit_image(self, image_input, model, face_limit, texture, pbr, quad,
                     texture_version='v3.0-20250812',
                     texture_quality='standard', delight=True):
        body = {
            'input': image_input,
            'model': model,
            'face_limit': face_limit,
            'texture': texture,
            'pbr': pbr,
        }
        if quad:
            body['quad'] = True
        if texture:
            body['texture_version'] = texture_version
            body['texture_quality'] = texture_quality
            if texture_version == 'v3.5-20260815':
                body['delight'] = delight
        try:
            response = self.session.post(
                f'{self.base_url}/generation/image-to-model',
                headers=self.headers,
                json=body,
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:
            raise TripoError('提交生成任务网络请求失败') from exc
        data = self._data(response, '提交生成任务')
        task_id = data.get('task_id')
        if not task_id:
            raise TripoError('提交生成任务: 响应缺少 task_id')
        return task_id

    def wait_for_task(self, task_id, interval, timeout, stop_event, on_progress,
                      output_key='model_url'):
        deadline = time.monotonic() + timeout
        previous = None
        while not stop_event.is_set():
            if time.monotonic() >= deadline:
                raise TripoError(f'等待生成超时，任务 {task_id} 可稍后查询')
            try:
                response = self.session.get(
                    f'{self.base_url}/tasks/{task_id}',
                    headers=self.headers,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as exc:
                raise TripoError(f'查询任务 {task_id} 网络请求失败') from exc
            data = self._data(response, '查询任务')
            status = data.get('status')
            progress = data.get('progress', 0)
            if (status, progress) != previous:
                on_progress(status, progress)
                previous = (status, progress)
            if status == 'success':
                output = data.get('output') or {}
                if not output.get(output_key):
                    raise TripoError(f'任务 {task_id} 成功但缺少 {output_key}')
                return output
            if status in ('failed', 'cancelled'):
                reason = str(data.get('error_message') or data.get('error_code') or status)[:200]
                raise TripoTaskError(status, f'任务 {task_id} {status}: {reason}')
            if status not in ('queued', 'running'):
                raise TripoError(f'任务 {task_id} 返回未知状态: {status}')
            stop_event.wait(min(interval, max(0.0, deadline - time.monotonic())))
        raise TripoError('节点正在退出')
