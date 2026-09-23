"""ROS 2 workflow from a prepared portrait to a Tripo mesh URL."""

import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import cv2
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ghost_game_interfaces.msg import MeshResult
from flux_image_editor.client import EditorClient, EditorError

from img2mesh.config import load_common_parameters, load_local_api_key
from img2mesh.image_codec import (
    array_to_image_message,
    compressed_image_to_png,
    image_to_png,
)
from img2mesh.tripo_client import TripoClient, TripoError, TripoTaskError


DEFAULT_PARAMETERS = load_common_parameters()


def validate_settings(values):
    if values['mode'] not in ('single', 'multi'):
        return 'mode 必须为 single 或 multi'
    if values['model'] not in ('P1-20260311', 'P2-20260801'):
        return 'model 必须为 P1-20260311 或 P2-20260801'
    if values['quad'] and values['model'] != 'P2-20260801':
        return 'quad 仅支持 P2-20260801'
    low = 48 if values['model'] == 'P2-20260801' else 50
    high = 25_000 if values['quad'] else (50_000 if values['model'] == 'P2-20260801' else 20_000)
    if not isinstance(values['face_limit'], int) or isinstance(values['face_limit'], bool):
        return 'face_limit 必须是整数'
    if not low <= values['face_limit'] <= high:
        return f'face_limit 必须在 {low}～{high} 之间'
    for name in ('input_compressed', 'texture', 'pbr', 'quad', 'delight',
                 'auto_submit', 'style_enabled',
                 'style_save_output', 'log_each_image'):
        if not isinstance(values[name], bool):
            return f'{name} 必须是布尔值'
    if not isinstance(values['style_server_url'], str):
        return 'style_server_url 必须是字符串'
    style_url = urlparse(values['style_server_url'])
    if (style_url.scheme not in ('http', 'https') or not style_url.netloc
            or style_url.username or style_url.password):
        return 'style_server_url 必须是无账号信息的 HTTP/HTTPS URL'
    if (not isinstance(values['style_output_directory'], str)
            or not values['style_output_directory'].strip()):
        return 'style_output_directory 必须是非空路径字符串'
    if values['pbr'] and not values['texture']:
        return 'pbr=true 会强制开启 texture，请一并设置 texture=true'
    if values['texture_version'] not in (
            'v2.5-20250123', 'v3.0-20250812', 'v3.5-20260815'):
        return 'texture_version 不受支持'
    if values['texture_quality'] not in ('fast', 'standard', 'detailed', 'extreme'):
        return 'texture_quality 必须为 fast、standard、detailed 或 extreme'
    if (values['texture_quality'] == 'fast' and
            values['texture_version'] != 'v3.5-20260815'):
        return 'texture_quality=fast 需要 texture_version=v3.5-20260815'
    for name in ('poll_interval_sec', 'request_timeout_sec', 'task_timeout_sec',
                 'auto_interval_sec', 'style_request_timeout_sec'):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 1.0:
            return f'{name} 必须至少为 1 秒'
    if values['task_timeout_sec'] <= values['request_timeout_sec']:
        return 'task_timeout_sec 必须大于 request_timeout_sec'
    if not isinstance(values['api_key'], str):
        return 'api_key 必须是字符串'
    if not isinstance(values['test_mode'], bool):
        return 'test_mode 必须是布尔值'
    if not isinstance(values['test_image_path'], str):
        return 'test_image_path 必须是文件路径字符串'
    if values['test_mode'] and not values['test_image_path']:
        return 'test_mode=true 时必须设置 test_image_path'
    if not isinstance(values['test_exit_on_complete'], bool):
        return 'test_exit_on_complete 必须是布尔值'
    if not isinstance(values['api_base_url'], str):
        return 'api_base_url 必须是字符串'
    parsed = urlparse(values['api_base_url'])
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        return 'api_base_url 必须是无账号信息的 HTTPS URL'
    for name in (
            'image_topic', 'style_prompt_topic', 'result_topic', 'status_topic',
            'model_url_topic', 'submit_service'):
        if not isinstance(values[name], str) or not values[name].strip():
            return f'{name} 必须是非空 ROS 名称'
    return ''


class Img2MeshNode(Node):
    PARAM_DEFAULTS = {'api_key': '', **DEFAULT_PARAMETERS}

    def __init__(self):
        super().__init__('img2mesh')
        self._local_api_key = load_local_api_key()
        for name, default in self.PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)
        reason = validate_settings(self._settings())
        if reason:
            raise ValueError(f'初始参数错误: {reason}')
        self.add_on_set_parameters_callback(self._validate_parameters)
        self.get_logger().info(
            'ROS 参数已加载；'
            f'本地 API Key {"可用" if self._local_api_key else "未配置"}')

        self._latest_image = None
        self._image_count = 0
        self._image_generation = 0
        self._last_auto_image_generation = 0
        self._latest_prompt = ''
        self._prompt_generation = 0
        self._last_auto_prompt_generation = 0
        self._last_auto_rejection = ''
        self._busy = False
        self._busy_lock = threading.Lock()
        self._stop = threading.Event()
        self._updates = queue.SimpleQueue()
        self._worker = None
        self._test_failed = False
        self._job_start_monotonic = None

        qos_result = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        result_topic = self.get_parameter('result_topic').value
        status_topic = self.get_parameter('status_topic').value
        model_url_topic = self.get_parameter('model_url_topic').value
        prompt_topic = self.get_parameter('style_prompt_topic').value
        image_topic = self.get_parameter('image_topic').value
        submit_service = self.get_parameter('submit_service').value
        self._publisher = self.create_publisher(MeshResult, result_topic, qos_result)
        self._status_publisher = self.create_publisher(
            String, status_topic, qos_result)
        self._model_url_publisher = self.create_publisher(
            String, model_url_topic, qos_result)
        self._prompt_subscriber = self.create_subscription(
            String, prompt_topic, self._on_prompt, 10,
        )
        self._subscriber = None
        if self.get_parameter('test_mode').value:
            self._test_timer = self.create_timer(0.2, self._start_file_test)
            self.get_logger().info(
                f'文件测试模式；读取 {self.get_parameter("test_image_path").value}'
            )
        else:
            input_type = (
                CompressedImage
                if self.get_parameter('input_compressed').value else Image)
            self._subscriber = self.create_subscription(
                input_type, image_topic, self._on_image, qos_profile_sensor_data,
            )
            if (self.get_parameter('auto_submit').value
                    and self.get_parameter('style_enabled').value):
                self.get_logger().info(
                    f'等待 {image_topic} 图像和 {prompt_topic} 提示词；'
                    '两者就绪后自动提交，每条提示词只提交一次'
                )
                self.get_logger().info(
                    '风格化后端：本地 FLUX；'
                    f'服务 {self.get_parameter("style_server_url").value}'
                )
            elif self.get_parameter('auto_submit').value:
                self.get_logger().info(
                    f'等待 {image_topic} 图像；每个新输入自动提交一次')
            else:
                self.get_logger().info(
                    f'等待 {image_topic} 图像；调用 {submit_service} 提交'
                )
        self._service = self.create_service(Trigger, submit_service, self._on_submit)
        self._drain_timer = self.create_timer(0.1, self._drain_updates)
        self._auto_timer = self.create_timer(0.5, self._retry_pending_auto_submission)
        self.get_logger().info(
            f'结果发布至 {result_topic}；状态 {status_topic}；'
            f'模型链接 {model_url_topic}')

    def _settings(self):
        return {name: self.get_parameter(name).value for name in self.PARAM_DEFAULTS}

    def _validate_parameters(self, parameters):
        values = self._settings()
        startup_only = (
            'test_mode', 'test_image_path', 'image_topic', 'input_compressed',
            'style_prompt_topic', 'result_topic', 'status_topic',
            'model_url_topic', 'submit_service')
        if any(p.name in startup_only and p.value != values[p.name]
               for p in parameters):
            return SetParametersResult(
                successful=False, reason='测试模式、文件路径和 ROS 名称只在启动时设置',
            )
        values.update({parameter.name: parameter.value for parameter in parameters})
        reason = validate_settings(values)
        return SetParametersResult(successful=not reason, reason=reason)

    def _on_image(self, message):
        self._latest_image = message
        self._image_count += 1
        self._image_generation += 1
        if self._image_count == 1 or self.get_parameter('log_each_image').value:
            stamp = message.header.stamp
            if isinstance(message, CompressedImage):
                description = f'{message.format or "compressed"} {len(message.data)} bytes'
            else:
                description = f'{message.width}x{message.height} {message.encoding}'
            self.get_logger().info(
                f'收到 {self.get_parameter("image_topic").value} #{self._image_count}: '
                f'{description} stamp={stamp.sec}.{stamp.nanosec:09d}'
            )
        self._maybe_auto_submit()

    def _on_prompt(self, message):
        self._latest_prompt = message.data.strip()
        prompt_preview = self._latest_prompt.replace('\r', ' ').replace('\n', ' ')
        self.get_logger().info(
            f'收到 {self.get_parameter("style_prompt_topic").value}: '
            f'{prompt_preview[:160] or "(空)"}'
        )
        if self._latest_prompt:
            self._prompt_generation += 1
        if self.get_parameter('style_enabled').value:
            self._maybe_auto_submit()

    def _maybe_auto_submit(self):
        if (not self.get_parameter('auto_submit').value
                or self.get_parameter('mode').value != 'single'):
            return
        if self.get_parameter('style_enabled').value:
            if (self._latest_image is None or not self._latest_prompt
                    or self._prompt_generation <= self._last_auto_prompt_generation):
                return
            accepted, detail = self._try_submit()
            if accepted:
                self._last_auto_prompt_generation = self._prompt_generation
                self._last_auto_rejection = ''
                self.get_logger().info(f'提示词触发自动提交：{detail}')
            elif detail != '已有生成任务进行中，请等待结果':
                if detail != self._last_auto_rejection:
                    self.get_logger().warning(f'自动提交尚未开始：{detail}')
                    self._last_auto_rejection = detail
            return
        if (self._latest_image is None or
                self._image_generation <= self._last_auto_image_generation):
            return
        accepted, detail = self._try_submit()
        if accepted:
            self._last_auto_image_generation = self._image_generation
            self._last_auto_rejection = ''
            self.get_logger().info(f'新图像触发自动提交：{detail}')
        elif detail != '已有生成任务进行中，请等待结果':
            if detail != self._last_auto_rejection:
                self.get_logger().warning(f'自动提交尚未开始：{detail}')
                self._last_auto_rejection = detail

    def _retry_pending_auto_submission(self):
        # A new input may arrive while a previous task is busy. Retry it after
        # the worker finishes even when the source publishes only one frame.
        if self.get_parameter('auto_submit').value:
            self._maybe_auto_submit()

    def _start_file_test(self):
        self._test_timer.cancel()
        path = self.get_parameter('test_image_path').value
        pixels = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if pixels is None:
            self._finish_file_test_with_error(f'无法读取测试图片: {path}')
            return
        if pixels.ndim == 2:
            encoding = 'mono8'
        elif pixels.ndim == 3 and pixels.shape[2] == 3:
            encoding = 'bgr8'
        elif pixels.ndim == 3 and pixels.shape[2] == 4:
            encoding = 'bgra8'
        else:
            self._finish_file_test_with_error('测试图片通道数不受支持')
            return
        message = array_to_image_message(pixels, encoding)
        message.header.stamp = self.get_clock().now().to_msg()
        self._latest_image = message
        accepted, detail = self._try_submit()
        if accepted:
            self.get_logger().info(detail)
        else:
            self._finish_file_test_with_error(detail)

    def _finish_file_test_with_error(self, message):
        self._test_failed = True
        self.get_logger().error(message)
        if self.get_parameter('test_exit_on_complete').value:
            rclpy.try_shutdown()

    def _on_submit(self, _request, response):
        response.success, response.message = self._try_submit()
        return response

    def _try_submit(self):
        # Flush the previous task's final status before announcing a new request.
        self._drain_updates()
        image = self._latest_image
        if image is None:
            return False, '尚未收到图像'
        settings = self._settings()
        if settings['mode'] == 'multi':
            return False, 'multi 模式预留，当前尚未实现；未调用 Tripo API'
        if settings['style_enabled']:
            if not self._latest_prompt:
                return False, ('风格化模式尚未收到非空提示词：'
                               f'{settings["style_prompt_topic"]}')
            settings['style_prompt'] = self._latest_prompt
        api_key = (settings['api_key'] or os.environ.get('TRIPO_API_KEY', '')
                   or self._local_api_key)
        if not api_key:
            return False, '未设置 api_key 参数、TRIPO_API_KEY 环境变量或本地 Key 文件'
        with self._busy_lock:
            if self._busy:
                return False, '已有生成任务进行中，请等待结果'
            self._busy = True
        try:
            png = (
                compressed_image_to_png(image)
                if isinstance(image, CompressedImage) else image_to_png(image))
        except Exception as exc:
            with self._busy_lock:
                self._busy = False
            return False, f'图像转换失败: {exc}'
        request_id = uuid.uuid4().hex
        settings['api_key'] = api_key
        self.get_logger().info(
            f'已接受生成请求 request_id={request_id}，PNG={len(png)} bytes'
        )
        initial_status = 'styling' if settings['style_enabled'] else 'uploading'
        initial_result = self._result(
            image, request_id, settings, status=initial_status,
            style_task_id=request_id if settings['style_enabled'] else '',
        )
        self._publisher.publish(initial_result)
        self._publish_status(initial_result)
        # Timer starts immediately before the local-style/Tripo worker begins.
        self._job_start_monotonic = time.monotonic()
        self._worker = threading.Thread(
            target=self._run_job,
            args=(image, png, request_id, settings),
            daemon=True,
        )
        self._worker.start()
        return True, f'已接受请求 request_id={request_id}'

    @staticmethod
    def _result(image, request_id, settings, status, task_id='', progress=0,
                model_url='', preview_url='', error_message='', style_task_id='',
                style_image_url=''):
        result = MeshResult()
        result.source_header = image.header
        result.request_id = request_id
        result.mode = settings['mode']
        result.model = settings['model']
        result.task_id = task_id
        result.style_task_id = style_task_id
        result.style_image_url = style_image_url
        result.status = status
        result.progress = int(progress or 0)
        result.model_url = model_url
        result.preview_url = preview_url
        result.error_message = error_message
        return result

    def _run_job(self, image, png, request_id, settings):
        task_id = ''
        style_task_id = ''
        style_image_url = ''
        client = None
        style_client = None
        try:
            if settings['style_enabled']:
                style_started_at = time.monotonic()
                style_task_id = request_id
                style_client = EditorClient(
                    settings['style_server_url'], settings['style_request_timeout_sec'],
                )
                if not style_client.ready():
                    raise EditorError(
                        f'本地 FLUX 推理服务尚未就绪: {settings["style_server_url"]}'
                    )
                self.get_logger().info(
                    f'开始本地 FLUX 图像编辑 request_id={request_id} '
                    f'server={settings["style_server_url"]} '
                    f'png_bytes={len(png)} prompt_chars={len(settings["style_prompt"])}'
                )
                self._updates.put(self._result(
                    image, request_id, settings, status='styling',
                    style_task_id=style_task_id,
                ))
                style_response = style_client.edit(
                    png, settings['style_prompt'], request_id,
                )
                style_task_id = style_response.server_request_id or request_id
                styled_png = style_response.png
                if settings['style_save_output']:
                    directory = Path(settings['style_output_directory']).expanduser()
                    directory.mkdir(parents=True, exist_ok=True)
                    target = (directory / f'{request_id}.png').resolve()
                    target.write_bytes(styled_png)
                    style_image_url = target.as_uri()
                style_elapsed_sec = time.monotonic() - style_started_at
                self.get_logger().info(
                    f'已收到本地风格化 PNG request_id={request_id} '
                    f'style_request_id={style_task_id} bytes={len(styled_png)} '
                    f'size={style_response.width}x{style_response.height} '
                    f'inference_sec={style_response.inference_sec:.3f} '
                    f'server_total_sec={style_response.server_total_sec:.3f} '
                    f'local_style_total_sec={style_elapsed_sec:.3f} '
                    f'output={style_image_url or "(未保存)"}'
                )
                self._updates.put(self._result(
                    image, request_id, settings, status='styling', progress=100,
                    style_task_id=style_task_id, style_image_url=style_image_url,
                ))
                png_for_mesh = styled_png
            else:
                png_for_mesh = png
            client = TripoClient(
                settings['api_key'], settings['api_base_url'], settings['request_timeout_sec'],
            )
            upload_kind = '本地风格化 PNG' if settings['style_enabled'] else 'PNG'
            self.get_logger().info(
                f'开始上传{upload_kind}至 Tripo request_id={request_id} '
                f'bytes={len(png_for_mesh)}'
            )
            upload_started_at = time.monotonic()
            file_token = client.upload_png(png_for_mesh)
            upload_elapsed_sec = time.monotonic() - upload_started_at
            self.get_logger().info(
                f'{upload_kind}上传成功 request_id={request_id} '
                f'upload_sec={upload_elapsed_sec:.3f}'
            )
            image_input = file_token
            self.get_logger().info(
                f'开始提交三维生成任务 request_id={request_id} '
                f'face_limit={settings["face_limit"]} '
                f'texture={settings["texture"]} pbr={settings["pbr"]} '
                f'texture_version={settings["texture_version"]} '
                f'texture_quality={settings["texture_quality"]} '
                f'delight={settings["delight"]}'
            )
            submit_started_at = time.monotonic()
            task_id = client.submit_image(
                image_input, settings['model'], settings['face_limit'],
                settings['texture'], settings['pbr'], settings['quad'],
                settings['texture_version'], settings['texture_quality'],
                settings['delight'],
            )
            submit_elapsed_sec = time.monotonic() - submit_started_at
            self.get_logger().info(
                f'三维生成任务已提交 request_id={request_id} task_id={task_id} '
                f'submit_sec={submit_elapsed_sec:.3f}'
            )
            self._updates.put(self._result(
                image, request_id, settings, status='queued', task_id=task_id,
                style_task_id=style_task_id, style_image_url=style_image_url,
            ))

            def on_progress(status, progress):
                if status in ('queued', 'running'):
                    self._updates.put(self._result(
                        image, request_id, settings, status=status,
                        task_id=task_id, progress=progress,
                        style_task_id=style_task_id, style_image_url=style_image_url,
                    ))

            generation_started_at = time.monotonic()
            output = client.wait_for_task(
                task_id, settings['poll_interval_sec'], settings['task_timeout_sec'],
                self._stop, on_progress,
            )
            generation_elapsed_sec = time.monotonic() - generation_started_at
            self.get_logger().info(
                f'Tripo 阶段计时 request_id={request_id} '
                f'upload_sec={upload_elapsed_sec:.3f} '
                f'submit_sec={submit_elapsed_sec:.3f} '
                f'generation_sec={generation_elapsed_sec:.3f}'
            )
            self._updates.put(self._result(
                image, request_id, settings, status='success', task_id=task_id,
                progress=100, model_url=output['model_url'],
                preview_url=output.get('rendered_image_url', ''),
                style_task_id=style_task_id, style_image_url=style_image_url,
            ))
        except TripoTaskError as exc:
            self._updates.put(self._result(
                image, request_id, settings, status=exc.status, task_id=task_id,
                error_message=str(exc),
                style_task_id=style_task_id, style_image_url=style_image_url,
            ))
        except (TripoError, EditorError) as exc:
            self._updates.put(self._result(
                image, request_id, settings, status='error', task_id=task_id,
                error_message=str(exc),
                style_task_id=style_task_id, style_image_url=style_image_url,
            ))
        except Exception as exc:
            self.get_logger().error(f'意外错误: {type(exc).__name__}')
            self._updates.put(self._result(
                image, request_id, settings, status='error', task_id=task_id,
                error_message=f'内部错误: {type(exc).__name__}',
                style_task_id=style_task_id, style_image_url=style_image_url,
            ))
        finally:
            if style_client is not None:
                style_client.close()
            if client is not None:
                client.session.close()
            with self._busy_lock:
                self._busy = False

    def _publish_status(self, result):
        status_payload = {
            'request_id': result.request_id,
            'task_id': result.task_id,
            'status': result.status,
            'progress': int(result.progress),
            'model': result.model,
            'model_ready': bool(
                result.status == 'success' and result.model_url),
            'error': result.error_message,
        }
        self._status_publisher.publish(String(
            data=json.dumps(
                status_payload, ensure_ascii=False,
                separators=(',', ':'))))

    def _drain_updates(self):
        while not self._updates.empty():
            result = self._updates.get_nowait()
            self._publisher.publish(result)
            self._publish_status(result)
            terminal = result.status in ('success', 'error', 'failed', 'cancelled')
            elapsed = None
            if terminal and self._job_start_monotonic is not None:
                elapsed = time.monotonic() - self._job_start_monotonic
                self._job_start_monotonic = None
            if result.status == 'success':
                if result.model_url:
                    self._model_url_publisher.publish(
                        String(data=result.model_url))
                self.get_logger().info(f'生成完成 task_id={result.task_id}')
                self.get_logger().info(
                    f'已收到模型 URL request_id={result.request_id} '
                    f'bytes={len(result.model_url.encode("utf-8"))}'
                )
                if elapsed is not None:
                    self.get_logger().info(f'从开始上传流程到收到 URL：{elapsed:.2f} 秒')
                if self.get_parameter('test_mode').value:
                    self.get_logger().info(f'preview_url={result.preview_url}')
            elif result.status in ('error', 'failed', 'cancelled'):
                self.get_logger().error(result.error_message)
                if elapsed is not None:
                    self.get_logger().info(f'本次请求耗时：{elapsed:.2f} 秒')
                if self.get_parameter('test_mode').value:
                    self._test_failed = True
            if (self.get_parameter('test_mode').value
                    and self.get_parameter('test_exit_on_complete').value
                    and result.status in ('success', 'error', 'failed', 'cancelled')):
                rclpy.try_shutdown()

    def destroy_node(self):
        self._stop.set()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Img2MeshNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            failed = node._test_failed
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                # ros2 launch may deliver SIGINT while rclpy is already tearing
                # down entities. The requested shutdown is still complete.
                pass
        else:
            failed = True
        rclpy.try_shutdown()
    return 1 if failed else 0


if __name__ == '__main__':
    main()
