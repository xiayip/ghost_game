"""ROS 2 node pairing a camera frame with a prompt for local FLUX editing."""

import copy
import json
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from flux_image_editor.client import EditorClient, EditorError
from flux_image_editor.config import load_ros_parameters
from flux_image_editor.image_codec import image_message_to_png, png_to_messages


DEFAULT_PARAMETERS = load_ros_parameters()


def utc_timestamp(wall_time=None):
    value = time.time() if wall_time is None else wall_time
    return datetime.fromtimestamp(value, timezone.utc).isoformat(
        timespec='milliseconds',
    ).replace('+00:00', 'Z')


def validate_settings(values):
    for name in (
        'image_topic', 'prompt_topic', 'output_image_topic', 'output_png_topic',
        'status_topic', 'submit_service', 'input_directory',
        'output_directory',
    ):
        if not isinstance(values[name], str) or not values[name].strip():
            return f'{name} 必须是非空字符串'
    parsed = urlparse(values['server_url']) if isinstance(values['server_url'], str) else None
    if (parsed is None or parsed.scheme not in ('http', 'https') or not parsed.netloc
            or parsed.username or parsed.password):
        return 'server_url 必须是无账号信息的 HTTP/HTTPS URL'
    timeout = values['request_timeout_sec']
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 1:
        return 'request_timeout_sec 必须至少为 1 秒'
    if not isinstance(values['max_input_bytes'], int) or isinstance(values['max_input_bytes'], bool):
        return 'max_input_bytes 必须是整数'
    if not 1024 <= values['max_input_bytes'] <= 100_000_000:
        return 'max_input_bytes 必须在 1024～100000000 之间'
    for name in (
            'auto_submit_on_prompt', 'save_input', 'save_output', 'log_timing'):
        if not isinstance(values[name], bool):
            return f'{name} 必须是布尔值'
    return ''


class FluxImageEditorNode(Node):
    PARAM_DEFAULTS = DEFAULT_PARAMETERS

    def __init__(self):
        super().__init__('flux_image_editor')
        for name, default in self.PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)
        reason = validate_settings(self._settings())
        if reason:
            raise ValueError(f'初始参数错误: {reason}')
        self.add_on_set_parameters_callback(self._validate_parameters)

        self._latest_image = None
        self._latest_image_received_wall = 0.0
        self._latest_image_received_monotonic = 0.0
        self._latest_prompt = ''
        self._latest_prompt_received_wall = 0.0
        self._latest_prompt_received_monotonic = 0.0
        self._image_count = 0
        self._prompt_generation = 0
        self._last_started_prompt_generation = 0
        self._busy = False
        self._busy_lock = threading.Lock()
        self._updates = queue.SimpleQueue()
        self._stop = threading.Event()
        self._worker = None

        result_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        settings = self._settings()
        self._raw_publisher = self.create_publisher(
            Image, settings['output_image_topic'], result_qos,
        )
        self._png_publisher = self.create_publisher(
            CompressedImage, settings['output_png_topic'], result_qos,
        )
        self._status_publisher = self.create_publisher(
            String, settings['status_topic'], result_qos,
        )
        self._image_subscriber = self.create_subscription(
            Image, settings['image_topic'], self._on_image, qos_profile_sensor_data,
        )
        self._prompt_subscriber = self.create_subscription(
            String, settings['prompt_topic'], self._on_prompt, 10,
        )
        self._submit_service = self.create_service(
            Trigger, settings['submit_service'], self._on_submit,
        )
        self._drain_timer = self.create_timer(0.05, self._drain_updates)

        self.get_logger().info('FLUX ROS bridge parameters loaded')
        self.get_logger().info(
            f'等待 {settings["image_topic"]} 图像和 {settings["prompt_topic"]} 提示词；'
            '每条非空提示词最多自动提交一次'
        )
        self.get_logger().info(
            f'输出：{settings["output_image_topic"]}、{settings["output_png_topic"]}；'
            f'推理服务：{settings["server_url"]}'
        )

    def _settings(self):
        return {name: self.get_parameter(name).value for name in self.PARAM_DEFAULTS}

    def _validate_parameters(self, parameters):
        values = self._settings()
        startup_only = {
            'image_topic', 'prompt_topic', 'output_image_topic', 'output_png_topic',
            'status_topic', 'submit_service',
        }
        if any(p.name in startup_only and p.value != values[p.name] for p in parameters):
            return SetParametersResult(
                successful=False, reason='话题名和服务名只在启动时设置',
            )
        values.update({parameter.name: parameter.value for parameter in parameters})
        reason = validate_settings(values)
        return SetParametersResult(successful=not reason, reason=reason)

    def _on_image(self, message):
        received_wall = time.time()
        received_monotonic = time.monotonic()
        self._latest_image = message
        self._latest_image_received_wall = received_wall
        self._latest_image_received_monotonic = received_monotonic
        self._image_count += 1
        if self._image_count == 1:
            stamp = message.header.stamp
            self.get_logger().info(
                f'收到首帧 {message.width}x{message.height} {message.encoding} '
                f'stamp={stamp.sec}.{stamp.nanosec:09d} '
                f'received_at={utc_timestamp(received_wall)}'
            )
        if self.get_parameter('auto_submit_on_prompt').value:
            self._try_start(force=False)

    def _on_prompt(self, message):
        received_wall = time.time()
        received_monotonic = time.monotonic()
        prompt = message.data.strip()
        preview = prompt.replace('\r', ' ').replace('\n', ' ')
        self.get_logger().info(
            f'收到提示词 received_at={utc_timestamp(received_wall)}：'
            f'{preview[:160] or "(空)"}'
        )
        if not prompt:
            return
        self._latest_prompt = prompt
        self._latest_prompt_received_wall = received_wall
        self._latest_prompt_received_monotonic = received_monotonic
        self._prompt_generation += 1
        if self.get_parameter('auto_submit_on_prompt').value:
            self._try_start(force=False)

    def _on_submit(self, _request, response):
        accepted, message = self._try_start(force=True)
        response.success = accepted
        response.message = message
        return response

    def _try_start(self, force=False):
        if self._latest_image is None:
            return False, '尚未收到输入图像'
        if not self._latest_prompt:
            return False, '尚未收到非空提示词'
        if not force and self._prompt_generation <= self._last_started_prompt_generation:
            return False, '当前提示词已经提交'
        with self._busy_lock:
            if self._busy:
                return False, '已有任务运行；最新提示词将在任务结束后提交'
            self._busy = True
        generation = self._prompt_generation
        self._last_started_prompt_generation = generation
        image = copy.deepcopy(self._latest_image)
        prompt = self._latest_prompt
        settings = self._settings()
        request_id = uuid.uuid4().hex
        accepted_wall = time.time()
        accepted_monotonic = time.monotonic()
        pair_ready_monotonic = max(
            self._latest_image_received_monotonic,
            self._latest_prompt_received_monotonic,
        )
        if pair_ready_monotonic == self._latest_image_received_monotonic:
            pair_ready_wall = self._latest_image_received_wall
        else:
            pair_ready_wall = self._latest_prompt_received_wall
        stamp = image.header.stamp
        timing_context = {
            'image_received_at': utc_timestamp(self._latest_image_received_wall),
            'prompt_received_at': utc_timestamp(self._latest_prompt_received_wall),
            'pair_ready_at': utc_timestamp(pair_ready_wall),
            'accepted_at': utc_timestamp(accepted_wall),
            'source_stamp': f'{stamp.sec}.{stamp.nanosec:09d}',
            'queue_wait_sec': max(0.0, accepted_monotonic - pair_ready_monotonic),
            '_pair_ready_monotonic': pair_ready_monotonic,
        }
        public_timing = {key: value for key, value in timing_context.items()
                         if not key.startswith('_')}
        self._queue_status(request_id, 'accepted', prompt, **public_timing)
        self.get_logger().info(
            f'图片+提示词已配对 request_id={request_id} '
            f'prompt_generation={generation} pair_ready_at={public_timing["pair_ready_at"]} '
            f'queue_wait={public_timing["queue_wait_sec"]:.3f}s '
            f'image={image.width}x{image.height} {image.encoding}'
        )
        self._worker = threading.Thread(
            target=self._run_job,
            args=(image, prompt, request_id, settings, timing_context), daemon=True,
        )
        self._worker.start()
        return True, f'已接受请求 request_id={request_id}'

    def _queue_status(self, request_id, status, prompt, **extra):
        payload = {
            'request_id': request_id,
            'status': status,
            'prompt': prompt,
            'input_path': '',
            'latest_input_path': '',
            'output_path': '',
            'width': 0,
            'height': 0,
            'input_png_bytes': 0,
            'output_png_bytes': 0,
            'image_received_at': '',
            'prompt_received_at': '',
            'pair_ready_at': '',
            'accepted_at': '',
            'encode_started_at': '',
            'request_sent_at': '',
            'png_received_at': '',
            'result_ready_at': '',
            'published_at': '',
            'source_stamp': '',
            'server_request_id': '',
            'model_load_sec': 0.0,
            'queue_wait_sec': 0.0,
            'encode_sec': 0.0,
            'http_roundtrip_sec': 0.0,
            'server_total_sec': 0.0,
            'inference_sec': 0.0,
            'transport_overhead_sec': 0.0,
            'push_to_png_received_sec': 0.0,
            'postprocess_sec': 0.0,
            'push_to_result_ready_sec': 0.0,
            'publish_sec': 0.0,
            'push_to_publish_sec': 0.0,
            'total_sec': 0.0,
            'seed': -1,
            'error': '',
            **extra,
        }
        self._updates.put({'kind': 'status', 'payload': payload})

    def _run_job(self, image, prompt, request_id, settings, timing_context):
        pair_ready_monotonic = timing_context['_pair_ready_monotonic']
        public_timing = {key: value for key, value in timing_context.items()
                         if not key.startswith('_')}
        client = None
        input_path = ''
        latest_input_path = ''
        input_png_bytes = 0
        try:
            encode_started = time.monotonic()
            encode_started_at = utc_timestamp()
            self._queue_status(
                request_id, 'encoding', prompt,
                **public_timing, encode_started_at=encode_started_at,
            )
            png = image_message_to_png(image, settings['max_input_bytes'])
            input_png_bytes = len(png)
            if settings['save_input']:
                input_directory = Path(
                    settings['input_directory']).expanduser()
                input_directory.mkdir(parents=True, exist_ok=True)
                input_target = input_directory / f'{request_id}_head_crop.png'
                latest_target = input_directory / 'latest_head_crop.png'
                input_target.write_bytes(png)
                latest_temp = input_directory / '.latest_head_crop.tmp'
                latest_temp.write_bytes(png)
                latest_temp.replace(latest_target)
                input_path = str(input_target)
                latest_input_path = str(latest_target)
            encode_sec = time.monotonic() - encode_started
            self.get_logger().info(
                f'输入 PNG 编码完成 request_id={request_id} bytes={len(png)} '
                f'encode_sec={encode_sec:.3f} '
                f'capture={input_path or "(未保存)"}'
            )
            client = EditorClient(settings['server_url'], settings['request_timeout_sec'])
            request_started = time.monotonic()
            request_sent_at = utc_timestamp()
            self._queue_status(
                request_id, 'running', prompt, **public_timing,
                encode_started_at=encode_started_at,
                request_sent_at=request_sent_at,
                input_path=input_path,
                latest_input_path=latest_input_path,
                input_png_bytes=len(png), encode_sec=encode_sec,
            )
            self.get_logger().info(
                f'已发送 PNG+提示词至 FLUX request_id={request_id} '
                f'sent_at={request_sent_at} png_bytes={len(png)} '
                f'prompt_chars={len(prompt)}'
            )
            response = client.edit(png, prompt, request_id)
            response_received_monotonic = time.monotonic()
            png_received_at = utc_timestamp()
            http_roundtrip_sec = response_received_monotonic - request_started
            push_to_png_received_sec = response_received_monotonic - pair_ready_monotonic
            transport_overhead_sec = max(0.0, http_roundtrip_sec - response.server_total_sec)
            self.get_logger().info(
                f'已收到新 PNG request_id={request_id} received_at={png_received_at} '
                f'bytes={len(response.png)} server_request_id={response.server_request_id or "(空)"} '
                f'http_roundtrip_sec={http_roundtrip_sec:.3f} '
                f'server_total_sec={response.server_total_sec:.3f} '
                f'inference_sec={response.inference_sec:.3f} '
                f'图片+提示词就绪至收到新图片={push_to_png_received_sec:.3f}s'
            )
            postprocess_started = time.monotonic()
            raw, compressed, width, height = png_to_messages(response.png, image.header)
            output_path = ''
            if settings['save_output']:
                directory = Path(settings['output_directory']).expanduser()
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f'{request_id}.png'
                target.write_bytes(response.png)
                output_path = str(target)
            postprocess_sec = time.monotonic() - postprocess_started
            result_ready_monotonic = time.monotonic()
            result_ready_at = utc_timestamp()
            push_to_result_ready_sec = result_ready_monotonic - pair_ready_monotonic
            self.get_logger().info(
                f'新 PNG 解码与保存完成 request_id={request_id} '
                f'result_ready_at={result_ready_at} postprocess_sec={postprocess_sec:.3f} '
                f'output={output_path or "(未保存)"}'
            )
            self._updates.put({
                'kind': 'success', 'raw': raw, 'compressed': compressed,
                'pair_ready_monotonic': pair_ready_monotonic,
                'payload': {
                    'request_id': request_id, 'status': 'success', 'prompt': prompt,
                    'input_path': input_path,
                    'latest_input_path': latest_input_path,
                    'output_path': output_path, 'width': width, 'height': height,
                    **public_timing,
                    'encode_started_at': encode_started_at,
                    'request_sent_at': request_sent_at,
                    'png_received_at': png_received_at,
                    'result_ready_at': result_ready_at,
                    'published_at': '',
                    'server_request_id': response.server_request_id,
                    'input_png_bytes': len(png),
                    'output_png_bytes': len(response.png),
                    'model_load_sec': response.model_load_sec,
                    'encode_sec': encode_sec,
                    'http_roundtrip_sec': http_roundtrip_sec,
                    'server_total_sec': response.server_total_sec,
                    'inference_sec': response.inference_sec,
                    'transport_overhead_sec': transport_overhead_sec,
                    'push_to_png_received_sec': push_to_png_received_sec,
                    'postprocess_sec': postprocess_sec,
                    'push_to_result_ready_sec': push_to_result_ready_sec,
                    'publish_sec': 0.0,
                    'push_to_publish_sec': 0.0,
                    'total_sec': push_to_result_ready_sec,
                    'seed': response.seed, 'error': '',
                },
            })
        except (ValueError, OSError, EditorError) as exc:
            elapsed = time.monotonic() - pair_ready_monotonic
            self._queue_status(
                request_id, 'error', prompt, **public_timing,
                input_path=input_path,
                latest_input_path=latest_input_path,
                input_png_bytes=input_png_bytes,
                total_sec=elapsed, error=str(exc),
            )
        except Exception as exc:  # Keep the ROS process alive on backend failures.
            elapsed = time.monotonic() - pair_ready_monotonic
            self.get_logger().error(
                f'FLUX 编辑内部错误 request_id={request_id}: {type(exc).__name__}'
            )
            self._queue_status(
                request_id, 'error', prompt, **public_timing, total_sec=elapsed,
                input_path=input_path,
                latest_input_path=latest_input_path,
                input_png_bytes=input_png_bytes,
                error=f'内部错误: {type(exc).__name__}',
            )
        finally:
            if client is not None:
                client.close()

    def _drain_updates(self):
        terminal = False
        while not self._updates.empty():
            update = self._updates.get_nowait()
            if update['kind'] == 'success':
                publish_started = time.monotonic()
                self._raw_publisher.publish(update['raw'])
                self._png_publisher.publish(update['compressed'])
                publish_finished = time.monotonic()
                payload = update['payload']
                payload['published_at'] = utc_timestamp()
                payload['publish_sec'] = publish_finished - publish_started
                payload['push_to_publish_sec'] = (
                    publish_finished - update['pair_ready_monotonic']
                )
                payload['total_sec'] = payload['push_to_publish_sec']
            payload = update['payload']
            self._status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            if payload['status'] == 'success':
                terminal = True
                timing = ''
                if self.get_parameter('log_timing').value:
                    timing = (
                        f' encode={payload["encode_sec"]:.3f}s'
                        f' http={payload["http_roundtrip_sec"]:.3f}s'
                        f' inference={payload["inference_sec"]:.3f}s'
                        f' postprocess={payload["postprocess_sec"]:.3f}s'
                        f' pair_to_png={payload["push_to_png_received_sec"]:.3f}s'
                        f' pair_to_publish={payload["push_to_publish_sec"]:.3f}s'
                    )
                self.get_logger().info(
                    f'FLUX 编辑完成 request_id={payload["request_id"]} '
                    f'{payload["width"]}x{payload["height"]}{timing} '
                    f'seed={payload["seed"]} '
                    f'published_at={payload["published_at"]} '
                    f'output={payload["output_path"] or "(未保存)"}'
                )
            elif payload['status'] == 'error':
                terminal = True
                self.get_logger().error(
                    f'FLUX 编辑失败 request_id={payload["request_id"]}: {payload["error"]}'
                )
        if terminal:
            with self._busy_lock:
                self._busy = False
            if (self.get_parameter('auto_submit_on_prompt').value
                    and self._prompt_generation > self._last_started_prompt_generation):
                self._try_start(force=False)

    def destroy_node(self):
        self._stop.set()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FluxImageEditorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
