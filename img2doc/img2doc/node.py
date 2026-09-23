from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .card_schema import CardValidationError, parse_and_validate_card, serialize_result
from .config import load_api_key
from .deepseek_client import DeepSeekError, DeepSeekVisionClient
from .image_codec import ImageEncodingError, encode_ros_image


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Img2DocNode(Node):
    def __init__(self) -> None:
        super().__init__("img2doc")
        defaults = {
            "image_topic": "/sensor_msgs/image_raw",
            "output_topic": "/img2doc/card",
            "submit_service": "/img2doc/submit",
            "api_base_url": "https://api.deepseek.com",
            "api_key_file": "",
            "model": "deepseek-flash",
            "image_detail": "original",
            "request_timeout_sec": 120.0,
            "max_retries": 2,
            "retry_interval_sec": 1.0,
            "max_tokens": 1200,
            "temperature": 0.2,
            "max_long_edge": 1300,
            "jpeg_quality": 90,
            "max_input_bytes": 20_000_000,
            "input_transient_local": False,
            "auto_submit": False,
            "auto_interval_sec": 60.0,
            "log_each_image": False,
            "system_prompt": "Return a json object.",
            "card_prompt": "Analyze the fictional character in this image and return json.",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._values = {name: self.get_parameter(name).value for name in defaults}
        self._lock = threading.Lock()
        self._latest_image: Optional[Image] = None
        self._image_count = 0
        self._busy = False
        self._last_auto_submit = 0.0

        result_qos = QoSProfile(depth=1)
        result_qos.reliability = ReliabilityPolicy.RELIABLE
        result_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._publisher = self.create_publisher(String, str(self._values["output_topic"]), result_qos)
        input_qos = qos_profile_sensor_data
        if bool(self._values["input_transient_local"]):
            input_qos = QoSProfile(depth=1)
            input_qos.reliability = ReliabilityPolicy.RELIABLE
            input_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._subscription = self.create_subscription(
            Image,
            str(self._values["image_topic"]),
            self._on_image,
            input_qos,
        )
        self._service = self.create_service(
            Trigger, str(self._values["submit_service"]), self._on_submit
        )

        api_key = load_api_key(str(self._values["api_key_file"]))
        self._client: Optional[DeepSeekVisionClient] = None
        if api_key:
            try:
                self._client = DeepSeekVisionClient(
                    api_key=api_key,
                    api_base_url=str(self._values["api_base_url"]),
                    model=str(self._values["model"]),
                    image_detail=str(self._values["image_detail"]),
                    timeout_sec=float(self._values["request_timeout_sec"]),
                    max_retries=int(self._values["max_retries"]),
                    retry_interval_sec=float(self._values["retry_interval_sec"]),
                    max_tokens=int(self._values["max_tokens"]),
                    temperature=float(self._values["temperature"]),
                )
            except ValueError as exc:
                self.get_logger().error(f"DeepSeek 配置无效: {exc}")
        else:
            self.get_logger().warning(
                "未设置 DeepSeek API Key；请配置 DEEPSEEK_API_KEY 或 config/local_api.yaml"
            )

        self.get_logger().info(
            f"等待 {self._values['image_topic']} 图像；调用 {self._values['submit_service']} 提交最新帧"
        )
        self.get_logger().info(
            f"名片 JSON 发布至 {self._values['output_topic']} (std_msgs/msg/String)，模型={self._values['model']}"
        )

    def _on_image(self, message: Image) -> None:
        with self._lock:
            self._latest_image = message
            self._image_count += 1
            count = self._image_count
        if count == 1 or bool(self._values["log_each_image"]):
            self.get_logger().info(
                f"收到{'首帧' if count == 1 else '图像'} #{count}: "
                f"{message.width}x{message.height} {message.encoding} received_at={utc_now()}"
            )
        if not bool(self._values["auto_submit"]):
            return
        now = time.monotonic()
        if now - self._last_auto_submit < float(self._values["auto_interval_sec"]):
            return
        # _start_request acquires _lock. Keep it outside the image-cache
        # critical section; the original package called it while holding the
        # same non-reentrant lock and deadlocked on the first auto submission.
        accepted, reason = self._start_request("auto")
        if accepted:
            self._last_auto_submit = now
        elif reason != "已有请求正在处理":
            self._last_auto_submit = now
            self._publish_payload({
                "request_id": "",
                "status": "error",
                "error_message": reason,
            })

    def _on_submit(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        accepted, message = self._start_request("service")
        response.success = accepted
        response.message = message
        return response

    def _start_request(self, source: str) -> tuple[bool, str]:
        with self._lock:
            if self._client is None:
                return False, "DeepSeek API Key 或客户端未配置"
            if self._latest_image is None:
                return False, "尚未收到图像"
            if self._busy:
                return False, "已有请求正在处理"
            self._busy = True
            image = self._latest_image
        request_id = uuid.uuid4().hex
        self.get_logger().info(
            f"接受名片生成请求 request_id={request_id} source={source} accepted_at={utc_now()}"
        )
        self._publish_payload({
            "request_id": request_id,
            "status": "running",
        })
        threading.Thread(
            target=self._process_request,
            args=(request_id, image),
            daemon=True,
            name=f"img2doc-{request_id[:8]}",
        ).start()
        return True, f"accepted request_id={request_id}"

    def _process_request(self, request_id: str, image: Image) -> None:
        started = time.monotonic()
        try:
            encoded_started = time.monotonic()
            encoded = encode_ros_image(
                image,
                max_long_edge=int(self._values["max_long_edge"]),
                jpeg_quality=int(self._values["jpeg_quality"]),
                max_input_bytes=int(self._values["max_input_bytes"]),
            )
            encode_sec = time.monotonic() - encoded_started
            self.get_logger().info(
                f"输入 JPEG 准备完成 request_id={request_id} size={encoded.width}x{encoded.height} "
                f"bytes={len(encoded.data)} encode_sec={encode_sec:.3f}"
            )
            self.get_logger().info(
                f"发送图像与固定提示词至 DeepSeek request_id={request_id} sent_at={utc_now()}"
            )
            assert self._client is not None
            reply = self._client.edit(
                encoded.data,
                str(self._values["system_prompt"]),
                str(self._values["card_prompt"]),
            )
            self.get_logger().info(
                f"收到 DeepSeek 响应 request_id={request_id} received_at={utc_now()} "
                f"api_roundtrip_sec={reply.elapsed_sec:.3f} model={reply.model} "
                f"finish_reason={reply.finish_reason} total_tokens={reply.usage.get('total_tokens', 'unknown')}"
            )
            card = parse_and_validate_card(reply.content)
            payload = {
                "request_id": request_id,
                "status": "success",
                **card,
            }
        except (ImageEncodingError, DeepSeekError, CardValidationError, Exception) as exc:
            # The broad final catch keeps a worker failure observable on the ROS result topic.
            self.get_logger().error(f"名片生成失败 request_id={request_id}: {exc}")
            payload = {
                "request_id": request_id,
                "status": "error",
                "error_message": str(exc),
            }
        finally:
            with self._lock:
                self._busy = False

        self._publish_payload(payload)
        self.get_logger().info(
            f"名片结果已发布 request_id={request_id} status={payload['status']} "
            f"total_sec={time.monotonic() - started:.3f} topic={self._values['output_topic']}"
        )

    def _publish_payload(self, payload) -> None:
        output = String()
        output.data = serialize_result(payload)
        self._publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Img2DocNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
