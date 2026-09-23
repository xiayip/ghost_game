#!/usr/bin/env python3
"""Run one isolated image -> FLUX style -> Tripo GLB timing probe.

The img2mesh ROS node must already be running with topics matching the CLI
arguments below.  The report deliberately omits signed asset URLs and secrets.
"""

import argparse
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import rclpy
import requests
from ghost_game_interfaces.msg import MeshResult
from img2mesh.image_codec import array_to_image_message
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import String


TERMINAL_STATUSES = {'success', 'error', 'failed', 'cancelled'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--timeout-sec', type=float, default=1200.0)
    parser.add_argument('--image-topic', default='/ghost_test/img2mesh/image')
    parser.add_argument('--prompt-topic', default='/ghost_test/img2mesh/prompt')
    parser.add_argument('--result-topic', default='/ghost_test/img2mesh/result')
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def file_uri_to_path(uri):
    parsed = urlparse(uri)
    if parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost'):
        return None
    return Path(unquote(parsed.path))


def validate_glb(path):
    with path.open('rb') as stream:
        header = stream.read(12)
    if len(header) != 12:
        raise RuntimeError('GLB header is truncated')
    magic, version, declared_length = struct.unpack('<4sII', header)
    actual_length = path.stat().st_size
    if magic != b'glTF':
        raise RuntimeError(f'invalid GLB magic: {magic!r}')
    if version != 2:
        raise RuntimeError(f'unsupported GLB version: {version}')
    if declared_length != actual_length:
        raise RuntimeError(
            f'GLB length mismatch: header={declared_length}, file={actual_length}')
    return {
        'magic': magic.decode('ascii'),
        'version': version,
        'declared_length': declared_length,
        'actual_length': actual_length,
    }


class Probe(Node):
    def __init__(self, args, pixels):
        super().__init__('img2mesh_pipeline_test_driver')
        self.args = args
        self.pixels = pixels
        self.started_monotonic = None
        self.started_utc = None
        self.finished_monotonic = None
        self.finished_utc = None
        self.result = None
        self.request_id = None
        self.source_stamp = None
        self.timeline = []
        self._last_event = None
        self._published = False
        self.image_publisher = self.create_publisher(
            type(array_to_image_message(pixels, 'bgr8')),
            args.image_topic,
            qos_profile_sensor_data,
        )
        self.prompt_publisher = self.create_publisher(
            String, args.prompt_topic, 10)
        result_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.result_subscription = self.create_subscription(
            MeshResult, args.result_topic, self._on_result, result_qos)
        self.connection_timer = self.create_timer(0.1, self._publish_once_ready)

    def _publish_once_ready(self):
        if self._published:
            return
        if (self.image_publisher.get_subscription_count() < 1
                or self.prompt_publisher.get_subscription_count() < 1):
            return
        image = array_to_image_message(self.pixels, 'bgr8')
        image.header.stamp = self.get_clock().now().to_msg()
        self.source_stamp = (
            int(image.header.stamp.sec), int(image.header.stamp.nanosec))
        self.started_monotonic = time.monotonic()
        self.started_utc = utc_now()
        self.image_publisher.publish(image)
        self.prompt_publisher.publish(String(data=self.args.prompt))
        self._published = True
        self.connection_timer.cancel()
        print('request_published image=1 prompt=1', flush=True)

    def _on_result(self, message):
        if not self._published:
            return
        message_stamp = (
            int(message.source_header.stamp.sec),
            int(message.source_header.stamp.nanosec),
        )
        if message_stamp != self.source_stamp:
            return
        # A previous TRANSIENT_LOCAL publisher may briefly replay its terminal
        # sample after a probe node restarts on the same topic.  The source
        # image timestamp above is the primary filter; bind the matching sample
        # to one request ID as a second guard.
        if self.request_id is None:
            if message.status != 'styling' or int(message.progress) != 0:
                return
            self.request_id = message.request_id
        elif message.request_id != self.request_id:
            return
        event = (message.status, int(message.progress))
        if event != self._last_event:
            elapsed = time.monotonic() - self.started_monotonic
            record = {
                'status': message.status,
                'progress': int(message.progress),
                'elapsed_sec': round(elapsed, 3),
            }
            self.timeline.append(record)
            print(
                f'status={message.status} progress={int(message.progress)} '
                f'elapsed_sec={elapsed:.3f}',
                flush=True,
            )
            self._last_event = event
        if message.status in TERMINAL_STATUSES:
            self.result = message
            self.finished_monotonic = time.monotonic()
            self.finished_utc = utc_now()


def stage_elapsed(timeline, status, progress=None):
    for event in timeline:
        if event['status'] == status and (
                progress is None or event['progress'] == progress):
            return event['elapsed_sec']
    return None


def render_markdown(report):
    stage_rows = '\n'.join(
        f"| {event['status']} | {event['progress']}% | "
        f"{event['elapsed_sec']:.3f} s |"
        for event in report['timeline'])
    return f"""# Image → styled image → Tripo GLB pipeline test

- Result: **{report['result']}**
- Started (UTC): `{report['started_utc']}`
- Finished (UTC): `{report['finished_utc']}`
- Input: `{report['input']['path']}` ({report['input']['width']}×{report['input']['height']}, {report['input']['bytes']} bytes)
- Model: `{report.get('model', '')}`
- Task ID: `{report.get('task_id', '')}`
- Time to styled image: `{report['timing'].get('time_to_style_complete_sec')}` s
- Time to Tripo queued: `{report['timing'].get('time_to_tripo_queued_sec')}` s
- Time to model URL: `{report['timing'].get('total_to_model_url_sec')}` s
- GLB download: `{report['timing'].get('glb_download_sec')}` s
- **Total to validated local GLB: `{report['timing'].get('total_to_local_glb_sec')}` s**
- Styled image: `{report.get('styled_image', {}).get('path', '')}`
- GLB: `{report.get('glb', {}).get('path', '')}`

| Pipeline state | Progress | Elapsed |
|---|---:|---:|
{stage_rows}

Signed model and preview URLs are intentionally omitted.
"""


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pixels = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if pixels is None:
        raise SystemExit(f'cannot read input image: {args.input}')

    report = {
        'result': 'running',
        'input': {
            'path': str(args.input.resolve()),
            'width': int(pixels.shape[1]),
            'height': int(pixels.shape[0]),
            'bytes': args.input.stat().st_size,
        },
        'prompt': args.prompt,
        'timeline': [],
        'timing': {},
    }
    exit_code = 1
    rclpy.init()
    node = Probe(args, pixels)
    deadline = time.monotonic() + args.timeout_sec
    try:
        while rclpy.ok() and node.result is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        report['timeline'] = node.timeline
        report['started_utc'] = node.started_utc
        report['finished_utc'] = node.finished_utc or utc_now()
        if node.result is None:
            report['result'] = 'timeout'
            report['error'] = (
                'pipeline result timed out' if node._published
                else 'ROS publishers did not find the isolated img2mesh node')
        elif node.result.status != 'success':
            report['result'] = node.result.status
            report['error'] = node.result.error_message
            report['request_id'] = node.result.request_id
            report['task_id'] = node.result.task_id
            report['model'] = node.result.model
        elif not node.result.model_url:
            report['result'] = 'error'
            report['error'] = 'success result did not include a model URL'
        else:
            report['request_id'] = node.result.request_id
            report['task_id'] = node.result.task_id
            report['model'] = node.result.model
            report['timing']['time_to_style_complete_sec'] = stage_elapsed(
                node.timeline, 'styling', 100)
            report['timing']['time_to_tripo_queued_sec'] = stage_elapsed(
                node.timeline, 'queued')
            model_elapsed = node.finished_monotonic - node.started_monotonic
            report['timing']['total_to_model_url_sec'] = round(model_elapsed, 3)

            style_path = file_uri_to_path(node.result.style_image_url)
            if style_path is not None and style_path.is_file():
                styled = cv2.imread(str(style_path), cv2.IMREAD_UNCHANGED)
                report['styled_image'] = {
                    'path': str(style_path),
                    'width': int(styled.shape[1]) if styled is not None else None,
                    'height': int(styled.shape[0]) if styled is not None else None,
                    'bytes': style_path.stat().st_size,
                }

            glb_path = args.output_dir / 'tripo_model.glb'
            download_started = time.monotonic()
            with requests.get(
                    node.result.model_url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with glb_path.open('wb') as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            download_elapsed = time.monotonic() - download_started
            glb = validate_glb(glb_path)
            glb['path'] = str(glb_path.resolve())
            report['glb'] = glb
            report['timing']['glb_download_sec'] = round(download_elapsed, 3)
            report['timing']['total_to_local_glb_sec'] = round(
                time.monotonic() - node.started_monotonic, 3)
            report['result'] = 'success'
            exit_code = 0
    except Exception as exc:
        report['result'] = 'error'
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    json_path = args.output_dir / 'pipeline_timing.json'
    markdown_path = args.output_dir / 'pipeline_timing.md'
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    markdown_path.write_text(render_markdown(report), encoding='utf-8')
    print(f'report_json={json_path}', flush=True)
    print(f'report_markdown={markdown_path}', flush=True)
    print(f'result={report["result"]}', flush=True)
    if report.get('error'):
        print(f'error={report["error"]}', flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
