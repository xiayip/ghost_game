#!/usr/bin/env python3
"""
Ghost Game Web Monitor - serves the cyberpunk-styled progress dashboard
(web/index.html + style.css + app.js + viewer.js) over plain HTTP, backed by
ghost_game_node's ~/state topic. Stdlib http.server only - no extra web
framework dependency for the state feed itself.

Never forwards the 'targets' field even if present in ~/state (the
dashboard only ever reads 'progress'/'distances'/'locked', never the raw
secret angles) - see ghost_game_node.py's publish_debug_distances /
debug_reveal_targets for what actually controls what's on the wire.

Also expands the real arm URDF (via xacro, same as robot_state_publisher
would) once at startup and proxies its mesh files, so the browser's 3D
panel can render the actual CAD meshes instead of a schematic skeleton -
see /robot/robot.urdf and /robot/pkg/<package>/<relpath> below.

Also proxies nearest_face's annotated compressed stream (sensor_msgs/
CompressedImage, JPEG) so the dashboard shows the selected face bounding box
- see /api/camera.jpg below.

The same bridge subscribes to ghost_tts's accepted-caption stream and exposes
the latest caption as JSON. The browser therefore shows exactly the line the
voice worker receives, without duplicating the game's stage script.

It also serves a validated local GLB fallback and subscribes to a generated
model URL. Remote models are downloaded and cached by the bridge so the
browser uses a same-origin URL and never sees signed upstream query strings.
"""

import functools
import http.server
import json
import mimetypes
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

try:
    import xacro
except ImportError:
    xacro = None


def _expand_urdf(package, xacro_relpath, mappings, logger):
    """Run the given package's xacro file through xacro.process_file, the
    same way robot_state_publisher does, so the browser gets a plain URDF
    with real mesh references - no need to hand-maintain a copy here."""
    if xacro is None:
        logger.warn('xacro not importable - 3D robot model disabled')
        return None
    try:
        share_dir = get_package_share_directory(package)
        xacro_path = os.path.join(share_dir, xacro_relpath)
        doc = xacro.process_file(xacro_path, mappings=mappings)
        return doc.toxml()
    except Exception as error:  # noqa: BLE001 - report and degrade gracefully
        logger.warn(f'Could not expand {package}/{xacro_relpath} for the 3D model: {error}')
        return None


def _resolve_package_file(package, relpath):
    """package://<package>/<relpath> -> absolute path inside that package's
    share dir, or None if it doesn't resolve/escapes the share dir.

    Deliberately does NOT call Path.resolve() on the final path: with
    --symlink-install, share dirs are symlinks into src/, and resolving would
    make a legitimate mesh path look like it "escapes" share_dir. Blocking
    '..' segments and absolute overrides lexically, before joining, is
    enough - Path('/a') / '/etc/passwd' would otherwise discard the base.
    """
    if relpath.startswith('/') or '..' in relpath.split('/'):
        return None
    try:
        share_dir = get_package_share_directory(package)
    except PackageNotFoundError:
        return None
    candidate = Path(share_dir) / relpath
    return candidate if candidate.is_file() else None


class _StateStore:
    """Thread-safe holder for the latest ~/state JSON string."""

    def __init__(self):
        self._lock = threading.Lock()
        self._json = '{}'

    def set(self, json_text):
        with self._lock:
            self._json = json_text

    def get(self):
        with self._lock:
            return self._json


class _CameraStore:
    """Thread-safe holder for the latest JPEG frame bytes (or None if no
    frame has arrived yet)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None

    def set(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes

    def get(self):
        with self._lock:
            return self._jpeg


class _TtsStore:
    """Thread-safe latest-value store for the web voice-caption endpoint."""

    def __init__(self):
        self._lock = threading.Lock()
        self._text = ''
        self._sequence = 0
        self._received_at = None

    def set(self, text):
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._text = text
            self._sequence += 1
            self._received_at = time.time()

    def get_json(self):
        with self._lock:
            payload = {
                'text': self._text,
                'sequence': self._sequence,
                'received_at': self._received_at,
            }
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


class _MeshStore:
    """Latest GLB model, seeded from disk and replaceable by a ROS URL.

    Remote files are downloaded by the ROS node into this bounded cache. The
    browser always fetches the same-origin /api/mesh/model.glb endpoint, so a
    signed model URL does not leak to the page and the upstream server does
    not need permissive CORS headers.
    """

    def __init__(self, local_path='', max_bytes=100 * 1024 * 1024,
                 timeout=60.0):
        self._lock = threading.Lock()
        self._data = None
        self._status = 'unavailable'
        self._message = 'waiting_for_model'
        self._source = None
        self._version = 0
        self._updated_at = None
        self._generation = 0
        self.max_bytes = int(max_bytes)
        self.timeout = float(timeout)
        if self.max_bytes <= 0 or self.timeout <= 0:
            raise ValueError('mesh max_bytes and timeout must be positive')
        if str(local_path).strip():
            self.load_local(local_path)

    @staticmethod
    def _validate_glb(data):
        if len(data) < 12 or data[:4] != b'glTF':
            raise ValueError('not_a_glb_file')
        version = int.from_bytes(data[4:8], 'little')
        declared_size = int.from_bytes(data[8:12], 'little')
        if version != 2:
            raise ValueError('unsupported_glb_version')
        if declared_size != len(data):
            raise ValueError('glb_length_mismatch')

    def _commit(self, data, source, generation=None):
        self._validate_glb(data)
        if len(data) > self.max_bytes:
            raise ValueError('mesh_exceeds_size_limit')
        with self._lock:
            if generation is not None and generation != self._generation:
                return False
            self._data = bytes(data)
            self._source = source
            self._status = 'ready'
            self._message = ''
            self._version += 1
            self._updated_at = time.time()
            return True

    def load_local(self, local_path):
        path = Path(local_path).expanduser()
        try:
            size = path.stat().st_size
            if size > self.max_bytes:
                raise ValueError('mesh_exceeds_size_limit')
            data = path.read_bytes()
            self._commit(data, 'local')
            return True
        except Exception as error:  # noqa: BLE001 - status is exposed to UI
            with self._lock:
                self._status = 'error'
                self._message = f'local_model_error:{type(error).__name__}'
            return False

    def update_from_url(self, url, opener=None):
        """Download one published URL. Intended to run on a daemon thread."""
        url = str(url).strip()
        with self._lock:
            self._generation += 1
            generation = self._generation
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme not in ('http', 'https') or not parsed.netloc or
                parsed.username is not None or parsed.password is not None):
            with self._lock:
                self._status = 'error'
                self._message = 'invalid_model_url'
            return False

        with self._lock:
            self._status = 'downloading'
            self._message = ''

        request = urllib.request.Request(
            url,
            headers={
                'Accept': 'model/gltf-binary,application/octet-stream',
                'User-Agent': 'ghost-game-mesh-bridge/1.0',
            })
        open_url = opener or urllib.request.urlopen
        try:
            with open_url(request, timeout=self.timeout) as response:
                content_length = response.headers.get('Content-Length')
                if (content_length is not None and
                        int(content_length) > self.max_bytes):
                    raise ValueError('mesh_exceeds_size_limit')
                chunks = []
                total = 0
                while True:
                    chunk = response.read(min(1024 * 1024,
                                              self.max_bytes + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise ValueError('mesh_exceeds_size_limit')
            return self._commit(b''.join(chunks), 'published_url', generation)
        except Exception as error:  # noqa: BLE001 - preserve last good model
            with self._lock:
                if generation == self._generation:
                    self._status = 'error'
                    self._message = f'model_download_error:{type(error).__name__}'
            return False

    def get_model(self):
        with self._lock:
            return self._data

    def get_json(self):
        with self._lock:
            payload = {
                'status': self._status,
                'version': self._version,
                'bytes': len(self._data) if self._data is not None else 0,
                'updated_at': self._updated_at,
                'source': self._source,
                'message': self._message,
                'model_url': (
                    f'/api/mesh/model.glb?v={self._version}'
                    if self._data is not None else None),
            }
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


class _ControlCall:
    """One HTTP request waiting for its ROS Trigger service response."""

    def __init__(self, command):
        self.command = command
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._expired = False
        self.status = 500
        self.payload = None

    def finish(self, status, ok, message):
        with self._lock:
            if self._expired:
                return
            self.status = status
            self.payload = {
                'ok': bool(ok),
                'command': self.command,
                'message': str(message),
            }
            self.event.set()

    def expire(self):
        with self._lock:
            if self.event.is_set():
                return False
            self._expired = True
            return True

    @property
    def expired(self):
        with self._lock:
            return self._expired


class _ControlBridge:
    """Bounded handoff from HTTP server threads to the ROS executor thread."""

    COMMANDS = frozenset(('start', 'abort', 'return_home', 'mock_solve'))

    def __init__(self, max_pending=8):
        self._pending = queue.Queue(maxsize=max_pending)

    def request(self, command, timeout=6.0):
        if command not in self.COMMANDS:
            return 404, {
                'ok': False,
                'command': command,
                'message': 'Unknown control command',
            }
        call = _ControlCall(command)
        try:
            self._pending.put_nowait(call)
        except queue.Full:
            return 429, {
                'ok': False,
                'command': command,
                'message': 'Control queue is busy',
            }
        if not call.event.wait(timeout):
            call.expire()
            return 504, {
                'ok': False,
                'command': command,
                'message': 'ROS service response timed out',
            }
        return call.status, call.payload

    def take_nowait(self):
        return self._pending.get_nowait()

    def take(self, timeout=None):
        return self._pending.get(timeout=timeout)


class _ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    # Default request_queue_size (5) drops/resets connections when the
    # browser fires ~30 concurrent mesh fetches at page load (surfaces as
    # net::ERR_CONTENT_LENGTH_MISMATCH / ERR_ABORTED in the browser).
    request_queue_size = 128


class _StateHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    state_store: _StateStore = None    # bound by GhostGameWebMonitor before serving
    camera_store: _CameraStore = None  # bound by GhostGameWebMonitor before serving
    tts_store: _TtsStore = None        # bound by GhostGameWebMonitor before serving
    mesh_store: _MeshStore = None      # local fallback + latest URL model
    control_bridge: _ControlBridge = None
    urdf_text: str = None              # expanded URDF XML, or None if unavailable
    protocol_version = 'HTTP/1.1'    # keep-alive, so ~30 mesh fetches don't need 30 fresh sockets

    def end_headers(self):
        # Applies to every response, including the inherited static-file
        # serving of index.html/style.css/app.js/viewer.js - without this,
        # browsers happily cache viewer.js across edits and you end up
        # staring at stale JS wondering why your change "didn't work".
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def _send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Strip the query string first - app.js cache-busts /api/camera.jpg
        # with ?t=<timestamp>, which would never exact-match self.path below.
        path = self.path.split('?', 1)[0]
        if path in ('/api/state', '/api/state/'):
            self._send_bytes(self.state_store.get().encode('utf-8'), 'application/json; charset=utf-8')
            return
        if path in ('/api/tts', '/api/tts/'):
            payload = self.tts_store.get_json() if self.tts_store else '{}'
            self._send_bytes(payload.encode('utf-8'), 'application/json; charset=utf-8')
            return
        if path in ('/api/mesh', '/api/mesh/',
                    '/api/mesh/info', '/api/mesh/info/'):
            payload = self.mesh_store.get_json() if self.mesh_store else '{}'
            self._send_bytes(
                payload.encode('utf-8'), 'application/json; charset=utf-8')
            return
        if path in ('/api/mesh/model.glb', '/api/mesh/model.glb/'):
            model = self.mesh_store.get_model() if self.mesh_store else None
            if model is None:
                self.send_error(404, 'No reconstructed model available')
                return
            self._send_bytes(model, 'model/gltf-binary')
            return
        if path in ('/api/camera.jpg', '/api/camera.jpg/'):
            jpeg_bytes = self.camera_store.get() if self.camera_store else None
            if jpeg_bytes is None:
                self.send_error(404, 'No camera frame received yet')
                return
            self._send_bytes(jpeg_bytes, 'image/jpeg')
            return
        if path in ('/robot/robot.urdf', '/robot/robot.urdf/'):
            if self.urdf_text is None:
                self.send_error(404, 'Robot model unavailable (xacro expansion failed at startup)')
                return
            self._send_bytes(self.urdf_text.encode('utf-8'), 'application/xml; charset=utf-8')
            return
        if path.startswith('/robot/pkg/'):
            pkg_and_rel = path[len('/robot/pkg/'):].split('/', 1)
            if len(pkg_and_rel) == 2 and '..' not in pkg_and_rel[1]:
                resolved = _resolve_package_file(pkg_and_rel[0], pkg_and_rel[1])
                if resolved is not None:
                    content_type = mimetypes.guess_type(str(resolved))[0] or 'application/octet-stream'
                    self._send_bytes(resolved.read_bytes(), content_type)
                    return
            self.send_error(404, 'Mesh not found')
            return
        super().do_GET()

    def do_POST(self):
        path = self.path.split('?', 1)[0].rstrip('/')
        prefix = '/api/control/'
        if not path.startswith(prefix) or self.control_bridge is None:
            self.send_error(404, 'Control endpoint not found')
            return
        command = path[len(prefix):]
        status, payload = self.control_bridge.request(command)
        body = json.dumps(
            payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self._send_bytes(body, 'application/json; charset=utf-8', status=status)

    def log_message(self, format_str, *args):
        pass  # quiet - the ROS logger already announces startup; polling would flood stdout


class GhostGameWebMonitor(Node):

    def __init__(self):
        super().__init__('ghost_game_web_monitor')
        self.declare_parameter('state_topic', '/ghost_game_node/state')
        self.declare_parameter('tts_text_topic', '/ghost/tts/caption')
        self.declare_parameter('start_service', '/ghost_game_node/start')
        self.declare_parameter('abort_service', '/ghost_game_node/abort')
        self.declare_parameter(
            'return_home_service', '/ghost_game_node/return_home')
        self.declare_parameter(
            'mock_solve_service', '/ghost_game_node/mock_solve')
        self.declare_parameter('port', 8765)
        self.declare_parameter('enable_robot_model', True)
        self.declare_parameter('robot_description_package', 'zephyr_arm_description')
        self.declare_parameter('robot_description_xacro', 'urdf/zephyr_arm.urdf.xacro')
        self.declare_parameter('enable_camera', True)
        self.declare_parameter(
            'camera_topic', '/nearest_face/debug_image/compressed')
        self.declare_parameter('enable_mesh_model', True)
        self.declare_parameter(
            'mesh_model_path',
            '/workspaces/zephyr-dev/zephyr_ws/outputs/'
            'tripo_pbr_model_141bec5f-e771-4e61-863f-5c5b663daabe.glb')
        self.declare_parameter(
            'mesh_url_topic', '/ghost/reconstruction/model_url')
        self.declare_parameter('mesh_max_bytes', 100 * 1024 * 1024)
        self.declare_parameter('mesh_download_timeout', 60.0)

        state_topic = self.get_parameter('state_topic').value
        port = int(self.get_parameter('port').value)

        self._store = _StateStore()
        self.create_subscription(String, state_topic, self._on_state, 10)

        self._tts_store = _TtsStore()
        tts_text_topic = self.get_parameter('tts_text_topic').value
        self.create_subscription(String, tts_text_topic, self._on_tts_text, 10)

        self._control_bridge = _ControlBridge()
        self._control_clients = {
            'start': self.create_client(
                Trigger, self.get_parameter('start_service').value),
            'abort': self.create_client(
                Trigger, self.get_parameter('abort_service').value),
            'return_home': self.create_client(
                Trigger, self.get_parameter('return_home_service').value),
            'mock_solve': self.create_client(
                Trigger, self.get_parameter('mock_solve_service').value),
        }
        self._control_timer = self.create_timer(0.05, self._dispatch_controls)

        self._camera_store = _CameraStore()
        camera_topic = self.get_parameter('camera_topic').value
        if self.get_parameter('enable_camera').value:
            self.create_subscription(
                CompressedImage, camera_topic, self._on_camera, qos_profile_sensor_data)

        self._mesh_store = _MeshStore(
            self.get_parameter('mesh_model_path').value
            if self.get_parameter('enable_mesh_model').value else '',
            max_bytes=self.get_parameter('mesh_max_bytes').value,
            timeout=self.get_parameter('mesh_download_timeout').value)
        if self.get_parameter('enable_mesh_model').value:
            self.create_subscription(
                String, self.get_parameter('mesh_url_topic').value,
                self._on_mesh_url, 10)

        web_dir = get_package_share_directory('ghost_game_orchestrator') + '/web'
        _StateHTTPRequestHandler.state_store = self._store
        _StateHTTPRequestHandler.camera_store = self._camera_store
        _StateHTTPRequestHandler.tts_store = self._tts_store
        _StateHTTPRequestHandler.mesh_store = self._mesh_store
        _StateHTTPRequestHandler.control_bridge = self._control_bridge
        if self.get_parameter('enable_robot_model').value:
            _StateHTTPRequestHandler.urdf_text = _expand_urdf(
                self.get_parameter('robot_description_package').value,
                self.get_parameter('robot_description_xacro').value,
                {'use_mock_hardware': 'true'},
                self.get_logger())
        handler_cls = functools.partial(_StateHTTPRequestHandler, directory=web_dir)
        self._httpd = _ThreadingHTTPServer(('0.0.0.0', port), handler_cls)
        self._server_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._server_thread.start()

        model_status = (
            'loaded' if _StateHTTPRequestHandler.urdf_text
            else 'unavailable (skeleton fallback)')
        self.get_logger().info(
            f'Ghost game dashboard: http://localhost:{port}  '
            f'(watching {state_topic} and {tts_text_topic}, robot model '
            f'{model_status}, reconstructed mesh '
            f'{json.loads(self._mesh_store.get_json())["status"]})')

    def _on_state(self, msg: String):
        self._store.set(msg.data)

    def _on_camera(self, msg: CompressedImage):
        self._camera_store.set(bytes(msg.data))

    def _on_tts_text(self, msg: String):
        self._tts_store.set(msg.data)

    def _on_mesh_url(self, msg: String):
        if not msg.data.strip():
            return
        self.get_logger().info(
            'Received reconstructed-model URL; downloading GLB in background')

        def download():
            if self._mesh_store.update_from_url(msg.data):
                info = json.loads(self._mesh_store.get_json())
                self.get_logger().info(
                    f'Reconstructed GLB ready: {info["bytes"]} bytes, '
                    f'version={info["version"]}')
            else:
                info = json.loads(self._mesh_store.get_json())
                self.get_logger().warning(
                    f'Reconstructed GLB update failed: {info["message"]}')

        threading.Thread(target=download, daemon=True).start()

    def _dispatch_controls(self):
        for _ in range(8):
            try:
                call = self._control_bridge.take_nowait()
            except queue.Empty:
                return
            if call.expired:
                continue
            client = self._control_clients[call.command]
            if not client.service_is_ready():
                call.finish(
                    503, False,
                    f'ROS service for {call.command} is unavailable')
                continue
            future = client.call_async(Trigger.Request())
            future.add_done_callback(
                lambda completed, pending=call:
                self._finish_control(pending, completed))

    def _finish_control(self, call, future):
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001 - return ROS failure to UI
            call.finish(502, False, f'ROS service call failed: {error}')
            return
        if response is None:
            call.finish(502, False, 'ROS service returned no response')
            return
        call.finish(
            200 if response.success else 409,
            response.success,
            response.message,
        )

    def destroy_node(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GhostGameWebMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
