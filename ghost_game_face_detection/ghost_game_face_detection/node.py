"""ROS 2 node that publishes the largest visible face as the nearest proxy."""

from pathlib import Path
import json
import time
from uuid import uuid4

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .core import expand_face_box, scale_face_box, valid_face_boxes
from .runtime import FaceTracker, LatestWorker, source_age_seconds


class NearestFaceNode(Node):
    """Detect faces in RGB images and publish at most one Detection2D."""

    def __init__(self):
        super().__init__('nearest_face')

        self.declare_parameter(
            'image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('input_compressed', True)
        self.declare_parameter('detection_topic', '/nearest_face/detection')
        self.declare_parameter('tracking_topic', '/nearest_face/tracking')
        self.declare_parameter('debug_image_topic', '/nearest_face/debug_image')
        self.declare_parameter(
            'debug_compressed_topic', '/nearest_face/debug_image/compressed')
        self.declare_parameter('model_path', '')
        self.declare_parameter('score_threshold', 0.70)
        self.declare_parameter('nms_threshold', 0.30)
        self.declare_parameter('top_k', 5000)
        # YuNet locates the face rather than the full head. Expand the final
        # selected box so downstream image crops include hair, ears, jaw, and
        # some neck for 3D reconstruction.
        self.declare_parameter('bbox_expand_left_ratio', 0.35)
        self.declare_parameter('bbox_expand_right_ratio', 0.35)
        self.declare_parameter('bbox_expand_top_ratio', 0.55)
        self.declare_parameter('bbox_expand_bottom_ratio', 0.30)
        self.declare_parameter('max_processing_rate', 30.0)
        self.declare_parameter('detector_input_width', 424)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('max_frame_age', 0.15)
        self.declare_parameter('debug_rate_hz', 5.0)
        self.declare_parameter('debug_max_width', 640)
        self.declare_parameter('track_lost_timeout', 0.4)

        self.image_topic = self.get_parameter('image_topic').value
        self.input_compressed = bool(
            self.get_parameter('input_compressed').value)
        self.detection_topic = self.get_parameter('detection_topic').value
        self.tracking_topic = self.get_parameter('tracking_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.debug_compressed_topic = self.get_parameter(
            'debug_compressed_topic').value
        score_threshold = float(self.get_parameter('score_threshold').value)
        nms_threshold = float(self.get_parameter('nms_threshold').value)
        top_k = int(self.get_parameter('top_k').value)
        self.bbox_expansion = (
            float(self.get_parameter('bbox_expand_left_ratio').value),
            float(self.get_parameter('bbox_expand_right_ratio').value),
            float(self.get_parameter('bbox_expand_top_ratio').value),
            float(self.get_parameter('bbox_expand_bottom_ratio').value),
        )
        self.max_processing_rate = float(
            self.get_parameter('max_processing_rate').value)
        self.detector_input_width = int(
            self.get_parameter('detector_input_width').value)
        self.publish_debug_image = bool(
            self.get_parameter('publish_debug_image').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.max_frame_age = float(self.get_parameter('max_frame_age').value)
        self.debug_rate_hz = float(self.get_parameter('debug_rate_hz').value)
        self.debug_max_width = int(self.get_parameter('debug_max_width').value)
        self.track_lost_timeout = float(self.get_parameter('track_lost_timeout').value)
        if any(not np.isfinite(v) or v <= 0 for v in (
                self.max_frame_age, self.debug_rate_hz, self.debug_max_width,
                self.track_lost_timeout)):
            raise ValueError('Age, debug and tracking limits must be finite and positive')
        if self.tracking_topic == self.detection_topic:
            raise ValueError('tracking_topic must differ from crop detection_topic')

        if not 0.0 < score_threshold <= 1.0:
            raise ValueError('score_threshold must be in (0, 1]')
        if not 0.0 < nms_threshold <= 1.0:
            raise ValueError('nms_threshold must be in (0, 1]')
        if top_k <= 0:
            raise ValueError('top_k must be positive')
        if any(not np.isfinite(value) or value < 0.0
               for value in self.bbox_expansion):
            raise ValueError(
                'bbox expansion ratios must be finite and non-negative')
        if self.max_processing_rate <= 0.0:
            raise ValueError('max_processing_rate must be positive')
        if self.detector_input_width <= 0:
            raise ValueError('detector_input_width must be positive')
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in [1, 100]')

        default_model = (
            'yunet_2026may.onnx'
            if int(cv2.__version__.split('.')[0]) >= 5
            else 'yunet_2023mar.onnx'
        )
        configured_model = str(self.get_parameter('model_path').value).strip()
        if not configured_model:
            raise ValueError(
                'model_path is required and may point to a YuNet ONNX file '
                'or to a directory containing the bundled model versions')
        model_path = Path(configured_model)
        if model_path.is_dir():
            model_path = model_path / default_model
        if not model_path.is_file():
            raise FileNotFoundError(f'YuNet model not found: {model_path}')

        self.score_threshold = score_threshold
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), '', (320, 320), score_threshold,
            nms_threshold, top_k)
        # Fail at startup if OpenCV and the bundled model are incompatible.
        self.detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

        self.detection_publisher = self.create_publisher(
            Detection2DArray, self.detection_topic, 1)
        self.tracking_publisher = self.create_publisher(
            Detection2DArray, self.tracking_topic, 1)
        self.metrics_publisher = self.create_publisher(String, '~/latency', 1)
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.debug_publisher = None
        self.debug_compressed_publisher = None
        if self.publish_debug_image:
            self.debug_publisher = self.create_publisher(
                Image, self.debug_image_topic, sensor_qos)
            self.debug_compressed_publisher = self.create_publisher(
                CompressedImage, self.debug_compressed_topic,
                sensor_qos)

        input_type = CompressedImage if self.input_compressed else Image
        self.image_subscription = self.create_subscription(
            input_type, self.image_topic, self._image_callback,
            sensor_qos)
        self.tracker = FaceTracker(self.track_lost_timeout)
        self.track_namespace = uuid4().hex[:8]
        self.worker = LatestWorker(self._process_image, self._processing_error, self.max_processing_rate)
        self.debug_worker = LatestWorker(self._render_debug, lambda job,error: None, self.debug_rate_hz)
        self._last_header = None
        self._last_stamp = 0
        self._last_clock_ns = 0
        self._last_debug = 0.
        self._last_invalid = 0.
        self._metrics = {}
        self._sequence = 0
        self._epoch = 0
        self._worker_epoch = 0
        self._flush_timer = self.create_timer(.005, self._flush)
        self._metrics_timer = self.create_timer(1., self._publish_metrics)

        self.get_logger().info(
            f'Nearest-face detector ready: input={self.image_topic}, '
            f'compressed_input={self.input_compressed}, '
            f'output={self.detection_topic}, OpenCV={cv2.__version__}, '
            f'model={model_path}, '
            f'detector_input_width={self.detector_input_width}, '
            'selection=largest visible face (RGB distance proxy), '
            'head_crop_expansion='
            f'(left={self.bbox_expansion[0]:.2f}, '
            f'right={self.bbox_expansion[1]:.2f}, '
            f'top={self.bbox_expansion[2]:.2f}, '
            f'bottom={self.bbox_expansion[3]:.2f})')

    @staticmethod
    def _image_to_bgr(message):
        """Decode common 8-bit ROS image encodings without cv_bridge.

        The workspace currently has a NumPy-2/cv_bridge ABI mismatch, and the
        camera publishes rgb8. Handling row stride explicitly also keeps this
        correct for images with padding.
        """
        encoding = message.encoding.lower()
        channels = 1 if encoding in ('mono8', '8uc1') else 3
        if encoding not in ('rgb8', 'bgr8', 'mono8', '8uc1'):
            raise ValueError(f'unsupported image encoding: {message.encoding}')
        minimum_step = message.width * channels
        if message.step < minimum_step:
            raise ValueError(
                f'invalid image step {message.step}, expected >= {minimum_step}')
        expected_bytes = message.step * message.height
        raw = np.frombuffer(message.data, dtype=np.uint8)
        if raw.size < expected_bytes:
            raise ValueError(
                f'truncated image data: {raw.size} < {expected_bytes}')
        rows = raw[:expected_bytes].reshape(message.height, message.step)
        pixels = rows[:, :minimum_step]
        if channels == 1:
            mono = pixels.reshape(message.height, message.width)
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        image = pixels.reshape(message.height, message.width, 3)
        if encoding == 'rgb8':
            return np.ascontiguousarray(image[:, :, ::-1])
        return np.ascontiguousarray(image)

    @staticmethod
    def _compressed_image_to_bgr(message):
        """Decode the camera transport JPEG directly into OpenCV BGR."""
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        if encoded.size == 0:
            raise ValueError('compressed image data is empty')
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(
                f'failed to decode compressed image format={message.format!r}')
        return image

    @staticmethod
    def _stamp(header):
        return header.stamp.sec*1_000_000_000 + header.stamp.nanosec

    def _age(self, header):
        return source_age_seconds(self._stamp(header), self.get_clock().now().nanoseconds, self.max_frame_age)

    def _image_callback(self, message):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self._last_clock_ns:
            self._epoch += 1
            self._last_stamp = 0
        self._last_clock_ns = now_ns
        stamp = self._stamp(message.header)
        if stamp <= self._last_stamp:
            return
        self._last_stamp = stamp
        self._last_header = message.header
        if not message.header.frame_id or self._age(message.header) is None:
            self._publish_boxes(message.header, None, None, '')
            return
        self._sequence += 1
        self.worker.pending.put((message, time.perf_counter(), self._sequence, self._epoch))

    def _processing_error(self, job, error):
        return dict(job=job, nearest=None, target=None, track_id='', bgr=None,
                    metrics={'reason':str(error), 'processing_ms':None})

    def _process_image(self, job):
        message, received, sequence, epoch = job
        started = time.perf_counter()
        if epoch != self._worker_epoch:
            self.tracker = FaceTracker(self.track_lost_timeout)
            self._worker_epoch = epoch
        if self._age(message.header) is None:
            return self._processing_error(job, 'stale_before_inference')
        bgr = (self._compressed_image_to_bgr(message) if self.input_compressed else self._image_to_bgr(message))
        height,width = bgr.shape[:2]
        detector_width = min(width, self.detector_input_width)
        detector_height = max(1, round(height*detector_width/width))
        sample = (cv2.resize(bgr,(detector_width,detector_height),interpolation=cv2.INTER_AREA)
                  if detector_width != width else bgr)
        self.detector.setInputSize((detector_width,detector_height))
        infer_started = time.perf_counter()
        _,faces = self.detector.detect(sample)
        inference_ms = (time.perf_counter()-infer_started)*1000
        candidates = [scale_face_box(box,bgr.shape,width/detector_width,height/detector_height)
                      for box in valid_face_boxes(faces,sample.shape,self.score_threshold)]
        if self._age(message.header) is None:
            return self._processing_error(job, 'stale_after_inference')
        face_box = max(candidates,key=lambda b:(b.area,b.score,-b.x,-b.y)) if candidates else None
        nearest = expand_face_box(face_box,bgr.shape,*self.bbox_expansion) if face_box else None
        target,track_id,reason = self.tracker.update(candidates,self._stamp(message.header)/1e9)
        track_id = f'{self.track_namespace}:{epoch}:{track_id}' if track_id else ''
        return dict(job=job,nearest=nearest,target=target,track_id=track_id,bgr=bgr,
                    metrics=dict(reason=reason, sequence=sequence, inference_ms=inference_ms,
                                 processing_ms=(time.perf_counter()-started)*1000,
                                 input_to_worker_ms=max(0.,(started-received)*1000),
                                 source_width=width, source_height=height))

    @staticmethod
    def _box_message(header, box, track_id):
        output = Detection2DArray()
        output.header = header
        if box is not None:
            detection = Detection2D()
            detection.header = header
            detection.id = track_id
            detection.bbox.center.position.x = box.x+box.width/2.
            detection.bbox.center.position.y = box.y+box.height/2.
            detection.bbox.center.theta = 0.
            detection.bbox.size_x, detection.bbox.size_y = float(box.width),float(box.height)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = 'face'
            hypothesis.hypothesis.score = float(box.score)
            detection.results.append(hypothesis)
            output.detections.append(detection)
        return output

    def _publish_boxes(self, header, nearest, target, track_id):
        self.detection_publisher.publish(self._box_message(header,nearest,'nearest_face'))
        self.tracking_publisher.publish(self._box_message(header,target,track_id))

    def _flush(self):
        now = time.monotonic()
        result = self.worker.completed.take()
        if result is not None and result['job'][3] == self._epoch:
            header = result['job'][0].header
            age = self._age(header)
            if age is None:
                result['nearest'] = result['target'] = None
                result['metrics']['reason'] = 'stale_at_publish'
            self._publish_boxes(header,result['nearest'],result['target'],result['track_id'])
            self._metrics = dict(result['metrics'], source_stamp_ns=self._stamp(header),
                                 publish_source_age_ms=None if age is None else age*1000,
                                 target_valid=result['target'] is not None)
            if (self.publish_debug_image and age is not None and result['bgr'] is not None
                    and now-self._last_debug >= 1/self.debug_rate_hz):
                raw = self.debug_publisher.get_subscription_count() > 0
                jpeg = self.debug_compressed_publisher.get_subscription_count() > 0
                if raw or jpeg:
                    self.debug_worker.pending.put((header,result['bgr'],result['nearest'],raw,jpeg))
                    self._last_debug = now
        if (self._last_header is not None and self._age(self._last_header) is None
                and now-self._last_invalid >= .05):
            self._publish_boxes(self._last_header,None,None,'')
            self._last_invalid = now
        debug = self.debug_worker.completed.take()
        if debug is not None:
            header,shape,raw,jpeg = debug
            # Debug is allowed a separate display budget; it never feeds control.
            if source_age_seconds(self._stamp(header), self.get_clock().now().nanoseconds, .5) is None:
                return
            if raw is not None:
                output = Image()
                output.header,output.height,output.width = header,shape[0],shape[1]
                output.encoding,output.step,output.data = 'bgr8',shape[1]*3,raw
                self.debug_publisher.publish(output)
            if jpeg is not None:
                output = CompressedImage()
                output.header,output.format,output.data = header,'jpeg',jpeg
                self.debug_compressed_publisher.publish(output)

    def _render_debug(self, job):
        header,bgr,box,want_raw,want_jpeg = job
        scale = min(1.,self.debug_max_width/bgr.shape[1])
        canvas = cv2.resize(bgr,(round(bgr.shape[1]*scale),round(bgr.shape[0]*scale))) if scale<1 else bgr.copy()
        if box is not None:
            cv2.rectangle(canvas,(round(box.x*scale),round(box.y*scale)),
                          (round((box.x+box.width)*scale),round((box.y+box.height)*scale)),(0,255,0),2)
        raw = canvas.tobytes() if want_raw else None
        jpeg = None
        if want_jpeg:
            ok,encoded = cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,self.jpeg_quality])
            if ok:
                jpeg = encoded.tobytes()
        return header,canvas.shape,raw,jpeg

    def _publish_metrics(self):
        message = String()
        stamp = self._metrics.get('source_stamp_ns',0)
        message.data = json.dumps(dict(self._metrics,
            pending_overwrites=self.worker.pending.overwritten,
            completed_overwrites=self.worker.completed.overwritten,
            latest_result_age_ms=(self.get_clock().now().nanoseconds-stamp)/1e6 if stamp else None))
        self.metrics_publisher.publish(message)

    def destroy_node(self):
        self.worker.close()
        self.debug_worker.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NearestFaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
