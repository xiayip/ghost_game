"""ROS 2 node that publishes the largest visible face as the nearest proxy."""

from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .core import expand_face_box, scale_face_box, select_largest_face


class NearestFaceNode(Node):
    """Detect faces in RGB images and publish at most one Detection2D."""

    def __init__(self):
        super().__init__('nearest_face')

        self.declare_parameter(
            'image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('input_compressed', True)
        self.declare_parameter('detection_topic', '/nearest_face/detection')
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

        self.image_topic = self.get_parameter('image_topic').value
        self.input_compressed = bool(
            self.get_parameter('input_compressed').value)
        self.detection_topic = self.get_parameter('detection_topic').value
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
            Detection2DArray, self.detection_topic, 10)
        self.debug_publisher = None
        self.debug_compressed_publisher = None
        if self.publish_debug_image:
            self.debug_publisher = self.create_publisher(
                Image, self.debug_image_topic, qos_profile_sensor_data)
            self.debug_compressed_publisher = self.create_publisher(
                CompressedImage, self.debug_compressed_topic,
                qos_profile_sensor_data)

        input_type = CompressedImage if self.input_compressed else Image
        self.image_subscription = self.create_subscription(
            input_type, self.image_topic, self._image_callback,
            qos_profile_sensor_data)
        self._last_processed_at = 0.0

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

    def _image_callback(self, message):
        now = time.monotonic()
        processing_period = 1.0 / self.max_processing_rate
        # A 30 Hz source reaches a nominal 15 Hz deadline at 66.67 ms. Small
        # scheduling jitter otherwise rejects the second frame and produces a
        # 66/100 ms cadence. A 10% tolerance keeps the intended every-other-
        # frame schedule without allowing unbounded processing.
        if now - self._last_processed_at < processing_period * 0.90:
            return
        self._last_processed_at = now

        try:
            bgr = (
                self._compressed_image_to_bgr(message)
                if self.input_compressed else self._image_to_bgr(message)
            )
            height, width = bgr.shape[:2]
            if width > self.detector_input_width:
                detector_width = self.detector_input_width
                detector_height = max(
                    1, int(round(height * detector_width / width)))
                detector_image = cv2.resize(
                    bgr, (detector_width, detector_height),
                    interpolation=cv2.INTER_AREA)
            else:
                detector_image = bgr
                detector_height, detector_width = height, width
            self.detector.setInputSize((detector_width, detector_height))
            _, faces = self.detector.detect(detector_image)
            face_box = select_largest_face(
                faces, detector_image.shape, self.score_threshold)
            face_box = scale_face_box(
                face_box, bgr.shape,
                width / detector_width,
                height / detector_height,
            )
            selected = (
                expand_face_box(face_box, bgr.shape, *self.bbox_expansion)
                if face_box is not None else None
            )
        except Exception as error:  # keep the camera callback alive
            self.get_logger().error(
                f'Face detection failed: {error}', throttle_duration_sec=2.0)
            return

        output = Detection2DArray()
        output.header = message.header
        if selected is not None:
            detection = Detection2D()
            detection.header = message.header
            detection.id = 'nearest_face'
            detection.bbox.center.position.x = selected.x + selected.width / 2.0
            detection.bbox.center.position.y = selected.y + selected.height / 2.0
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = float(selected.width)
            detection.bbox.size_y = float(selected.height)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = 'face'
            hypothesis.hypothesis.score = float(selected.score)
            detection.results.append(hypothesis)
            output.detections.append(detection)
        self.detection_publisher.publish(output)

        if self.debug_publisher is not None:
            canvas = bgr.copy()
            if selected is not None:
                top_left = (selected.x, selected.y)
                bottom_right = (
                    selected.x + selected.width,
                    selected.y + selected.height)
                cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(
                    canvas, f'nearest head crop {selected.score:.2f}',
                    (selected.x, max(20, selected.y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            # Avoid allocating and serializing a full uncompressed debug frame
            # when nobody subscribes. The Web dashboard uses only JPEG.
            if self.debug_publisher.get_subscription_count() > 0:
                debug = Image()
                debug.header = message.header
                debug.height = canvas.shape[0]
                debug.width = canvas.shape[1]
                debug.encoding = 'bgr8'
                debug.is_bigendian = 0
                debug.step = canvas.shape[1] * 3
                debug.data = canvas.tobytes()
                self.debug_publisher.publish(debug)

            encoded_ok, encoded = cv2.imencode(
                '.jpg', canvas,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if encoded_ok:
                compressed = CompressedImage()
                compressed.header = message.header
                compressed.format = 'jpeg'
                compressed.data = encoded.tobytes()
                self.debug_compressed_publisher.publish(compressed)
            else:
                self.get_logger().error(
                    'Failed to encode nearest-face debug JPEG',
                    throttle_duration_sec=2.0)


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
