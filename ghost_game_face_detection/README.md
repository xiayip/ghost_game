# ghost_game_face_detection

ROS 2 Jazzy RGB nearest-face detector for the Zephyr Ghost Game.

This ROS package subscribes to the Gemini 305 JPEG transport, runs the
bundled OpenCV YuNet model, and treats the largest visible face bounding box
as the nearest-person proxy. RGB alone cannot provide metric distance.

## Interfaces

- Input: `/camera/color/image_raw/compressed` (`sensor_msgs/msg/CompressedImage`)
- Output: `/nearest_face/detection` (`vision_msgs/msg/Detection2DArray`)
- Debug image: `/nearest_face/debug_image` (`sensor_msgs/msg/Image`)
- Web/debug JPEG: `/nearest_face/debug_image/compressed`
  (`sensor_msgs/msg/CompressedImage`)

The detection array contains either zero detections (no qualifying face) or
one detection with ID `nearest_face`, class ID `face`, confidence, source
image header, and a pixel-space `BoundingBox2D`. The published box is expanded
from YuNet's facial region to include the full head for downstream cropping.
By default it adds 35% of face width on both sides, 55% of face height above,
and 30% below, then clips the result to the source-image boundary.

`vision_msgs` is an official ROS perception interface, so this package does
not define or generate a custom message.

## Build and run

```bash
cd /workspaces/zephyr-dev/zephyr_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select ghost_game_face_detection
source install/setup.bash
ros2 launch ghost_game_face_detection face_detection.launch.py \
  model_path:=$(ros2 pkg prefix ghost_game)/share/ghost_game/models
```

Inspect the result and debug image:

```bash
ros2 topic echo /nearest_face/detection
ros2 run rqt_image_view rqt_image_view /nearest_face/debug_image
```

The Ghost Game launch starts this node by default. Disable it with:

```bash
ros2 launch ghost_game ghost_game.launch.py enable_face_detection:=false
```

## Notes

- Selection is recalculated every processed frame; there is no identity
  tracking or hysteresis.
- `max_processing_rate` limits CPU usage independently of camera frame rate.
- `bbox_expand_left_ratio`, `bbox_expand_right_ratio`,
  `bbox_expand_top_ratio`, and `bbox_expand_bottom_ratio` control the full-head
  crop margins. Set all four to `0.0` to publish the original face box.
- Compressed camera input is decoded directly with OpenCV. Set
  `input_compressed: false` to use raw `rgb8`, `bgr8`, or `mono8` images
  without `cv_bridge` when commissioning another camera.
- The `ghost_game` meta package owns the bundled YuNet models in its
  `models/` directory and
  passes the installed model directory through `model_path`. OpenCV 4 selects
  the 2023 model and OpenCV 5 selects the 2026 model. This package itself
  stays model-agnostic and can also receive a direct ONNX file path.
