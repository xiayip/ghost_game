# Validation

Target environment: Ubuntu 24.04, ROS 2 Jazzy, Gemini 305 RGB topic.

Completed on 2026-09-22:

1. The detector and `ghost_game` meta package build successfully with symlink install.
2. Five core selection unit tests pass.
3. YuNet 2023 model loads and runs under OpenCV 4.13.0.
4. Live Gemini 305 `rgb8` input is decoded at 848 x 530.
5. Live `/nearest_face/detection` and `/nearest_face/debug_image` publishers
   are present; no-face frames publish an empty detection array.
6. An isolated end-to-end replay of a real Gemini frame containing two faces
   selected the larger clipped face and published one `Detection2D` with
   center `(805.0, 40.5)`, size `(86.0, 81.0)`, score `0.7231`, source stamp,
   and `camera_color_optical_frame`.
7. The live-frame comparison selected `score_threshold: 0.70`: 0.80 rejected
   both visible faces, while 0.70 detected both before largest-area selection.
8. `/nearest_face/debug_image/compressed` publishes a valid 848 x 530 JPEG;
   `ghost_game_web_monitor` subscribes to it and `/api/camera.jpg` was visually
   verified to contain the green selected-face bbox and confidence label.

The largest face is only an RGB distance proxy. This package does not claim
metric person range or identity tracking.
