# Ghost Game

This repository is split into seven ROS 2 packages with one-way ownership:

| Package | Responsibility |
| --- | --- |
| `ghost_game` | Meta package, unified launch, and the YuNet model assets used by the application. It contains no runtime Python logic. |
| `ghost_game_interfaces` | Shared ROS 2 interfaces, including the cancellable `Speak` action and detailed `MeshResult` progress message. |
| `ghost_game_face_detection` | RGB image input, nearest-face selection, `vision_msgs` bbox output, and annotated debug images. It does not know the game state. |
| `ghost_game_orchestrator` | Game state machine, controller/gripper coordination, success flow, and terminal/web monitors. It does not implement face detection. |
| `ghost_tts` | Switchable offline Piper or cloud Doubao speech, bounded FIFO/cancellation, cyberpunk effects, and host PipeWire/PulseAudio playback. It does not control the arm. |
| `flux_image_editor` | Asynchronous ROS bridge from the captured full-head crop to the FLUX HTTP image-editing service. Its prepared output feeds `img2mesh`. |
| `img2mesh` | Uploads each prepared portrait to Tripo, tracks generation progress, and publishes the completed signed GLB URL to the Web bridge. |

The dependency direction is `ghost_game -> {ghost_game_orchestrator,
ghost_game_face_detection, ghost_tts, flux_image_editor, img2mesh,
ghost_game_interfaces}`.
Runtime packages share only the interface package and communicate over ROS.

"Find the Ghost" arm interaction demo. The arm silently picks 6 secret joint
angles, goes compliant (damping-like), and the audience hand-guesses each
joint's angle. A joint that's held near its secret target for long enough
locks rigid (impedance hold); once all 6 are found the gripper opens and a
smooth, zero-endpoint-velocity/acceleration trajectory moves the arm to the
configured success pose through the impedance JTC. At that approximately
180-degree observation pose, Step 2 scans for a nearby stable face and aims
the wrist camera until the full-head bbox is centered. An optional dance can
run afterward. Everything runs as one background-thread state machine inside
`ghost_game_node` - no BT involved.

Step 2 keeps YuNet warm from launch but ignores detections until the success
pose has been measured and settled. It rejects stale boxes and boxes smaller
than `face_min_bbox_area_ratio`, requires the configured spatial-stability
dwell, then centers the target with a 60 Hz direct MIT impedance reference
stream on bounded `joint5` yaw and `joint4` pitch. Normalized image error
commands a bounded joint speed, so motion stays continuous between detector
frames.
During centering it rejects sudden bbox center/area jumps as a detector switch
to another visitor, then resumes scanning instead of chasing the new face.
When no acceptable face is visible, it repeats the configured serpentine
wrist scan. The default `face_search_timeout: 0.0` keeps searching until a
face is acquired or the operator requests abort/return-home. This uses bbox
area as the distance proxy because the current camera launch has unregistered
depth (`depth_registration: false`).

Once tracking has held the expanded bbox at image center, the orchestrator
publishes one reconstruction prompt. The detector continuously publishes that
same selected full-head region as `sensor_msgs/Image` on
`/nearest_face/head_crop`, and `flux_image_editor` pairs its latest crop with
the prompt without blocking arm motion. The edited image is published on
`/ghost/reconstruction/image` and its exact PNG bytes on
`/ghost/reconstruction/image_png`; JSON progress and the saved path appear on
`/ghost/reconstruction/status` and inside the orchestrator state as
`reconstruction`. `img2mesh` consumes each completed FLUX image, starts one
Tripo image-to-model task, publishes JSON progress on
`/ghost/reconstruction/mesh_status`, and sends the completed signed GLB URL
on `/ghost/reconstruction/model_url` to the Web bridge. The robot sequence
continues while FLUX and Tripo work asynchronously.
Every submitted full-head crop is saved before the HTTP request under
`/workspaces/zephyr-dev/zephyr_ws/outputs/ghost_face_captures/<request_id>_head_crop.png`.
The most recent capture is also available at the stable path
`/workspaces/zephyr-dev/zephyr_ws/outputs/ghost_face_captures/latest_head_crop.png`,
even when the FLUX inference service is offline. This directory is mounted
from the workspace, so captures survive container recreation.

Each important state transition also emits a Chinese voice cue through the
`/ghost/tts/speak` Action: game setup,
search start, every newly locked joint, all-found, face scan, dance, done,
return-home, stuck, and abort. Speech synthesis and playback run on the TTS
worker thread, so they never block controller switching or the arm loop.
Safety/return-home cues send `interrupt=true`, cancel active playback and
older queued speech, then announce the new state. The legacy text topic and
stop service remain available for manual testing and older clients.

The web dashboard keeps its camera panel hidden during the search and the
success-pose motion. It starts displaying camera frames only after the arm has
reached that pose (`success_pose_reached`, the `face_*` phases, `dancing`, or
`done`). At the same time it hides the joint-card list and Arm Pose 3D panel,
restoring them automatically when the game leaves those post-turn phases.

The dashboard also includes a PBR GLB viewer labelled “Ghost 三维重建体”.
The panel stays hidden until a stable face has been captured. It then shows a
single cumulative progress bar across FLUX preprocessing, Tripo upload/queue,
Tripo generation, GLB download, and browser parsing. A previous round's model
is hidden while a new visitor is being reconstructed, and the interactive
model replaces the progress display only after the new GLB parses successfully.
During integration the backend can be seeded with the local sample at
`/workspaces/zephyr-dev/zephyr_ws/outputs/tripo_pbr_model_141bec5f-e771-4e61-863f-5c5b663daabe.glb`.
The browser receives it from the same-origin `/api/mesh/model.glb` endpoint,
auto-fits it to the viewport, preserves its textures/materials, and supports
orbit, zoom, and slow automatic rotation.

`img2mesh` replaces that sample by publishing an HTTP(S) GLB URL as
`std_msgs/msg/String` on `/ghost/reconstruction/model_url`. The web bridge downloads the file on a
background thread, validates the GLB header and size, caches up to 100 MiB,
and updates the viewer without exposing a signed upstream URL to the browser
or requiring upstream CORS headers:

```bash
ros2 topic pub --once /ghost/reconstruction/model_url std_msgs/msg/String \
  "{data: 'https://example.invalid/generated/ghost.glb'}"
```

Override the temporary local model or topic with the unified launch arguments
`web_mesh_model_path` and `web_mesh_url_topic`.

The success turn is also checked against live joint feedback. If the
impedance JTC reports a goal-tolerance abort because of the real arm's small
static load residual, the dashboard can still enter the camera view only when
all joints are within `success_position_tolerance` and remain settled for
`success_settle_time`. A blocked or unfinished move therefore does not reveal
the camera.

After the last fragment is found, the controller ramps from the compliant
game lock gains to `success_stiffness`. This gives the load-bearing joints
enough authority for the display/vision pose without making the interactive
search phase harder to move by hand.

The meta launch also starts `ghost_game_face_detection` and `ghost_tts` by
default. The detector
consumes the Gemini 305 RGB stream and publishes the largest visible face
(the RGB-only nearest-person proxy) as an official
`vision_msgs/msg/Detection2DArray` on
`/nearest_face/detection`; the array contains zero or one detection. Set
`enable_face_detection:=false` to run the game without face detection.
The published bbox expands YuNet's facial region into a full-head crop for
downstream 3D reconstruction; its four directional margins are configurable
in `ghost_game_face_detection/config/gemini305.yaml` and are clipped to the
source-image boundary.
The detector also publishes its annotated JPEG on
`/nearest_face/debug_image/compressed`; `ghost_game_web_monitor` uses that as
its default Camera source, so the success screen shows the selected bbox.
The YuNet ONNX files live in this package's `models/` directory; the launch
passes that installed directory to the detector through `model_path`.

Only one controller, `mit_impedance_controller`, is used for both the free
and locked behavior per joint: free is just `stiffness=0`. See the module
docstring in `ghost_game_orchestrator/ghost_game_orchestrator/ghost_game_node.py`
for the full design rationale
(why not the JTC during search, the stale-setpoint fix in `_return_home`,
the stuck/blocked back-off, etc).

## Build

Install the offline TTS runtime and voice once inside the
`zephyr_dev_24.04-aarch64:latest` container. The model weights stay outside
the source tree:

```bash
bash /workspaces/zephyr-dev/zephyr_ws/src/zephyr_mission/ghost_game/scripts/install_hackathon_dependencies.sh
```

The script is intended for this hackathon container and is safe to rerun. It
installs `python3-pip`/PortAudio when missing, installs the pinned Python
packages, and downloads the voice only when its two model files are absent.
Even when invoked through root `docker exec`, it installs into the `admin`
user's home so the normal devcontainer launch can find the model. Set
`GHOST_TTS_USER` or `GHOST_TTS_MODEL_DIR` to override that behavior.

Equivalent manual commands:

```bash
python3 -m pip install --user --break-system-packages \
  -r /workspaces/zephyr-dev/zephyr_ws/src/zephyr_mission/ghost_game/ghost_tts/requirements.txt
mkdir -p ~/.local/share/ghost_tts
python3 -m piper.download_voices zh_CN-huayan-medium \
  --data-dir ~/.local/share/ghost_tts
```

`zh_CN-huayan-medium` is the default demo voice. Its model card labels the
dataset license as unknown, so use an explicitly licensed voice before
shipping a product; see `ghost_tts/THIRD_PARTY.md`.

```bash
source /opt/ros/jazzy/setup.bash
cd /workspaces/zephyr-dev/zephyr_ws
colcon build --symlink-install --packages-select \
  ghost_game_interfaces ghost_tts flux_image_editor img2mesh ghost_game_orchestrator \
  ghost_game_face_detection ghost_game
```

## Running it: which terminal runs what

**Terminal 1 - arm bringup** (controller_manager + hardware/mock, must already
be up before the game node can switch controllers):
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch zephyr_arm_bringup real_world.launch.py \
  activate_trajectory_controller:=false \
  enable_camera:=auto
```

**Terminal 2 - unified Ghost Game launch**:
```bash
source /workspaces/zephyr-dev/zephyr_ws/install/setup.bash
ros2 launch ghost_game ghost_game.launch.py
```
This starts the orchestrator, face detector, Piper TTS, the lightweight FLUX
ROS bridge, and `img2mesh`. Add
`enable_web_monitor:=true` to start the dashboard in the same launch, or
`enable_face_detection:=false` when no camera/detector is needed. Use
`enable_tts:=false` for silent operation. The TTS launch arguments are
`tts_backend` (`auto`, `piper`, or `doubao`), `tts_model`, `tts_preset` (`clean`,
`subtle`, `ghost`, or `machine`), and
`tts_audio_device` (`pulse` follows the host default speaker; a direct
PortAudio name/index such as `0` is also accepted). Watch this
terminal's log during testing - it's where "stuck/blocked",
"found joint", "gravity compensation ramping to 0", etc. get printed.
Use `enable_face_reconstruction:=false` when the FLUX service is not needed,
or set `face_reconstruction_server_url:=http://HOST:8090` when it runs on a
different machine. A missing inference server reports an asynchronous
reconstruction error and does not stop the gesture or game flow.
Set `TRIPO_API_KEY` before launch or place an ignored
`img2mesh/config/local_api.yaml` file locally. Use
`enable_mesh_reconstruction:=false` to skip Tripo while testing the robot.

For development, each functional package can also run independently:

```bash
ros2 run ghost_game_orchestrator ghost_game_node --ros-args \
  --params-file $(ros2 pkg prefix ghost_game_orchestrator)/share/ghost_game_orchestrator/config/ghost_game.yaml

ros2 launch ghost_game_face_detection face_detection.launch.py \
  model_path:=$(ros2 pkg prefix ghost_game)/share/ghost_game/models

ros2 launch ghost_tts ghost_tts.launch.py audio_device:=pulse preset:=ghost

# Cloud voice: export the three variables first, then switch the backend.
export DOUBAO_TTS_API_KEY='<API key from the Doubao Voice console>'
export DOUBAO_TTS_VOICE_TYPE='<S_... cloned voice ID>'
export DOUBAO_TTS_RESOURCE_ID='seed-icl-2.0'
ros2 launch ghost_tts ghost_tts.launch.py \
  backend:=auto audio_device:=pulse preset:=ghost

# Cancellable action request. Set interrupt=true to preempt active/queued speech.
ros2 action send_goal /ghost/tts/speak \
  ghost_game_interfaces/action/Speak \
  "{text: '紧急链路接管。', interrupt: true}" --feedback

ros2 run flux_image_editor flux_image_editor_node --ros-args \
  --params-file $(ros2 pkg prefix flux_image_editor)/share/flux_image_editor/config/ghost_game.yaml

# Legacy compatibility interface; the stop service cancels every job.
ros2 topic pub --once /ghost/tts/text std_msgs/msg/String \
  "{data: '正在潜入深网。意识端口已经接入。'}"
ros2 service call /ghost/tts/stop std_srvs/srv/Trigger {}
```

**Terminal 3 - live dashboard** (optional, only useful while
`publish_debug_distances: true`) - pick one:
```bash
source /workspaces/zephyr-dev/zephyr_ws/install/setup.bash

# Cyberpunk web dashboard (styled after zephyr_robot_design_system) - open
# the printed URL (http://localhost:8765 by default) in a browser. J1-J6
# progress bars (updated at 20 Hz) plus a live 3D model of the real arm
# pose next to them. A phase-driven bilingual mission panel narrates shell
# infiltration, fragment recovery, visual acquisition, and awakening; each
# joint is presented as a recoverable GHOST FRAGMENT. The GHOST VOICE_LINK
# panel subscribes to the same
# /ghost/tts/caption stream and shows the latest line with a
# cyber-pixel subtitle treatment. The operator console calls the existing
# start, abort, and return_home Trigger services and displays the ROS result
# in place. A deliberately low-visibility `mu` developer button in the
# console heading calls `mock_solve` during the searching phase - good for an
# audience-facing screen without exposing the demo shortcut prominently.
# Needs internet at
# the venue (three.js/urdf-loader load from a CDN); the progress bars
# and subtitle panel themselves have no such dependency and keep working
# offline, using their system-font fallbacks.
ros2 run ghost_game_orchestrator ghost_game_web_monitor

# Or the plain-text terminal table instead:
ros2 run ghost_game_orchestrator ghost_game_monitor
```
Both render `~/state`; the web monitor also consumes `/ghost/tts/caption` and
serves its latest caption at `/api/tts`. Neither ever displays `targets` even
if the state topic carries it (`debug_reveal_targets`) - the dashboard only
reads `progress`.

**Terminal 4 - control commands** (start a round, abort, or send the arm
home):
```bash
source /workspaces/zephyr-dev/zephyr_ws/install/setup.bash
ros2 service call /ghost_game_node/start std_srvs/srv/Trigger {}
ros2 service call /ghost_game_node/abort std_srvs/srv/Trigger {}
ros2 service call /ghost_game_node/return_home std_srvs/srv/Trigger {}
```
`return_home` also works mid-round (it asks the running round's background
thread to bail out and go home instead of racing a second thread against it).

For repeated commissioning runs, replace hand-guiding with the development
auto-solver:

```bash
ros2 run ghost_game_orchestrator ghost_game_mock_solve
```

The helper starts an idle round, waits until the controller enters
`searching`, then calls `~/mock_solve`. The orchestrator moves each unlocked
joint toward the current secret target with a soft, speed-bounded quintic
reference. Real `/joint_states`, `match_tolerance`, dwell time, stiffness
ramping, gripper opening, and the normal success-pose flow are still used.
It therefore tests the actual game transition instead of faking locked state.
Pass `--no-start` when the round has already been started separately.

## Services

| Service | Type | What it does |
| --- | --- | --- |
| `~/start` | `std_srvs/srv/Trigger` | Begin a new round (random or `fixed_targets`). Fails if a round/return_home is already running. |
| `~/mock_solve` | `std_srvs/srv/Trigger` | During `searching`, run the soft development trajectory toward the current secret pose. |
| `~/abort` | `std_srvs/srv/Trigger` | Stop the current round ASAP, relax to a safe stiffness. |
| `~/return_home` | `std_srvs/srv/Trigger` | Glide to the resting pose via the impedance JTC, then close the gripper and fully relax (`gravity_compensation.factor -> 0`). Also interrupts a running round. |

`~/state` (`std_msgs/msg/String`, JSON) publishes phase/locked-joint status at
5 Hz. With `publish_debug_distances: true` it also carries, per joint:
`distances` (rad from the secret target) and `progress` (0-100, the value the
dashboards render as a bar). `progress` is `100 * clamp01(1 - (distance -
match_tolerance) / (worst_case_separation - match_tolerance))`, pinned to 100
once a joint is `locked` - see `_publish_state` in `ghost_game_node.py`. The
raw `targets` array is only ever added when `debug_reveal_targets: true`
(both dashboards ignore it either way).

## Config

All game tuning lives in
`ghost_game_orchestrator/config/ghost_game.yaml` (parameter comments explain the
non-obvious ones - stiffness ramping, the JTC stale-setpoint seed trick, the
stuck/blocked back-off, etc). Before a real show, double check:

- `fixed_targets` - clear it back to the NaN-sentinel default (see the
  comment above it) so each round picks fresh random targets instead of the
  fixed test pose.
- `enable_dance` - set `true` so the finale trajectory actually plays.
- `publish_debug_distances` - keep this `true` if you want the web/terminal
  dashboard's progress bars during the show; it never leaks the secret angle.
- `debug_reveal_targets` - set `false` so the secret target angles never
  appear anywhere, including `~/state` (this one actually reveals the answer).
- `success_positions` / `success_time_from_start` - post-game pose and motion
  duration. The configured zero endpoint velocity and acceleration produce a
  smooth quintic stop; increase the duration to reduce peak speed further.
- `mock_solve_stiffness` / `mock_solve_max_velocity` - development-only
  auto-solver compliance and per-joint peak reference-speed bounds.
- `dance_positions` / `dance_time_from_start` - replace the placeholder
  choreography with a real recorded routine.
- `face_min_bbox_area_ratio` / `face_stable_time` - distance proxy and stable
  acquisition dwell before the arm follows a face.
- `face_yaw_direction` / `face_pitch_direction` - image-error-to-joint signs.
  The defaults match the commissioned success pose; reverse one sign if that
  axis moves the bbox farther from image center during hardware commissioning.
- `face_scan_offsets` / `face_*_max_offset` / `face_servo_*` - scan coverage,
  direct-command rate, speed limit, stale-frame freeze, and hard local motion
  bounds around the measured observation pose.
- `enable_face_reconstruction` / `face_reconstruction_*` - asynchronous FLUX
  submission, ROS topics, and the identity-preserving preprocessing prompt.
- `enable_mesh_reconstruction` / `mesh_reconstruction_config` - automatic
  Tripo submission and the Ghost-specific image/status/model URL topics.
- `tts_*_text` / `tts_joint_found_texts` - edit the stage script without
  changing Python. The joint-found list must contain exactly six lines.

Voice backend, DSP, queue limits, volume, speed, and the output device live in
`ghost_tts/config/ghost.yaml`. The unified launch can override the model,
preset, audio device, and backend from the command line. The Doubao API key,
voice ID, and resource ID are read from `DOUBAO_TTS_*` environment variables
in the ROS launch terminal and are not ROS parameters.

## The web dashboard's 3D robot model

`ghost_game_web_monitor` renders the *real* arm meshes, not a stand-in shape:

1. At startup it runs the actual `zephyr_arm.urdf.xacro` through `xacro`
   (same as `robot_state_publisher` would) and caches the expanded URDF -
   see `_expand_urdf` in `web_monitor.py`. Controlled by `enable_robot_model`
   / `robot_description_package` / `robot_description_xacro` params.
2. It serves that URDF at `/robot/robot.urdf`, and proxies every
   `package://<pkg>/<path>` mesh reference at `/robot/pkg/<pkg>/<path>` by
   resolving `<pkg>`'s installed share directory - no meshes are copied into
   this repo, they're streamed straight from `zephyr_arm_description` /
   `rebot_arm_rs_description` / `omni_picker_description`.
3. The browser (`web/viewer.js`) loads it with `urdf-loader` (three.js), and
   drives it with the live `positions` from `~/state` via
   `robot.setJointValue(name, angle)` at 20 Hz.
4. If any of that fails (xacro error, package missing, no internet for the
   three.js/urdf-loader CDN), it falls back to a lightweight kinematic
   skeleton built from the same joint origins/axes, so the panel still shows
   a geometrically correct pose either way.

This adds `xacro`, `zephyr_arm_description`, `rebot_arm_rs_description`, and
`omni_picker_description` as runtime dependencies of
`ghost_game_orchestrator` (see its `package.xml`) - all already present in
this workspace.
