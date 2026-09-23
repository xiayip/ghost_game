# 手势标签与机械臂交互接口

手势识别已经合并进统一 ROS2 包 `ghost_game_perception`。该节点在 FACE 与 GESTURE 模式间切换，共用一个 RGB 订阅和一个最新帧 worker，不重复占用 Orbbec，也不修改机械臂控制循环。

## 已实现的标签

| 标签 | 含义与识别方式 | 默认交互请求 |
|---|---|---|
| `open_palm` | 张掌；稳定保持后，以当前位置为中点输出手掌平移量 | `palm_follow` |
| `handshake_offer` | 三指以上伸展、手掌纵轴朝向镜头的握手邀请候选 | `handshake` |
| `fist_bump_offer` | 四指弯曲、拳面朝向镜头的碰拳邀请候选 | `fist_bump` |
| `pointing` | 预训练指向类别或食指伸直、其余手指弯曲规则 | `point_at` |
| `wave` | 张掌轨迹在短窗口内左右多次变向 | `wave` |
| `thumb_up` | 竖拇指 | `acknowledge` |
| `closed_fist` | 普通握拳 | `fist` |
| `thumb_down`、`victory`、`ilove_you` | 其他静态类别 | 默认不触发动作 |

MediaPipe 原模型提供 21 个手部关键点及有限的静态类别，**不自带握手、碰拳、挥手分类器**。这里的握手、碰拳是意图候选；侧视、遮挡和手部旋转会影响规则，需在真机视角采集数据验证。它们不能证明双方已接触，也不能仅凭 RGB 得到手掌在机械臂基座下的位置。[官方能力与坐标定义](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python)

每条有效结果包含标签、置信度、来源 `classifier/heuristic`、手部跟踪 ID 和源时间戳。内置标签的 `score` 来自模型；几何候选使用未校准的固定分数 `0.70`，同时保留 `raw_label/raw_score` 供现场调参，不能将两类分数直接比较。`hand_id` 是短期几何关联编号，不是人物身份。当前不关联人脸与手所属人物；默认检查最多两只手，多手同时出现时只跟踪可见 bbox 面积最大的手。

## 像“张掌移动无人机”一样使用

稳定张掌约 0.20 秒后记录中点和初始手掌尺寸，移动同一只手得到 `control.dx/dy/dz`：右、下、靠近相机为正，范围 `[-1, 1]`，有死区。同一帧检测到多只手时，节点只把可见裁剪 bbox 面积最大的手送入状态机、深度采样和机械臂控制，检测消息与调试画面也只标出这只主手。掌心随动资格独立于 MediaPipe 的语义类别：模型输出 `Open_Palm` 时直接接受；模型因转动或前后运动偶发输出 `None`、`wave` 或 `handshake_offer` 时，只要 21 点几何仍显示至少三根非拇指手指伸展，`palm_open` 仍保持为真。这样不会因为一帧类别抖动重新等待 0.20 秒。语义手势事件仍沿用原来的严格标签门控。

兼容字段 `dz` 仍来自手掌表观尺寸变化；实际前后控制使用 `/camera/depth/image_raw/compressedDepth` 的彩色对齐深度。节点缓存最近 12 个压缩深度帧，为每个 RGB 推理结果选择时间戳最接近的 D2C 帧，解码对应 PNG 后在掌心附近圆形区域过滤零值、无穷值和范围外像素并取中值。压缩传输避免把约 0.9 MB 的原始深度帧复制进 Python；现场 profile 中对齐深度接收率约提高一倍。手掌中心相对画面中心的水平误差同时驱动 `joint5`，在局部 ±0.45 rad 范围内以最高 1.00 rad/s 转动相机；J5 在互动阶段使用 `kp=18`，六轴求解器将它作为显式受控关节，因此深度伺服不会抵消该偏航动作。`/gestures/palm_control` 会发布 `palm_open`、`palm_source`、`distance_m`、`distance_valid`、RGB/深度时间差、推理耗时及有效像素数。偶发的单帧丢手或深度失效会短暂保持最后一个有界位置目标，最长 0.20 秒；持续丢失、最大目标换手或超时后才冻结运动。

完整游戏在人脸捕获并提交重建、完成打量动作后进入 `gesture_interaction`。进入该阶段前会重新发布一次夹爪打开命令，并等待 `gripper_settle_time`，夹爪稳定张开后才启用手掌感知与随动；Palm Mock 也经过相同入口。第一帧稳定张掌的米制距离成为中点：手掌靠近相机时距离减小，机械臂沿初始相机光轴后退；手掌远离时机械臂向前探。控制器以 80 Hz 对六轴 URDF 几何雅可比做阻尼最小二乘求解，同时保持相机朝向；随动配置使用 1 cm 死区、±8 cm 行程、16 cm/s 直线速度和 0.65 rad/s 关节速度。短于 0.20 秒的丢手或深度失效保持最后位置目标，随后冻结；原始中点保留 1.00 秒，避免一帧分类或深度抖动反复把当前位置重置为零点。超过该窗口或换手才重新采集中点。机械臂受阻且跟踪误差超过 0.18 rad 时在实测位置重新建立命令，避免继续累积误差。

调试时 Web 引导会显示 `control_reason`。`baseline_captured` 表示刚建立中点，`inside_deadzone` 表示掌距变化小于 1 cm，`tracking` 表示已经输出运动，`arming` 表示张掌还未稳定满 0.20 秒，`rgb_depth_skew`/`stale_sample` 表示时间同步或输入时效问题。也可以直接检查：

```bash
curl -s http://127.0.0.1:8765/api/state | jq .palm_interaction
```

若只调试随手运动，不想重复寻找 Ghost 和人脸阶段，可点击 Web 操作台标题右下角低可见度的 `psi / MOCK_PALM` 按钮，或直接调用：

```bash
ros2 service call /ghost_game_node/mock_palm_interaction std_srvs/srv/Trigger {}
```

它会抢占当前 searching、人脸、轨迹或其他游戏阶段，等待原控制线程安全退出后，先恢复重力补偿，通过阻抗 JTC 平滑运动到并实测校验 `success_positions`。到位后才以 success pose 作为关节锚点和相机光轴基准，切换到直接阻抗控制并进入 `gesture_interaction`。此过程跳过人脸检测、打量动作和人脸重建。调试模式忽略上一轮遗留的 FLUX/Mesh 终态，持续到 `gesture_interaction_max_duration`；可随时使用现有 Abort 或 Return Home 按钮结束。

游戏节点每秒还会输出一条 `Palm servo accepted`，其中包含 `offset`、`joint_step` 和 `tracking_error`；若未接管则每两秒输出 `Palm servo waiting` 及拒绝原因。图像里仅出现 bbox 和距离说明检测与深度采样有效，不代表稳定张掌门控已经通过。

Orbbec 启动文件已开启 `depth_registration=true`、`align_target_stream=COLOR`，并关闭未使用的点云。该设置只在相机驱动重启后生效。互动至少持续 8 秒；Tripo 的 `/ghost/reconstruction/mesh_status` 成功或失败后结束，最长 90 秒。相关行程、速度、时长和深度参数集中在 `ghost_game_orchestrator/config/ghost_game.yaml` 与 `ghost_game_perception/config/ghost_game.yaml`。

`pointing` 额外给出二维指向单位向量，不能直接确定“指的是哪个物体”。若要指物，需要目标检测/分割、对齐深度和指向射线匹配；本次不加入这些重模型。

## 安装：Ubuntu 24.04 / ROS2 Jazzy

项目已合并到 `/workspaces/zephyr-dev/zephyr_ws/src/zephyr_mission/ghost_game`。
在 `zephyr_dev_24.04-aarch64` 容器内运行统一安装脚本即可安装 TTS 与手势依赖、下载并校验官方手势模型：

```bash
source /opt/ros/jazzy/setup.bash
bash /workspaces/zephyr-dev/zephyr_ws/src/zephyr_mission/ghost_game/scripts/install_hackathon_dependencies.sh
```

默认模型路径为 `~/.local/share/ghost_game/gesture_recognizer.task`。下载器使用官方版本 1 URL，校验文件长度与本项目固定的 SHA-256，并原子替换文件；原文件有效时不重复下载。这是已下载官方模型的完整性校验，不是上游签名认证。自训模型另存文件，通过 `model_path` 指定，不要用官方模型下载器校验自训文件。

依赖固定为 `mediapipe==1.0.1`。安装脚本使用它的 Python 3.12/aarch64 wheel，并保留容器当前已经被人脸识别与 FLUX 验证过的 NumPy 2/OpenCV 组合；不会让 pip 另装 `opencv-contrib-python` 覆盖现有视觉运行时。脚本最后会实际加载 `.task` 模型并跑一帧空图作为自检。本节点显式使用 CPU，**没有承诺 Spark 的 CUDA/TensorRT 加速**。

安装后构建并加载工作区：

```bash
cd /workspaces/zephyr-dev/zephyr_ws
colcon build --symlink-install --packages-select \
  ghost_game_perception ghost_game_orchestrator ghost_game
source install/setup.bash
```

## 启动、观察和接入动作

完整流程由 `ghost_game_perception` 统一订阅相机，并按阶段切换后端。单独测试手势时启动统一感知节点及预览路由器：

```bash
ros2 launch ghost_game_perception perception.launch.py \
  initial_mode:=GESTURE face_enabled:=true gesture_enabled:=true \
  enable_gesture_router:=true gesture_dry_run:=true
ros2 topic echo /gestures/state
ros2 topic echo /gestures/events
ros2 topic echo /gestures/palm_control
ros2 topic echo /gestures/detections
ros2 topic echo /interaction/requests
ros2 topic echo /interaction/status
```

除 `/gestures/detections` 使用 `vision_msgs/msg/Detection2DArray` 外，其余手势与交互 topic 都是 `std_msgs/msg/String`，`data` 为 JSON。分别使用不同终端运行 `echo`。输入默认 `/camera/color/image_raw/compressed`；统一节点的相机、阈值、bbox 和输出话题参数位于 `ghost_game_perception/config/ghost_game.yaml`。

总 launch 默认允许手势后端并启动 dry-run 路由器；在非手势阶段不会运行 MediaPipe：

```bash
ros2 launch ghost_game ghost_game.launch.py enable_gestures:=true gesture_dry_run:=true
```

| 输出 | 用途 |
|---|---|
| `/gestures/state` | 当前帧是否有效、原因、标签、归一化位置和性能信息 |
| `/gestures/events` | 稳定识别后仅发送一次的事件，带唯一 `event_id` |
| `/gestures/palm_control` | 张掌连续控制，含 `active`、`distance_m`、深度质量和时间戳 |
| `/gestures/detections` | 由 21 个关键点外接得到的手部 bbox 与原始 MediaPipe 类别 |
| `/interaction/requests` | 标签映射后的交互请求，默认 `dry_run: true`、`executed: false` |
| `/interaction/status` | 预览、拒绝、忙碌、执行完成或后端错误 |

默认阈值为 0.65，稳定保持 0.35 秒，释放 0.25 秒后才能再次触发，事件冷却 1.5 秒。持续保持同一个手势不会连续启动同一个动作。输入队列和推理槽只保留最新帧；手势推理上限 20 Hz，默认拒绝超过 0.20 秒的源图像。帧率、处理耗时、源年龄与稳定保持时间是不同指标，0.35 秒意图确认不等于相机延迟。

张掌推拉游戏已经由 `ghost_game_orchestrator` 直接执行；**握手、碰拳等其余动作仍没有真实执行服务。** 默认路由只发布可检查的请求，不发送这些动作的机械臂关节指令。接入已经实现的动作后，可将感知 launch 设置为
`enable_gesture_router:=false`，再运行独立路由器并配置白名单映射：

```bash
ros2 run ghost_game_perception gesture_action_router --ros-args \
  -p dry_run:=false \
  -p 'routes_json:={"handshake_offer":"handshake","fist_bump_offer":"fist_bump"}' \
  -p 'service_map_json:={"handshake":"/your_robot/handshake","fist_bump":"/your_robot/fist_bump"}'
```

`/your_robot/...` 是需自行替换的接口示例，不是此仓库已有服务。这里仅适配 `std_srvs/srv/Trigger`，后端必须在**动作完成**时响应；立即返回“已排队”的服务不能作为完成反馈。路由器同一时刻只允许一个在途服务；超时仍保持忙碌，因为 Trigger 无法取消正在执行的运动。若后端使用 ROS Action 或需要三维掌心目标，应由独立执行器消费事件，并负责与人脸跟随状态机互斥、取消及完成反馈；不要让多个控制器同时驱动机械臂。

## 电脑摄像头预览（不需 ROS）

安装上述 Python 依赖，在仓库根目录运行：

```bash
export PYTHONPATH="$PWD/ghost_game_perception${PYTHONPATH:+:$PYTHONPATH}"
python -m ghost_game_perception.download_model
python -m ghost_game_perception.preview --camera 0
# 可选镜像；方向定义跟随显示图像。
python -m ghost_game_perception.preview --camera 0 --mirror
# 离线回放，不打开摄像头；按视频原始帧率播放，推理忙时覆盖旧帧。
python -m ghost_game_perception.preview --input-video /path/to/clip.mp4 \
  --headless --json-every-frame --max-frames 100
```

Windows PowerShell 对应设置：`$env:PYTHONPATH = (Resolve-Path .\ghost_game_perception).Path`，其余 `python -m ...` 命令相同。窗口显示骨架、原始静态分类、最终标签、手 ID、控制提示、最近事件和处理时间；按 Q/Esc 或关闭窗口退出。默认不保存照片/视频，整个预览程序不建立机械臂连接。预览中的年龄从 OpenCV 读取完成时计时，不包含相机曝光、USB 和驱动缓存，不能当成真机端到端延迟。

## 如何可靠增加更多动作

1. **少量明确静态手型**：补充关键点几何规则并记录误触发，适合快速调试。规则输出保留 `source: heuristic`，不把分数当作经过校准的意图概率。
2. **大量或相似静态手型**：采集本项目视角的图像，训练关键点分类器或自定义 MediaPipe `.task`，通过 `model_path` 替换。新增类别还需加入 `gesture_core.py` 的标签归一化表及 `routes_json`，未知类别默认不触发。官方 Model Maker 可导出 `.task`，但官方已标注其不再积极维护，训练环境应独立固定，不能直接升级真机运行环境。[官方自定义模型指南](https://developers.google.com/edge/mediapipe/solutions/customization/gesture_recognizer)
3. **动作过程**：将 0.5–1.5 秒的归一化手/臂关键点、位移和速度序列送入小型 TCN/GRU；类别多、上下文复杂时再比较 Transformer。握手邀请、碰拳邀请与普通伸手的区别常依赖手臂方向和时序，只换单帧模型不一定解决。这是下一阶段方案，本次没有训练这些模型。

建议先约定标签边界：`handshake_offer` 是邀请，`handshake_contact` 是接触，`handshake_complete` 是交互完成；后三者需要不同证据，不能共用一个 RGB 静态标签。

建议的数据组织（本次不自动采集）：

```text
dataset/
  static/train/{none,handshake_offer,fist_bump_offer,pointing}/*.jpg
  static/val/{none,handshake_offer,fist_bump_offer,pointing}/*.jpg
  static/test/{none,handshake_offer,fist_bump_offer,pointing}/*.jpg
  sequences/<person_id>/<session_id>/<clip_id>.jsonl
  manifest.csv   # 文件、标签、人/场次、摄像头视角、时间范围、split
```

经参与者同意后，仅保存训练需要的手/上身画面或关键点，原图本地保存、明确保留期限；录制开关应由操作者显式开启。每类加入不同人、左右手、距离、照明与旋转，并大量采集无动作、挠头、拿手机、普通握拳等负例。按“人 + 录制场次”拆分训练/验证/测试，避免同一段视频相邻帧泄漏。验收除准确率外还看每分钟误启动次数、漏检率、标签确认延迟及开启手势前后人脸跟随的 p95 延迟；最终阈值应由现场数据确定。
