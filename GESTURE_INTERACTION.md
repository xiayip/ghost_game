# 手势标签与机械臂交互接口

在 `xiayip/ghost_game` 的最新基线上新增独立 ROS2 包 `ghost_game_gestures`。原有人脸跟随可继续运行；手势节点订阅同一个 RGB topic，不重新占用 Orbbec，不修改人脸推理或机械臂控制循环。

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

每条有效结果包含标签、置信度、来源 `classifier/heuristic`、手部跟踪 ID 和源时间戳。内置标签的 `score` 来自模型；几何候选使用未校准的固定分数 `0.70`，同时保留 `raw_label/raw_score` 供现场调参，不能将两类分数直接比较。`hand_id` 是短期几何关联编号，不是人物身份。当前不关联人脸与手所属人物；默认检查最多两只手，多手同时出现时停止触发，先采用单人、单手交互。

## 像“张掌移动无人机”一样使用

稳定张掌约 0.35 秒后记录中点和初始手掌尺寸，移动同一只手得到 `control.dx/dy/dz`：右、下、靠近相机为正，范围 `[-1, 1]`，有死区。`dz` 来自手掌表观尺寸相对初始值的变化，手掌转动也会影响它，因此不是实际深度。松手、低置信度、多人/多手歧义、过期帧都会退出控制并将三个分量归零。挥手判定成立时进入 `wave`，不同时保留张掌控制。

这是**无量纲的位移提示**，还不是机械臂速度或三维目标。要得到可靠的前后米制距离，应将掌心与 Orbbec 对齐深度配准；当前近似值只适合低速交互原型。`palm_follow` 是一次启动请求，连续位移从 `/gestures/palm_control` 读取。消费端应将有效新消息映射到经过标定的控制轴，并自行检查消息时效、限速和互斥；停止收到消息也要主动停。相机固定在末端时，相机自身转动也会改变图像中的手掌位置，需要结合末端姿态及手眼标定后再做空间平移。

`pointing` 额外给出二维指向单位向量，不能直接确定“指的是哪个物体”。若要指物，需要目标检测/分割、对齐深度和指向射线匹配；本次不加入这些重模型。

## 安装：Ubuntu 24.04 / ROS2 Jazzy

下面假定仓库在 `~/zephyr_ws/src/ghost_game`；按实际工作区修改路径。

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install python3-venv python3-colcon-common-extensions
python3 -m venv --system-site-packages ~/.venvs/ghost-gestures
source ~/.venvs/ghost-gestures/bin/activate
python -m pip install -r ~/zephyr_ws/src/ghost_game/ghost_game_gestures/requirements.txt

cd ~/zephyr_ws
# 显式用这个 venv 的 Python 构建，保证节点能找到 MediaPipe。
python /usr/bin/colcon build --symlink-install --packages-select ghost_game_gestures
source install/setup.bash
ros2 run ghost_game_gestures download_gesture_model
```

默认模型路径为 `~/.local/share/ghost_game/gesture_recognizer.task`。下载器使用官方版本 1 URL，校验文件长度与本项目固定的 SHA-256，并原子替换文件；原文件有效时不重复下载。这是已下载官方模型的完整性校验，不是上游签名认证。自训模型另存文件，通过 `model_path` 指定，不要用官方模型下载器校验自训文件。

依赖固定 `mediapipe==0.10.18`、OpenCV 4.10、NumPy 1.x，避免 NumPy 2.x 与系统 ROS 二进制扩展不匹配。该 MediaPipe 版本有 Python 3.12 的 Linux x86_64 和 aarch64 wheel；DGX Spark 仍须在目标系统确认全部依赖可以安装、模型能实际推理。本节点显式使用 CPU，**没有承诺 Spark 的 CUDA/TensorRT 加速**。[官方 PyPI 文件列表](https://pypi.org/project/mediapipe/0.10.18/#files)

每个新终端先 `source /opt/ros/jazzy/setup.bash`、激活上述 venv，再 `source ~/zephyr_ws/install/setup.bash`。

## 启动、观察和接入动作

先沿用现有命令启动相机与人脸跟随，再独立启动手势节点及预览路由器：

```bash
ros2 launch ghost_game_gestures gestures.launch.py enable_action_router:=true dry_run:=true
ros2 topic echo /gestures/state
ros2 topic echo /gestures/events
ros2 topic echo /gestures/palm_control
ros2 topic echo /interaction/requests
ros2 topic echo /interaction/status
```

所有上述 topic 均为 `std_msgs/msg/String`，`data` 为 JSON。分别使用不同终端运行 `echo`。输入默认 `/camera/color/image_raw/compressed`，也可在包的 `config/gestures.yaml` 副本中设置 `image_transport: raw` 及原图话题，再传 `config:=/absolute/path/gestures.yaml`。

也可以从总 launch 按需启用；默认仍关闭手势功能，不改变原有人脸跟随启动行为：

```bash
ros2 launch ghost_game ghost_game.launch.py enable_gestures:=true gesture_dry_run:=true
```

| 输出 | 用途 |
|---|---|
| `/gestures/state` | 当前帧是否有效、原因、标签、归一化位置和性能信息 |
| `/gestures/events` | 稳定识别后仅发送一次的事件，带唯一 `event_id` |
| `/gestures/palm_control` | 张掌的连续控制提示，含 `active` 与时间戳 |
| `/interaction/requests` | 标签映射后的交互请求，默认 `dry_run: true`、`executed: false` |
| `/interaction/status` | 预览、拒绝、忙碌、执行完成或后端错误 |

默认阈值为 0.65，稳定保持 0.35 秒，释放 0.25 秒后才能再次触发，事件冷却 1.5 秒。持续保持同一个手势不会连续启动同一个动作。输入队列和推理槽只保留最新帧；手势推理上限 20 Hz，默认拒绝超过 0.20 秒的源图像。帧率、处理耗时、源年龄与稳定保持时间是不同指标，0.35 秒意图确认不等于相机延迟。

**当前仓库没有握手/碰拳等动作的真实执行服务。** 默认路由只发布可检查的请求，不发送机械臂关节指令。接入已经实现的动作后，可运行独立路由器并配置白名单映射；不要同时启动 launch 中的预览路由器：

```bash
ros2 run ghost_game_gestures gesture_action_router --ros-args \
  -p dry_run:=false \
  -p 'routes_json:={"handshake_offer":"handshake","fist_bump_offer":"fist_bump"}' \
  -p 'service_map_json:={"handshake":"/your_robot/handshake","fist_bump":"/your_robot/fist_bump"}'
```

`/your_robot/...` 是需自行替换的接口示例，不是此仓库已有服务。这里仅适配 `std_srvs/srv/Trigger`，后端必须在**动作完成**时响应；立即返回“已排队”的服务不能作为完成反馈。路由器同一时刻只允许一个在途服务；超时仍保持忙碌，因为 Trigger 无法取消正在执行的运动。若后端使用 ROS Action 或需要三维掌心目标，应由独立执行器消费事件，并负责与人脸跟随状态机互斥、取消及完成反馈；不要让多个控制器同时驱动机械臂。

## 电脑摄像头预览（不需 ROS）

安装上述 Python 依赖，在仓库根目录运行：

```bash
export PYTHONPATH="$PWD/ghost_game_gestures${PYTHONPATH:+:$PYTHONPATH}"
python -m ghost_game_gestures.download_model
python -m ghost_game_gestures.preview --camera 0
# 可选镜像；方向定义跟随显示图像。
python -m ghost_game_gestures.preview --camera 0 --mirror
# 离线回放，不打开摄像头；按视频原始帧率播放，推理忙时覆盖旧帧。
python -m ghost_game_gestures.preview --input-video /path/to/clip.mp4 \
  --headless --json-every-frame --max-frames 100
```

Windows PowerShell 对应设置：`$env:PYTHONPATH = (Resolve-Path .\ghost_game_gestures).Path`，其余 `python -m ...` 命令相同。窗口显示骨架、原始静态分类、最终标签、手 ID、控制提示、最近事件和处理时间；按 Q/Esc 或关闭窗口退出。默认不保存照片/视频，整个预览程序不建立机械臂连接。预览中的年龄从 OpenCV 读取完成时计时，不包含相机曝光、USB 和驱动缓存，不能当成真机端到端延迟。

## 如何可靠增加更多动作

1. **少量明确静态手型**：补充关键点几何规则并记录误触发，适合快速调试。规则输出保留 `source: heuristic`，不把分数当作经过校准的意图概率。
2. **大量或相似静态手型**：采集本项目视角的图像，训练关键点分类器或自定义 MediaPipe `.task`，通过 `model_path` 替换。新增类别还需加入 `core.py` 的标签归一化表及 `routes_json`，未知类别默认不触发。官方 Model Maker 可导出 `.task`，但官方已标注其不再积极维护，训练环境应独立固定，不能直接升级真机运行环境。[官方自定义模型指南](https://developers.google.com/edge/mediapipe/solutions/customization/gesture_recognizer)
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
