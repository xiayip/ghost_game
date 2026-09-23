# RS 末端相机轻量跟随部署

日期：2026-09-23。基于 `xiayip/ghost_game` 提交 `961fab056aae235cb5d26f6a13bb6c2e2ed30b63`。

本版直接改造现有 ROS2 Jazzy / Zephyr RS 游戏第二阶段跟随，不引入 YOLO、PyTorch、Transformer 或新机械臂驱动。YuNet 保持 CPU 推理，JPEG 输入保持仓库已经使用的路径。新增依赖仅是 ROS 官方 `std_msgs`（用于诊断）。

**实现的是：通过末端相机图像误差，用 joint5 / joint4 对准并持续跟随人脸。不是六轴末端 XYZ 位置跟随。** 仓库当前深度未对齐，本版本不伪造三维坐标；基于图像误差的腕部跟随也不需要先把人脸换算到基座坐标。若以后做空间平移，再加入对齐深度、手眼标定和源时刻 TF。

代码、离线测试和打包已完成；此 Windows 环境没有 ROS2、Orbbec 或 RS 连接，尚未进行 colcon 构建、真实 DDS 或机械臂运动验收。没有把代码推送到远端仓库，也没有修改目标机器。

## 改了什么

| 原仓库 | 修改后 |
|---|---|
| 图像回调同步解码、推理、画图、JPEG 编码 | 回调只放最新消息；独立推理线程；预览另有独立线程 |
| 输入 DDS 保留多帧，结果队列 10 | 输入 depth=1；推理待处理槽 1；完成槽 1；结果收发 depth=1 |
| 15 Hz 推理上限 | 30 Hz 上限，保留 424 像素宽检测；忙时覆盖旧帧 |
| 控制样本以“刚收到消息”重新计时 | 保留 RGB 原始时间戳，推理前后、发布前和控制端都检查总年龄 |
| 整个扩展头部框用于瞄准 | `/nearest_face/detection` 保留截图框；新增 `/nearest_face/tracking` 发布未扩展的人脸框 |
| 最大人脸变化时直接改变输出对象 | 跟随话题保持几何目标 ID，关联不确定时不提供有效目标；截图话题仍选当前最大脸 |
| 速度随图像误差阶跃变化 | 新样本低通滤波、连续死区、参考速度与加速度限制 |
| 参考位置可持续领先实际关节 | 参考领先实际关节不超过设定值；关节反馈过期停止跟随阶段 |
| 居中后再跟随 3 秒即结束 | 保留原流程；可用 `continuous_face_follow:=true` 持续跟随到中止/回家 |

关键修复：原 `_face_detection_cb` 没有用 `msg.header.stamp` 判断源图像年龄，而是 `received_at=now`。所以即使控制端配置 `face_servo_detection_timeout=0.20`，排队了几百毫秒的图像刚到达时仍可能被当成“新鲜”。新版本将接收时源年龄加到后续控制年龄中，并拒绝旧序号/倒序消息。

本版本沿用仓库中已配置的关节方向、局部角度限制、峰值速度、阻抗参数与阶段切换。未提高腕部最大速度；也没有自动跳过游戏或执行观测姿态运动。

## 文件与接口

- `ghost_game_face_detection/.../runtime.py`：最新帧工作线程、短期目标关联、时间戳检查。
- `ghost_game_face_detection/.../node.py`：JPEG/原图输入、YuNet 推理、双话题输出、异步预览和诊断。
- `ghost_game_orchestrator/.../face_tracking.py`：源年龄、ID 检查、平滑与限幅算法。
- `ghost_game_orchestrator/.../ghost_game_node.py`：接入现有 MIT 阻抗指令链路，反馈过期检查及退出时速度前馈清零。
- `scripts/probe_face_latency.py`：只订阅的现场延迟采集工具。

| 话题 | 类型 | 用途 |
|---|---|---|
| `/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 保留现有输入 |
| `/nearest_face/detection` | `vision_msgs/Detection2DArray` | 最大人脸的扩展头部截图框；原图像素坐标 |
| `/nearest_face/tracking` | `vision_msgs/Detection2DArray` | 稳定几何目标的人脸框；原图像素坐标，带目标 ID |
| `/nearest_face/latency` | `std_msgs/String` JSON | 推理/处理时间、结果年龄、覆盖旧帧数量 |
| `/nearest_face/debug_image/compressed` | `sensor_msgs/CompressedImage` | 最长宽 640、最高 5 Hz 的 JPEG 预览 |
| `/ghost_game_node/state` | `std_msgs/String` JSON | 既有状态，新增 face 源年龄、目标 ID、控制时效数据 |
| `/zephyr_arm_impedance_controller/commands` | `trajectory_msgs/JointTrajectoryPoint` | 既有直接控制接口，保持 60 Hz、局部腕部跟随 |

目标 ID 只用于短期几何关联，不是身份识别；人脸重叠/交叉仍可能难以区分，此版本对明显歧义不输出跟随目标。框丢失时不拿缓存坐标继续积分，丢失超时后沿用原有重新扫描流程。

预览为缩略图，截图话题坐标仍属于原始 RGB。不能将原图 bbox 直接用于裁剪缩略图；3D/拍照功能应继续使用原 RGB。预览 5 FPS 不等于检测或控制只有 5 Hz。

## 部署到当前 RS 工作区

推荐先保存现场未提交修改，再将补丁应用到对应仓库。提供的 `ghost-game-low-latency.patch` 包含新增源码、测试和本说明，未包含机械臂驱动修改。若仓库 HEAD 与上述提交不同，先用 `git apply --check` 检查，不要强制覆盖冲突。

```bash
cd /workspaces/zephyr-dev/zephyr_ws/src/zephyr_mission/ghost_game
git status --short
git apply --check /path/to/ghost-game-low-latency.patch
git apply /path/to/ghost-game-low-latency.patch

source /opt/ros/jazzy/setup.bash
cd /workspaces/zephyr-dev/zephyr_ws
colcon build --symlink-install --packages-select \
  ghost_game_face_detection ghost_game_orchestrator ghost_game
source install/setup.bash
```

也提供完整 `ghost-game-low-latency.zip` 源码快照，适合独立测试工作区。不要在 ROS workspace 内同时保留两份包含同名包的源码。完整包保留了原仓库 ghost_tts，但轻量跟随启动时可禁用；不需要重新下载 TTS、图像生成或大语言模型。

目标机先运行回归测试：

```bash
cd /workspaces/zephyr-dev/zephyr_ws
colcon test --packages-select ghost_game_face_detection ghost_game_orchestrator
colcon test-result --verbose
```

这里包含仓库原有的 ROS 适配/TTS 测试，只有在实际 ROS 环境里才能完整验证；本机离线测试覆盖范围见后文。

## 启动和测试

机械臂 bringup 使用原来已经调通的命令：

```bash
source /opt/ros/jazzy/setup.bash
source /workspaces/zephyr-dev/zephyr_ws/install/setup.bash
ros2 launch zephyr_arm_bringup real_world.launch.py \
  activate_trajectory_controller:=false enable_camera:=auto
```

另一个终端启动轻量跟随配置：

```bash
source /workspaces/zephyr-dev/zephyr_ws/install/setup.bash
ros2 launch ghost_game ghost_game.launch.py \
  enable_tts:=false enable_web_monitor:=false continuous_face_follow:=true
```

`autostart` 仍为 false。使用原来的游戏启动方式（`/ghost_game_node/start`），在现有观测姿态到位后进入第二阶段。此参数仅延长跟随阶段，不会自动启动、不跳过姿态确认，也不会调用 mock_solve。

持续跟随模式不再因 3 秒结束；用现有 `/ghost_game_node/abort` 或 `/ghost_game_node/return_home` 退出。回家会执行仓库原有的回家流程。跟随退出会发出零速度前馈并保持最后受限位置参考；停止机械运动的真实响应仍取决于底层控制器，不等同于硬件急停。

若只检查感知、完全不启动游戏节点，可运行：

```bash
ros2 launch ghost_game_face_detection face_detection.launch.py \
  model_path:=$(ros2 pkg prefix ghost_game)/share/ghost_game/models
```

同一时间只运行一个检测节点。独立检测通过后，再使用统一 launch 启动控制链路。

## 实测延迟

在 ROS 工作区已 source 的终端，只读记录 60 秒：

```bash
cd /workspaces/zephyr-dev/zephyr_ws/src/zephyr_mission/ghost_game
python3 scripts/probe_face_latency.py --seconds 60 --out face-follow-latency.json
ros2 topic echo /nearest_face/latency --once
ros2 topic info /nearest_face/tracking -v
ros2 topic info /zephyr_arm_impedance_controller/commands -v
```

重点看 `valid_hz`、`valid_source_age_ms.p95`、无有效目标的间隔，以及 `face.latency.control_source_age_ms`。多机运行必须先同步时钟；输入图像/CameraInfo 的 frame_id 必须一致。机械臂收到 JointTrajectoryPoint 的订阅 QoS 也要检查：本补丁能修改发布侧为 depth=1，但底层控制器订阅实现不在该仓库内，尚未核验其队列/超时策略。

本版以 150 ms 作为过期丢弃门限，不代表端到端保证 150 ms。发布后的传输/执行器等待仍会增加延迟；控制端再次检查年龄。现场第一轮可争取感知 P95 <100 ms、稳定接近相机帧率，再用真实运动反馈确认跟随响应。不要只看“FPS 很高”。

若大量帧过期，先看输入源年龄、CPU/温度和任务负载，暂时将 `max_processing_rate` 调至 20，而不是扩大年龄门限让旧坐标通过。当前参数在节点初始化时读取，修改 YAML 后需重启；不要只调用 `ros2 param set` 以为运行逻辑已更新。

## 默认参数和调节顺序

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `detector_input_width` | 424 | 沿用该仓库已有尺寸，映射回原图；不是原版附件的全分辨率推理 |
| `max_processing_rate` | 30 Hz | 推理启动上限，不保证目标机一定达到 |
| `max_frame_age` | 0.15 s | 感知源时效 |
| `face_detection_max_age` / `face_servo_detection_timeout` | 0.15 s | 控制端总源时效 |
| `face_servo_rate_hz` | 60 Hz | 沿用现有控制参考频率 |
| `face_servo_max_joint_speed` | 0.25 rad/s | 沿用原峰值速度 |
| `face_servo_acceleration` | 1.0 rad/s² | 参考加速度限幅；硬限位/失效停止优先于平滑 |
| `face_filter_cutoff_hz` | 6 Hz | 仅新图像更新滤波，约 26.5 ms 时间常数 |
| `face_max_command_lead` | 0.06 rad | 不让位置参考持续跑到实际关节前方 |
| `face_joint_feedback_timeout` | 0.20 s | 每个关节反馈时效；有有效源时间戳时计入传输年龄，零时间戳只能用接收年龄 |
| `face_stable_time` | 0.20 s | 初次目标确认时间，跟随期间不反复等待 |
| `face_min_bbox_area_ratio` | 0.005 | 对未扩展人脸框，约对应旧 0.015 头部框；需要现场复核距离范围 |

先检查方向和年龄，再观察过冲/抖动。若动作偏软，先区分是控制器实际跟不上、参考领先限幅触发，还是滤波/加速度设置造成；不要同时提高增益、速度和加速度。既有腕部轴映射只在当前已调通的观测姿态附近成立，并非任意机械臂姿态下的通用视觉雅可比。

## 本机验证记录

38 项离线测试通过，覆盖原有选脸/框扩展/运动插值、新的最新帧覆盖、身份关联歧义、源时间戳/乱序、实际回调的年龄保存、关节反馈时效、低通只处理新样本、速度/加速度/参考领先限制，以及中止时清零速度前馈。所有 Python 语法和 package.xml 已检查。

实际 YuNet 模型测试使用 OpenCV 4.10 CPU、公开人脸图拼成 848×530 双人静态画面、检测输入 424×265。直接执行修改前后检测方法，模拟 ROS 消息和队列；每组预热模型后运行 6 秒，取后 5 秒。**没有实际 DDS、相机或机械臂；不代表 Spark 成绩。**

| 工况 | 原仓库有效输出 | 修改后有效输出 | 原结果年龄 P50/P95 | 修改后 P50/P95 |
|---|---:|---:|---:|---:|
| 正常负载，30 FPS 输入 | 14.8 Hz | 29.8 Hz | 21.7 / 60.1 ms | 13.2 / 53.0 ms |
| 每次检测人为额外等待 80 ms | 9.8 Hz | 10.2 Hz | 246.7 / 264.5 ms | 112.2 / 135.0 ms |

第二行是**合成压力测试**，用于验证积压时覆盖旧帧，并非目标机真实推理速度。正常负载仍存在 Windows 调度长尾，新增平滑参数也会改变运动响应，真实流畅性需现场调参验证。输入供给快于处理能力时，主动丢帧是预期行为；不能据此声称负载过高时仍保持 30 FPS。

详见 `validation/benchmark.json`。验证脚本和模型样例均为离线用途；现场只读探针见上方命令。

## 当前边界

- 未执行真机运动、未测机械臂物理延迟、未改底层电机驱动或阻抗控制器。
- 未实现真实三维位置跟随或生物身份识别。
- 原有 return_home / 扫描 / 游戏逻辑保留；新的连续跟随只在原第二阶段生效。
- 本机没有 ROS，`colcon build/test`、Jazzy 运行时、RS 指令消费频率和 D2C/TF 都需在目标机验证。
