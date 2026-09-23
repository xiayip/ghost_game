# flux_image_editor（ROS 2 Jazzy / DGX Spark）

接收一帧 `sensor_msgs/msg/Image` 和一条文本提示词，调用 DGX Spark 上常驻的 FLUX.2 [klein] 4B 服务生成新图片，同时发布原始 ROS 图像和精确 PNG 字节。该功能包不调用 Tripo；其输出可作为 `img2mesh` 的输入，再执行 P1 三维生成。

模型服务和 ROS 节点分开运行：模型位于 GPU Docker 容器中，ROS 节点通过本机 HTTP 调用它。这样可以保持模型常驻，也避免 PyTorch/CUDA 依赖影响 ROS 2 的 Python 环境。

## 数据流

```text
/sensor_msgs/image_raw (sensor_msgs/msg/Image) ─┐
                                                ├→ 缓存图像和提示词
/flux_image_editor/prompt (std_msgs/msg/String) ─┘
    → 每条非空提示词最多自动提交一次
    → ROS Image 编码为 PNG
    → POST http://127.0.0.1:8090/v1/edit
    → FLUX.2 [klein] 4B：PNG + prompt → 新 PNG
    ├→ /flux_image_editor/image_raw (sensor_msgs/msg/Image, bgr8)
    ├→ /flux_image_editor/image_png (sensor_msgs/msg/CompressedImage, png)
    ├→ /flux_image_editor/status (std_msgs/msg/String, JSON)
    └→ ~/flux_outputs/<request_id>.png
```

连续相机帧只更新缓存，不会反复推理。提示词先到时，节点会等待下一帧图像；图像先到时，下一条非空提示词触发任务。任务运行期间收到多条提示词时只保留最新一条，并在当前任务结束后提交。也可关闭自动提交，通过 `/flux_image_editor/submit` 手动提交当前图像和提示词。

## 话题与服务

| 名称 | 类型 | 作用 |
| --- | --- | --- |
| `/sensor_msgs/image_raw` | `sensor_msgs/msg/Image` | 输入图像，sensor-data QoS |
| `/flux_image_editor/prompt` | `std_msgs/msg/String` | 非空编辑指令；默认触发一次推理 |
| `/flux_image_editor/image_raw` | `sensor_msgs/msg/Image` | 新图，`bgr8`，可直接接入后续 ROS 节点 |
| `/flux_image_editor/image_png` | `sensor_msgs/msg/CompressedImage` | 新图的原始 PNG 字节，`format=png` |
| `/flux_image_editor/status` | `std_msgs/msg/String` | JSON 状态和计时，transient-local |
| `/flux_image_editor/submit` | `std_srvs/srv/Trigger` | 手动提交最近图像和提示词 |

状态依次为 `accepted`、`encoding`、`running`、`success`；失败时为 `error`。成功 JSON 示例：

```json
{
  "request_id": "7cd6...",
  "status": "success",
  "prompt": "将人物转换为赛博朋克风格",
  "output_path": "/home/user/flux_outputs/7cd6....png",
  "width": 784,
  "height": 1024,
  "input_png_bytes": 1854201,
  "output_png_bytes": 2143058,
  "image_received_at": "2026-09-22T23:10:01.120Z",
  "prompt_received_at": "2026-09-22T23:10:02.400Z",
  "pair_ready_at": "2026-09-22T23:10:02.400Z",
  "accepted_at": "2026-09-22T23:10:02.401Z",
  "encode_started_at": "2026-09-22T23:10:02.402Z",
  "request_sent_at": "2026-09-22T23:10:02.438Z",
  "png_received_at": "2026-09-22T23:10:07.302Z",
  "result_ready_at": "2026-09-22T23:10:07.350Z",
  "published_at": "2026-09-22T23:10:07.371Z",
  "source_stamp": "1790133001.100000000",
  "server_request_id": "7cd6...",
  "model_load_sec": 18.21,
  "queue_wait_sec": 0.001,
  "encode_sec": 0.03,
  "http_roundtrip_sec": 4.864,
  "server_total_sec": 4.801,
  "inference_sec": 4.72,
  "transport_overhead_sec": 0.063,
  "push_to_png_received_sec": 4.902,
  "postprocess_sec": 0.05,
  "push_to_result_ready_sec": 4.950,
  "publish_sec": 0.001,
  "push_to_publish_sec": 4.971,
  "total_sec": 4.971,
  "seed": 42,
  "error": ""
}
```

计时以图像与提示词均已到达节点的 `pair_ready_at` 为起点。`push_to_png_received_sec` 是用户最关心的“图片＋提示词就绪到收到新 PNG”时间；`push_to_publish_sec` 还包含解码、落盘、排队和 ROS 发布。`inference_sec` 由 DGX 推理服务测量，`http_roundtrip_sec` 由 ROS 客户端测量，`transport_overhead_sec` 是两者在当前实现中的差值。所有墙上时间使用 UTC ISO 8601，持续时间使用单调时钟，避免系统校时导致负值。

终端会以同一个 `request_id` 打印以下关键节点：图片和提示词配对、输入 PNG 编码完成、请求发出、收到新 PNG、解码与保存完成、ROS 发布完成。DGX 容器日志也会打印请求接收和生成完成，并带相同的 `request_id`，便于跨进程核对。

## DGX Spark 部署

DGX Spark 是 ARM64 Ubuntu 24.04，建议使用仓库提供的 NVIDIA NGC PyTorch 容器。`Dockerfile.dgx_spark` 基于 NVIDIA 官方 DGX Spark ComfyUI 示例使用的 `nvcr.io/nvidia/pytorch:26.02-py3`，并固定了 Diffusers 提交版本，避免构建结果随上游变化。容器不会安装 ROS。

### 1. 检查 DGX Spark

通过 SSH 登录后执行：

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.02-py3 \
  python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())'
```

若当前用户没有 Docker 权限：

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

### 2. 克隆并构建 ROS 工作空间

```bash
git clone <你的仓库地址> ~/hks_ws
source /opt/ros/jazzy/setup.bash
cd ~/hks_ws
colcon build --packages-select flux_image_editor --symlink-install
source install/setup.bash
```

需要 ROS 依赖：`rclpy`、`sensor_msgs`、`std_msgs`、`std_srvs`、`ament_index_python`；系统 Python 还需 `python3-opencv`、`python3-numpy`、`python3-requests` 和 `python3-yaml`。ROS 图像转换不依赖 `cv_bridge`，因此兼容当前容器的 NumPy 2 环境。

## Ghost Game 集成

统一启动使用 `config/ghost_game.yaml`，数据流为：

```text
/nearest_face/head_crop
  + /ghost/reconstruction/prompt
  -> FLUX HTTP service
  -> /ghost/reconstruction/image
  -> /ghost/reconstruction/image_png   (exact img2mesh input)
  -> /ghost/reconstruction/status
```

编排节点只在人脸稳定并居中后发布一次提示词。FLUX 请求在工作线程中
运行，服务未启动或推理失败只会发布 `status=error`，不会阻塞机械臂动作。
本功能包完成图像预处理，不直接生成网格；图生 3D 节点订阅
`/ghost/reconstruction/image_png`，直接上传 FLUX 返回的 PNG 字节。

原始完整头部裁剪会先保存到
`/workspaces/zephyr-dev/zephyr_ws/outputs/ghost_face_captures/<request_id>_head_crop.png`，
并同步更新固定地址
`/workspaces/zephyr-dev/zephyr_ws/outputs/ghost_face_captures/latest_head_crop.png`。
两个绝对路径也会写入状态
JSON 的 `input_path` 和 `latest_input_path`，保存过程不依赖推理服务。

### 3. 启动常驻模型服务

```bash
cd ~/hks_ws/src/flux_image_editor
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml up -d
docker compose -f docker/compose.yaml logs -f flux-klein
```

第一次运行需要下载模型。完整 Hugging Face 仓库约 23.7 GB，应预留额外空间供容器层和缓存使用。默认缓存目录为功能包根目录下的 `.model_cache/`，已加入 `.gitignore`。可在启动前改到独立磁盘：

```bash
export FLUX_MODEL_CACHE=/data/models/huggingface
docker compose -f docker/compose.yaml up -d --build
```

日志出现以下内容后表示模型可用：

```text
Model ready in ... seconds
```

检查接口：

```bash
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/ready
```

端口只绑定到 DGX Spark 的 `127.0.0.1`，不会直接暴露到局域网。如果 ROS 节点在另一台机器，可建立 SSH 隧道：

```bash
ssh -N -L 8090:127.0.0.1:8090 <用户>@<DGX-SPARK-IP>
```

此时远端 ROS 节点仍使用 `http://127.0.0.1:8090`。

### 4. 先进行文件测试

服务就绪后，不启动 ROS 也能验证 PNG 加提示词：

```bash
source /opt/ros/jazzy/setup.bash
source ~/hks_ws/install/setup.bash
ros2 run flux_image_editor flux_test_file \
  /absolute/path/input.png \
  '将人物转换为赛博朋克风格，保留人物身份、姿势和干净背景' \
  --output ~/flux_outputs/file_test.png
```

终端输出服务端推理时间、客户端总时间、尺寸和保存路径。

### 5. 启动 ROS 节点

```bash
source /opt/ros/jazzy/setup.bash
source ~/hks_ws/install/setup.bash
ros2 launch flux_image_editor flux_image_editor.launch.py
```

另开终端发布输入。下面复用已有测试照片；`image_publisher` 会将文件解码成 `sensor_msgs/msg/Image`：

```bash
ros2 run image_publisher image_publisher_node \
  /home/ddw/hks_ws/src/img2mesh/img/input.jpg \
  --ros-args -r image:=/sensor_msgs/image_raw
```

发布一次提示词并观察状态：

```bash
ros2 topic pub --once /flux_image_editor/prompt std_msgs/msg/String \
  "{data: '将人物转换为赛博朋克风格，保留人物身份、脸部比例、姿势和白色背景'}"

ros2 topic echo /flux_image_editor/status
```

收到 `status=success` 后，PNG 位于 `output_path`；`/flux_image_editor/image_raw` 和 `/flux_image_editor/image_png` 也各保存最近一次结果。

## YAML 配置

### ROS 节点

编辑 `config/flux_image_editor.yaml`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `image_topic` | `/sensor_msgs/image_raw` | 输入图像话题，启动时生效 |
| `prompt_topic` | `/flux_image_editor/prompt` | 提示词话题，启动时生效 |
| `output_image_topic` | `/flux_image_editor/image_raw` | 输出原始图像话题 |
| `output_png_topic` | `/flux_image_editor/image_png` | 输出 PNG 压缩消息话题 |
| `status_topic` | `/flux_image_editor/status` | JSON 状态话题 |
| `submit_service` | `/flux_image_editor/submit` | 手动提交服务 |
| `server_url` | `http://127.0.0.1:8090` | 推理服务地址 |
| `request_timeout_sec` | `300.0` | 单次编辑请求超时 |
| `auto_submit_on_prompt` | `true` | 每条非空提示词自动提交一次 |
| `input_directory` | `~/flux_inputs` | 原始输入裁剪的保存目录 |
| `save_input` | `true` | HTTP 推理前保存输入 PNG 与 `latest_head_crop.png` |
| `output_directory` | `~/flux_outputs` | PNG 保存目录 |
| `save_output` | `true` | 是否落盘保存 PNG |
| `max_input_bytes` | `20000000` | 输入 PNG 最大字节数 |
| `log_timing` | `true` | 保留的计时日志开关 |

话题名与服务名只能在启动时设置。其余参数可在调试时使用 `ros2 param set` 修改，修改只影响后续任务。

### 模型服务

编辑 `config/server.yaml` 后重启容器：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model_id` | `black-forest-labs/FLUX.2-klein-4B` | Hugging Face 模型 ID |
| `device` | `cuda` | 推理设备 |
| `dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `inference_steps` | `4` | 蒸馏模型默认快速步数 |
| `guidance_scale` | `1.0` | 文本引导强度 |
| `seed` | `-1` | `-1` 每次随机；非负值用于复现 |
| `max_long_edge` | `1024` | 输入和输出的最大长边 |
| `max_megapixels` | `1.1` | 最大输出像素量 |
| `dimension_multiple` | `16` | 宽高向下对齐倍数 |
| `enable_cpu_offload` | `false` | 统一内存足够时保持关闭以降低延迟 |

重启命令：

```bash
docker compose -f docker/compose.yaml restart flux-klein
docker compose -f docker/compose.yaml logs -f flux-klein
```

改变 `model_id` 或镜像依赖时需要重新创建容器；只改推理参数时执行上面的重启即可。

## 接入 img2mesh

先单独验证 FLUX 输出的人物一致性和风格效果。验证完成后，把 `img2mesh/config/img2mesh.yaml` 中的输入改为：

```yaml
image_topic: /flux_image_editor/image_png
input_compressed: true
```

使用普通的 `img2mesh.launch.py`，不要使用会再次调用 Tripo 图像编辑 API 的 `img2mesh_style.launch.py`。目前两个包之间没有自动调用三维提交服务；可先在收到 FLUX `success` 后手动调用：

```bash
ros2 service call /img2mesh/submit std_srvs/srv/Trigger '{}'
```

后续可以增加一个编排节点，在 FLUX 成功消息后自动调用该服务。

## 验证范围与限制

- 本机测试使用模拟推理服务，不下载模型，也不执行 GPU 推理。
- DGX Spark 上的首次模型加载时间、热启动推理时间和峰值统一内存必须实测记录。
- FLUX 输出会改变原图像素；即使提示词要求保留身份，也不能保证面部完全不变。
- 图像中新增护目镜等内容不保证被 Tripo 重建为独立三维几何。
- FLUX.2 [klein] 4B 模型权重采用 Apache 2.0 许可；部署产品时仍需遵守模型卡的使用限制。

参考资料：

- [FLUX.2 Klein 4B 模型卡](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [FLUX.2 官方推理仓库](https://github.com/black-forest-labs/flux2)
- [NVIDIA DGX Spark 图像生成指南](https://build.nvidia.com/spark/comfyui/video-gen-workflow)
- [DGX Spark ARM64 与统一内存说明](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)
