# img2doc

`img2doc` 是 ROS 2 Jazzy Python 功能包。Ghost 游戏配置会自动接收 FLUX 生成的风格画像，将图像和 YAML 中的固定提示词发送给 DeepSeek 视觉模型，并把结构化访客赛博资料作为 `std_msgs/msg/String` JSON 发布到 Web 面板。

## 工作流

```text
/ghost/reconstruction/image (sensor_msgs/Image)
                │ FLUX 每轮发布一次
                ▼
       自动提交（也可调用 /ghost/profile/submit）
                │ ROS Image → 缩放 → JPEG → Base64
                ▼
 DeepSeek deepseek-flash + YAML 固定提示词
                │ JSON 解析、字段与义体等级校验
                ▼
 /ghost/profile/card (std_msgs/String，Transient Local)
```

Ghost 游戏默认 `auto_submit: true`。输入并非相机视频流，而是每轮唯一的 FLUX 结果，因此一张画像只会触发一次建档。独立运行时仍可关闭自动提交并使用 Trigger 服务。

## 接口

| 方向 | 名称 | 类型 | 说明 |
|---|---|---|---|
| 订阅 | `/ghost/reconstruction/image` | `sensor_msgs/msg/Image` | FLUX 风格画像，Sensor Data QoS |
| 服务 | `/ghost/profile/submit` | `std_srvs/srv/Trigger` | 对最新画像手动发起一次建档 |
| 发布 | `/ghost/profile/card` | `std_msgs/msg/String` | UTF-8 JSON，Reliable + Transient Local |

成功消息示例：

```json
{
  "request_id": "a1b2c3...",
  "status": "success",
  "character_name": "镜面幽灵",
  "codename": "007",
  "character_gender": "未知",
  "role": "网络侦察员",
  "cyberware_level": {"code": "C2", "name": "增强级"},
  "introduction": "游走于城市网络边缘的侦察员……"
}
```

失败消息仍发布到同一话题：

```json
{"request_id":"a1b2c3...","status":"error","error_message":"错误原因"}
```

## 名片字段

- `character_name`：中文角色代号或称号，不生成真实姓名形式
- `codename`：3 至 8 位纯数字编号字符串，可保留前导零
- `character_gender`：角色外观性别，`男`、`女` 或 `未知`
- `role`：角色定位
- `cyberware_level`：AI 根据可见义体判断的等级
- `introduction`：简短角色介绍

义体等级在提示词中固定为 C0 至 C5：原生级、辅助级、增强级、战术级、深度义体化、全身义体。节点会检查代码与中文名称是否匹配。

## API Key

推荐使用环境变量：

```bash
export DEEPSEEK_API_KEY='你的密钥'
```

也可以复制本地配置文件：

```bash
cd ~/hks_ws/src/img2doc
cp config/local_api.example.yaml config/local_api.yaml
chmod 600 config/local_api.yaml
```

`config/local_api.yaml` 已加入 `.gitignore`。密钥不会写入 ROS 日志或作为 ROS 参数公开。环境变量优先于本地文件。

## 构建

Ubuntu 24.04 和 ROS 2 Jazzy：

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hks_ws
colcon build --packages-select img2doc --symlink-install
source install/setup.bash
```

运行：

```bash
ros2 launch img2doc img2doc.launch.py
```

另开终端查看结果：

```bash
source /opt/ros/jazzy/setup.bash
source ~/hks_ws/install/setup.bash
ros2 topic echo /ghost/profile/card std_msgs/msg/String
```

发布测试图片。`image_publisher` 的实际输出话题名是 `image_raw`：

```bash
ros2 run image_publisher image_publisher_node \
  ~/hks_ws/src/img2mesh/img/input.jpg \
  --ros-args -r image_raw:=/ghost/reconstruction/image
```

节点收到首帧后提交一次：

```bash
ros2 service call /ghost/profile/submit std_srvs/srv/Trigger '{}'
```

## YAML 参数

配置文件为 `config/img2doc.yaml`。修改后重新构建或使用 `--symlink-install`，然后重启节点。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `model` | `deepseek-flash` | DeepSeek 视觉模型 |
| `image_detail` | `original` | `low/high/original/auto` |
| `max_long_edge` | `1300` | 发送前最长边缩放上限 |
| `jpeg_quality` | `90` | JPEG 质量 |
| `input_transient_local` | `true` | 重启后接收 FLUX 缓存的本轮画像 |
| `request_timeout_sec` | `120.0` | 单次 HTTP 超时 |
| `max_retries` | `2` | 空响应、网络错误、429 或 5xx 的重试上限 |
| `max_tokens` | `1200` | 文本输出上限；若被截断会自动加倍重试 |
| `temperature` | `0.2` | 降低格式与内容波动 |
| `auto_submit` | `true` | 收到每轮 FLUX 画像后自动建档 |
| `auto_interval_sec` | `1.0` | 自动模式的重复保护间隔 |
| `system_prompt` | 多行文本 | 固定系统提示词 |
| `card_prompt` | 多行文本 | 固定名片字段与判断规范 |

话题名和服务名也可在同一 YAML 中修改。提示词没有独立 ROS 话题，每次请求都使用 YAML 当前配置。

## 日志与计时

节点会打印：收到首帧、请求 ID、JPEG 编码耗时、API 发送时间、API 往返时间、模型和 token 用量、总耗时及发布结果。默认不逐帧打印，避免相机流刷屏；需要调试时设置 `log_each_image: true`。

## 方案选择

DeepSeek 官方接口与 OpenAI Chat Completions 兼容。`deepseek-flash` 支持图片与文本共同输入，图片可使用 Base64 data URL；JSON Output 通过 `response_format: {"type":"json_object"}` 请求。这里选择内联 JPEG，省去图片公网 URL 和对象存储依赖。

Gemini 和 OpenAI 也支持视觉输入和结构化输出，可作为后续供应商备选；当前功能包直接使用 DeepSeek，是因为接口满足视觉分析和 JSON 输出需求，并且与项目指定服务一致。本实现把 API 客户端、字段校验和 ROS 节点分开，后续替换供应商时可以保持 ROS 接口不变。

官方资料：

- [DeepSeek Vision](https://api-docs.deepseek.com/guides/vision/)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
- [Gemini Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output)

图片会发送到 DeepSeek 云端。部署时应根据产品隐私规则决定是否保存输入图像；本功能包不主动落盘保存相机帧。

## 测试

单元测试不会访问真实 API：

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hks_ws
colcon test --packages-select img2doc --event-handlers console_direct+
colcon test-result --verbose
```
