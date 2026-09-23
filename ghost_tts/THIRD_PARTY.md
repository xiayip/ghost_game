# 第三方组件与模型

- 本项目节点、音效和测试代码采用 MIT 许可证，见 `LICENSE`。
- [Piper](https://github.com/OHF-Voice/piper1-gpl)：使用 `piper-tts==1.8.0`，上游 GPL-3.0；其依赖另有各自许可证。
- [Piper Python API](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md)。
- [sounddevice](https://python-sounddevice.readthedocs.io/)：PortAudio 音频输出接口。
- [libpulse](https://www.freedesktop.org/wiki/Software/PulseAudio/)：将容器音频发送到主机 PipeWire/PulseAudio 默认输出。
- [NumPy](https://numpy.org/)：音频数组、FIR 滤波和重采样。
- [Requests](https://requests.readthedocs.io/)：豆包 V3 HTTPS/SSE 客户端。
- [豆包语音 V3 HTTP Chunked/SSE API](https://docs.volcengine.com/docs/DoubaoVoice/HTTPChunkedSSEUnidirectionalStreaming-V3?lang=zh)：可选云端 TTS 后端；使用者需自行开通服务并遵守其条款与音色授权。
- 默认演示模型：[zh_CN-huayan-medium](https://huggingface.co/rhasspy/piper-voices/tree/main/zh/zh_CN/huayan/medium)，22,050Hz，单说话人中文模型。
- 该模型 [MODEL_CARD](https://huggingface.co/rhasspy/piper-voices/blob/main/zh/zh_CN/huayan/medium/MODEL_CARD) 将数据集 License 标为 **Unknown**；仓库级标签不能代替具体音色授权。包内不分发权重；正式产品使用前请确认该声音的授权，或替换为授权明确的 Piper 模型。

试听文件由上述模型在本机合成并经本项目音效处理，仅用于展示本次音色方案。
