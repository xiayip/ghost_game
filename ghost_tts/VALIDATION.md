# Validation

Validated on 2026-09-22 in `zephyr_dev_24.04-aarch64:latest` with Ubuntu
24.04, ROS 2 Jazzy, Python 3.12.3, Piper 1.8.0, ONNX Runtime 1.30.0, NumPy
2.2.6, and sounddevice 0.5.6.

- `ghost_tts`, `ghost_game_orchestrator`, `ghost_game_perception`, and
  `ghost_game` build successfully with `colcon build --symlink-install`.
- The 20 model-independent TTS tests and four orchestrator/TTS integration
  tests pass. The Doubao tests cover V3 SSE request construction, PCM decode,
  cancellation, API errors, and credential redaction without consuming cloud
  quota. The wider workspace test-result summary has zero failures.
- The aarch64 Piper wheel loads `zh_CN-huayan-medium` and synthesized the
  smoke-test line to 3.55 seconds of mono 16-bit PCM at 22,050 Hz.
- The unified launch starts both `ghost_game_node` and `ghost_tts`, loads the
  game YAML, and resolves the default model under
  `~/.local/share/ghost_tts/zh_CN-huayan-medium.onnx`.
- The container's PortAudio default resolves to disconnected NVIDIA HDMI.
  The default `pulse` backend instead routes through the host PipeWire session
  to its current default sink.

Audible level and subjective voice quality still depend on the venue speaker.
Before the show, select that speaker as the host default, play one preview
line, and adjust `volume` and `effect_strength`. Use
`ghost_tts_preview --list-devices` only when selecting a direct PortAudio
device instead of the host Pulse route.

The Doubao backend was built and protocol-tested without a live request because
the `DOUBAO_TTS_*` variables were not present in the validation container.
After exporting them in the ROS launch terminal, run one short Action goal to
confirm that the cloned voice is active and entitled for `seed-icl-2.0`.
