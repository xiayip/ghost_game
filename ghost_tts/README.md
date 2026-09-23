# ghost_tts

`ghost_tts` is the switchable Chinese voice output for Ghost Game. The
`piper` backend runs fully offline on CPU; the `doubao` backend calls the
Volcengine Doubao Voice V3 API. Both backends use the same electronic effects,
audio output, bounded queue, and cancellable ROS 2 Action without blocking the
robot control executor.

The unified application launch starts this package automatically:

```bash
ros2 launch ghost_game ghost_game.launch.py
```

Install Piper and download the default voice once before the first launch:

```bash
bash ../scripts/install_hackathon_dependencies.sh
```

Or run the equivalent commands manually:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
mkdir -p ~/.local/share/ghost_tts
python3 -m piper.download_voices zh_CN-huayan-medium \
  --data-dir ~/.local/share/ghost_tts
```

The package can also run by itself:

```bash
ros2 launch ghost_tts ghost_tts.launch.py preset:=ghost audio_device:=pulse
ros2 action send_goal /ghost/tts/speak \
  ghost_game_interfaces/action/Speak \
  "{text: '系统已苏醒。意识端口已经接入。', interrupt: false}" --feedback

# Stops active playback, clears queued speech, then speaks immediately.
ros2 action send_goal /ghost/tts/speak \
  ghost_game_interfaces/action/Speak \
  "{text: '紧急链路接管。', interrupt: true}" --feedback

# Backward-compatible topic/service API:
ros2 topic pub --once /ghost/tts/text std_msgs/msg/String \
  "{data: '系统已苏醒。意识端口已经接入。'}"
ros2 service call /ghost/tts/stop std_srvs/srv/Trigger '{}'
```

To use the cloned Doubao voice, export its credentials in the same terminal
that starts ROS. The default `auto` backend then selects Doubao:

```bash
export DOUBAO_TTS_API_KEY='<API key from the Doubao Voice console>'
export DOUBAO_TTS_VOICE_TYPE='<S_... cloned voice ID>'
export DOUBAO_TTS_RESOURCE_ID='seed-icl-2.0'

ros2 launch ghost_tts ghost_tts.launch.py \
  backend:=auto preset:=ghost audio_device:=pulse
```

The API key is read directly from the process environment and is never stored
in ROS parameters, YAML, status topics, or logs. `DOUBAO_TTS_VOICE_TYPE` and
`DOUBAO_TTS_RESOURCE_ID` are also required when `backend:=doubao`. The default
request uses the ICL 2.0 standard model, mono 16-bit PCM at 24 kHz, and the
official V3 SSE endpoint. Tune `doubao_model`, `doubao_speech_rate`,
`doubao_loudness_rate`, and `doubao_timeout_seconds` in `config/ghost.yaml`.
Set `apply_effects: false` there to keep the cloud voice unprocessed while
still applying the configured output `volume`.

The unified launch uses `tts_backend:=auto` by default:

```bash
ros2 launch ghost_game ghost_game.launch.py tts_backend:=auto
```

`backend:=auto` falls back to Piper when any required variable is absent, so a
missed export cannot silence the show. Explicit `backend:=doubao` fails closed
with a clear initialization error instead of silently using another voice.

Available effect presets are `clean`, `subtle`, `ghost`, and `machine`.
`audio_device:=pulse` sends audio to the host PipeWire/PulseAudio default
speaker. The devcontainer mounts `/run/user/1000/pulse` and sets
`PULSE_SERVER` for this route. After changing `.devcontainer/devcontainer.json`,
recreate the container once so the mount is active.

An already-created host-network container cannot gain that bind mount through
`docker restart`. For that legacy container, restore the loopback bridge from
a **host** terminal without restarting ROS:

```bash
pactl load-module module-native-protocol-tcp \
  listen=127.0.0.1 auth-ip-acl=127.0.0.1
```

The listener is restricted to host loopback. Recreate the devcontainer when
convenient to use the Unix socket route and remove this compatibility step.

For a direct ALSA/PortAudio device instead, pass its name or index and list
available outputs with:

```bash
ros2 run ghost_tts ghost_tts_preview --list-devices
```

Play and save a standalone preview through the host default speaker with:

```bash
ros2 run ghost_tts ghost_tts_preview --play \
  --text '正在接入幽灵协议。'
```

The Action returns after playback and reports
`queued/synthesizing/processing/playing/done` feedback. Client cancellation
stops only its specific queued or active job. A goal with `interrupt=true`
preempts all older speech; preempted goals finish as aborted with message
`preempted`, because ROS 2 reserves the canceled terminal state for explicit
client cancellation.

The node publishes accepted text on `/ghost/tts/caption`, JSON job events on
`/ghost/tts/status`, aggregate activity on `/ghost/tts/busy`, and playback
activity on `/ghost/tts/speaking`. See the
top-level Ghost Game README for game cue configuration and `THIRD_PARTY.md`
for dependency and voice-model licensing notes.
