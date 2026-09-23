# ghost_tts

`ghost_tts` is the offline Chinese voice output for Ghost Game. It uses a
Piper CPU model, applies a selectable electronic voice effect, and plays on
the host running the node. Text jobs are bounded, queued in order, and can be
cancelled without blocking the ROS executor.

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
ros2 topic pub --once /ghost/tts/text std_msgs/msg/String \
  "{data: '系统已苏醒。意识端口已经接入。'}"
ros2 service call /ghost/tts/stop std_srvs/srv/Trigger '{}'
```

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

The node publishes JSON job events on `/ghost/tts/status`, aggregate activity
on `/ghost/tts/busy`, and playback activity on `/ghost/tts/speaking`. See the
top-level Ghost Game README for game cue configuration and `THIRD_PARTY.md`
for dependency and voice-model licensing notes.
