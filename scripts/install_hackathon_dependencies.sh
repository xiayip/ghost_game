#!/usr/bin/env bash

# Restore the runtime dependencies used by the Ghost Game demo after the
# zephyr_dev container is recreated. Safe to run more than once.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REQUIREMENTS_FILE="${REPO_ROOT}/ghost_tts/requirements.txt"
VOICE_NAME="${GHOST_TTS_VOICE:-zh_CN-huayan-medium}"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "ERROR: requirements file not found: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

# Devcontainer terminals normally run as admin. docker exec without -u runs
# as root, so select admin automatically to keep pip packages and the model in
# the same home directory used by the normal ROS launch.
if [[ -n "${GHOST_TTS_USER:-}" ]]; then
  TARGET_USER="${GHOST_TTS_USER}"
elif [[ "${EUID}" -eq 0 ]] && id admin >/dev/null 2>&1; then
  TARGET_USER="admin"
else
  TARGET_USER="$(id -un)"
fi

TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
if [[ -z "${TARGET_HOME}" ]]; then
  echo "ERROR: unable to resolve home directory for ${TARGET_USER}" >&2
  exit 1
fi

run_as_target() {
  if [[ "$(id -un)" == "${TARGET_USER}" ]]; then
    "$@"
  elif [[ "${EUID}" -eq 0 ]]; then
    runuser -u "${TARGET_USER}" -- env HOME="${TARGET_HOME}" "$@"
  else
    sudo -n -u "${TARGET_USER}" env HOME="${TARGET_HOME}" "$@"
  fi
}

run_as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

missing_apt_packages=()
for package in \
  python3-pip libportaudio2 libpulse0 \
  python3-opencv python3-numpy python3-requests python3-yaml; do
  if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null \
      | grep -q 'ok installed'; then
    missing_apt_packages+=("${package}")
  fi
done

if (( ${#missing_apt_packages[@]} > 0 )); then
  echo "Installing apt packages: ${missing_apt_packages[*]}"
  run_as_root apt-get update
  run_as_root apt-get install -y --no-install-recommends \
    "${missing_apt_packages[@]}"
else
  echo "System audio/Python packages already installed."
fi

echo "Installing Ghost TTS Python packages for ${TARGET_USER}..."
run_as_target python3 -m pip install \
  --user \
  --break-system-packages \
  --disable-pip-version-check \
  --no-cache-dir \
  -r "${REQUIREMENTS_FILE}"

MODEL_DIR="${GHOST_TTS_MODEL_DIR:-${TARGET_HOME}/.local/share/ghost_tts}"
MODEL_PATH="${MODEL_DIR}/${VOICE_NAME}.onnx"
MODEL_CONFIG_PATH="${MODEL_PATH}.json"
run_as_target mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_PATH}" && -f "${MODEL_CONFIG_PATH}" ]]; then
  echo "Piper voice already present: ${MODEL_PATH}"
else
  echo "Downloading Piper voice ${VOICE_NAME}..."
  run_as_target python3 -m piper.download_voices \
    "${VOICE_NAME}" \
    --data-dir "${MODEL_DIR}"
fi

run_as_target python3 - "${MODEL_PATH}" <<'PY'
from pathlib import Path
import sys

import numpy
import onnxruntime
import piper
import sounddevice

model = Path(sys.argv[1])
config = Path(f'{model}.json')
if not model.is_file() or not config.is_file():
    raise SystemExit(f'voice download incomplete: {model}')

print(f'Python:       {sys.version.split()[0]}')
print(f'NumPy:        {numpy.__version__}')
print(f'ONNX Runtime: {onnxruntime.__version__}')
print(f'sounddevice:  {sounddevice.__version__}')
print(f'Voice model:  {model}')
PY

# The preferred route is the host Pulse/PipeWire Unix socket mounted by
# .devcontainer/devcontainer.json.  Containers created before that mount was
# added use the loopback TCP bridge in ghost_tts.engine instead.  Check both
# here so a dependency install cannot appear healthy while playback is mute.
if [[ -S /run/user/1000/pulse/native ]]; then
  echo "Audio route:  host Pulse socket /run/user/1000/pulse/native"
elif timeout 1 bash -c '</dev/tcp/127.0.0.1/4713' 2>/dev/null; then
  echo "Audio route:  host Pulse loopback bridge tcp:127.0.0.1:4713"
else
  cat >&2 <<'EOF'
WARNING: no host Pulse/PipeWire route is reachable from this container.
For an older host-network container, run this once in a HOST terminal:
  pactl load-module module-native-protocol-tcp listen=127.0.0.1 auth-ip-acl=127.0.0.1
The durable fix is to recreate the devcontainer so its Pulse socket mount and
PULSE_SERVER setting from .devcontainer/devcontainer.json take effect.
EOF
fi

cat <<EOF

Ghost Game TTS dependencies are ready for user ${TARGET_USER}.

Next steps:
  source /opt/ros/jazzy/setup.bash
  cd /workspaces/zephyr-dev/zephyr_ws
  colcon build --symlink-install --packages-select \\
    ghost_game_interfaces ghost_tts flux_image_editor ghost_game_orchestrator \
    ghost_game_face_detection ghost_game
  source install/setup.bash
  ros2 launch ghost_game ghost_game.launch.py
EOF
