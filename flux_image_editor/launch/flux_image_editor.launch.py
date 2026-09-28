"""Start or reuse the FLUX service, wait for readiness, then start the bridge."""

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from flux_image_editor.config import config_dir, load_ros_parameters


def _as_bool(value):
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _legacy_package_root(package_root):
    try:
        return package_root.parents[2] / "tmp" / "flux_image_editor"
    except IndexError:
        return package_root / ".legacy-not-found"


def _default_inference_python(package_root):
    override = os.environ.get("FLUX_EDITOR_PYTHON", "").strip()
    if override:
        return str(Path(override).expanduser())
    candidates = [
        package_root / ".venv" / "bin" / "python",
        _legacy_package_root(package_root) / ".venv" / "bin" / "python",
    ]
    return str(next(
        (path for path in candidates if path.is_file()), Path(sys.executable)
    ))


def _default_model_cache(package_root):
    override = os.environ.get("HF_HOME", "").strip()
    if override:
        return str(Path(override).expanduser())
    candidates = [
        package_root / ".model_cache",
        _legacy_package_root(package_root) / ".model_cache",
    ]
    for path in candidates:
        if (path / "hub" / "models--black-forest-labs--FLUX.2-klein-4B").is_dir():
            return str(path)
    return str(candidates[0])


def _server_is_ready(url):
    try:
        with urlopen(url.rstrip("/") + "/ready", timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return response.status == 200 and payload.get("ready") is True
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def _is_local_server_url(url):
    return urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}


def _launch_setup(context):
    configuration_dir = config_dir().resolve()
    package_root = configuration_dir.parent
    config_file = LaunchConfiguration("config_file").perform(context)
    server_url = LaunchConfiguration("server_url").perform(context).rstrip("/")
    start_server = _as_bool(LaunchConfiguration("start_server").perform(context))
    wait_for_server = _as_bool(
        LaunchConfiguration("wait_for_server").perform(context)
    )
    ready_timeout = LaunchConfiguration("ready_timeout_sec").perform(context)
    inference_python = LaunchConfiguration("inference_python").perform(context)
    server_config = LaunchConfiguration("server_config").perform(context)
    model_cache = LaunchConfiguration("model_cache").perform(context)

    node = Node(
        package="flux_image_editor",
        executable="flux_image_editor_node",
        name="flux_image_editor",
        parameters=[
            config_file,
            {"server_url": ParameterValue(server_url, value_type=str)},
        ],
        output="screen",
    )
    actions = []
    server_ready = _server_is_ready(server_url)
    if start_server and _is_local_server_url(server_url) and not server_ready:
        python_path = os.environ.get("PYTHONPATH", "")
        server_process = ExecuteProcess(
            cmd=[inference_python, "-m", "flux_image_editor.inference_server"],
            # onnxruntime telemetry may create a ':memory:.ses' file in cwd.
            cwd="/tmp",
            additional_env={
                "FLUX_EDITOR_SERVER_CONFIG": server_config,
                "HF_HOME": model_cache,
                "PYTHONPATH": str(package_root) + (
                    os.pathsep + python_path if python_path else ""
                ),
                "PYTHONUNBUFFERED": "1",
            },
            output="screen",
            name="flux_inference_server",
        )

        def on_server_exit(event, launch_context):
            if launch_context.is_shutdown:
                return []
            return [
                LogInfo(
                    msg=(
                        "ERROR: FLUX inference service exited with code "
                        f"{event.returncode}; shutting down launch"
                    )
                ),
                EmitEvent(event=Shutdown(reason="FLUX inference service exited")),
            ]

        actions.extend([
            RegisterEventHandler(
                OnProcessExit(target_action=server_process, on_exit=on_server_exit)
            ),
            server_process,
        ])
    elif server_ready:
        actions.append(LogInfo(msg=f"Reusing ready FLUX service at {server_url}"))
    elif start_server:
        actions.append(LogInfo(
            msg=f"FLUX URL {server_url} is remote; waiting without starting locally"
        ))

    if not wait_for_server:
        actions.append(node)
        return actions

    readiness_probe = ExecuteProcess(
        cmd=[
            sys.executable,
            "-m",
            "flux_image_editor.wait_ready",
            "--url",
            server_url + "/ready",
            "--timeout",
            ready_timeout,
        ],
        output="screen",
        name="flux_wait_ready",
    )

    def on_probe_exit(event, launch_context):
        if launch_context.is_shutdown:
            return []
        if event.returncode == 0:
            return [node]
        return [
            LogInfo(msg="ERROR: FLUX service readiness timed out; bridge not started"),
            EmitEvent(event=Shutdown(reason="FLUX service readiness failed")),
        ]

    actions.extend([
        RegisterEventHandler(
            OnProcessExit(target_action=readiness_probe, on_exit=on_probe_exit)
        ),
        readiness_probe,
    ])
    return actions


def generate_launch_description():
    configuration_dir = config_dir().resolve()
    package_root = configuration_dir.parent
    ros_parameters = load_ros_parameters()
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=str(configuration_dir / "flux_image_editor.yaml"),
            description="FLUX bridge ROS parameter file",
        ),
        DeclareLaunchArgument(
            "server_url",
            default_value=str(ros_parameters["server_url"]),
            description="FLUX inference service URL",
        ),
        DeclareLaunchArgument(
            "start_server",
            default_value="true",
            description="Start the local service unless server_url is already ready",
        ),
        DeclareLaunchArgument(
            "wait_for_server",
            default_value="true",
            description="Wait for /ready before starting the ROS bridge",
        ),
        DeclareLaunchArgument(
            "ready_timeout_sec",
            default_value="900",
            description="Maximum FLUX startup wait in seconds",
        ),
        DeclareLaunchArgument(
            "inference_python",
            default_value=_default_inference_python(package_root),
            description="Python interpreter containing torch and diffusers",
        ),
        DeclareLaunchArgument(
            "server_config",
            default_value=str(configuration_dir / "server.yaml"),
            description="FLUX inference server YAML path",
        ),
        DeclareLaunchArgument(
            "model_cache",
            default_value=_default_model_cache(package_root),
            description="Hugging Face model cache directory",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
