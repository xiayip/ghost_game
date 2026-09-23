from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    enable_face_detection = LaunchConfiguration("enable_face_detection")
    face_detection_config = LaunchConfiguration("face_detection_config")
    face_detection_model_path = LaunchConfiguration(
        "face_detection_model_path")
    enable_face_reconstruction = LaunchConfiguration(
        "enable_face_reconstruction")
    face_reconstruction_config = LaunchConfiguration(
        "face_reconstruction_config")
    face_reconstruction_server_url = LaunchConfiguration(
        "face_reconstruction_server_url")
    enable_web_monitor = LaunchConfiguration("enable_web_monitor")
    enable_tts = LaunchConfiguration("enable_tts")
    enable_gestures = LaunchConfiguration("enable_gestures")
    gesture_model_path = LaunchConfiguration("gesture_model_path")
    gesture_dry_run = LaunchConfiguration("gesture_dry_run")
    tts_config = LaunchConfiguration("tts_config")
    tts_model = LaunchConfiguration("tts_model")
    tts_preset = LaunchConfiguration("tts_preset")
    tts_audio_device = LaunchConfiguration("tts_audio_device")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("ghost_game_orchestrator"),
                        "config",
                        "ghost_game.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "enable_face_detection",
                default_value="true",
                description="Start the RGB nearest-face detector",
            ),
            DeclareLaunchArgument(
                "face_detection_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("ghost_game_face_detection"),
                        "config",
                        "gemini305.yaml",
                    ]
                ),
                description="Face-detection parameter file",
            ),
            DeclareLaunchArgument(
                "face_detection_model_path",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("ghost_game"), "models"]
                ),
                description="YuNet model file or model directory",
            ),
            DeclareLaunchArgument(
                "enable_web_monitor",
                default_value="true",
                description="Start the optional web dashboard on port 8765",
            ),
            DeclareLaunchArgument(
                "enable_face_reconstruction",
                default_value="true",
                description=(
                    "Start FLUX full-head preprocessing after stable face capture"
                ),
            ),
            DeclareLaunchArgument(
                "face_reconstruction_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("flux_image_editor"),
                        "config",
                        "ghost_game.yaml",
                    ]
                ),
                description="Ghost-specific FLUX bridge parameter file",
            ),
            DeclareLaunchArgument(
                "face_reconstruction_server_url",
                default_value="http://127.0.0.1:8090",
                description="FLUX HTTP endpoint (local or SSH-forwarded)",
            ),
            DeclareLaunchArgument(
                "enable_tts",
                default_value="true",
                description="Start offline Piper TTS and enable game announcements",
            ),
            DeclareLaunchArgument(
                "enable_gestures", default_value="false",
                description="Start hand gesture labels and the dry-run interaction router",
            ),
            DeclareLaunchArgument(
                "gesture_model_path",
                default_value=str(Path.home() / ".local/share/ghost_game/gesture_recognizer.task"),
            ),
            DeclareLaunchArgument(
                "gesture_dry_run", default_value="true",
                description="Publish interaction requests without calling robot services",
            ),
            DeclareLaunchArgument(
                "tts_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("ghost_tts"), "config", "ghost.yaml"]
                ),
                description="ghost_tts parameter file",
            ),
            DeclareLaunchArgument(
                "tts_model",
                default_value=str(
                    Path.home()
                    / ".local/share/ghost_tts/zh_CN-huayan-medium.onnx"
                ),
                description="Piper ONNX voice model",
            ),
            DeclareLaunchArgument(
                "tts_preset",
                default_value="ghost",
                description="Voice effect preset: clean, subtle, ghost, or machine",
            ),
            DeclareLaunchArgument(
                "tts_audio_device",
                default_value="pulse",
                description="Audio output: pulse, a PortAudio device name/index, or default",
            ),
            Node(
                package="ghost_game_orchestrator",
                executable="ghost_game_node",
                name="ghost_game_node",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "tts_enabled": ParameterValue(
                            enable_tts, value_type=bool
                        ),
                        "enable_face_reconstruction": ParameterValue(
                            enable_face_reconstruction, value_type=bool
                        ),
                    },
                ],
            ),
            Node(
                package="ghost_tts",
                executable="ghost_tts_node",
                name="ghost_tts",
                output="screen",
                parameters=[
                    tts_config,
                    {
                        "model_path": ParameterValue(tts_model, value_type=str),
                        "preset": ParameterValue(tts_preset, value_type=str),
                        "audio_device": ParameterValue(
                            tts_audio_device, value_type=str
                        ),
                    },
                ],
                condition=IfCondition(enable_tts),
            ),
            Node(
                package="ghost_game_face_detection",
                executable="face_detection_node",
                name="nearest_face",
                output="screen",
                parameters=[
                    face_detection_config,
                    {"model_path": face_detection_model_path},
                ],
                condition=IfCondition(enable_face_detection),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("ghost_game_gestures"), "launch", "gestures.launch.py"
                    ])
                ),
                launch_arguments={
                    "model_path": gesture_model_path,
                    "enable_action_router": "true",
                    "dry_run": gesture_dry_run,
                }.items(),
                condition=IfCondition(enable_gestures),
            ),
            Node(
                package="flux_image_editor",
                executable="flux_image_editor_node",
                name="flux_image_editor",
                output="screen",
                parameters=[
                    face_reconstruction_config,
                    {
                        "server_url": ParameterValue(
                            face_reconstruction_server_url, value_type=str
                        ),
                    },
                ],
                condition=IfCondition(enable_face_reconstruction),
            ),
            Node(
                package="ghost_game_orchestrator",
                executable="ghost_game_web_monitor",
                name="ghost_game_web_monitor",
                output="screen",
                condition=IfCondition(enable_web_monitor),
            ),
        ]
    )
