from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
    enable_web_monitor = LaunchConfiguration("enable_web_monitor")
    enable_tts = LaunchConfiguration("enable_tts")
    continuous_face_follow = LaunchConfiguration("continuous_face_follow")
    tts_config = LaunchConfiguration("tts_config")
    tts_model = LaunchConfiguration("tts_model")
    tts_preset = LaunchConfiguration("tts_preset")
    tts_audio_device = LaunchConfiguration("tts_audio_device")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "continuous_face_follow", default_value="false",
                description="Keep following in Stage 2 until abort/home instead of finishing after the dwell",
            ),
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
                "enable_tts",
                default_value="true",
                description="Start offline Piper TTS and enable game announcements",
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
                    {"face_continuous_follow": ParameterValue(continuous_face_follow, value_type=bool)},
                    {
                        "tts_enabled": ParameterValue(
                            enable_tts, value_type=bool
                        )
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
            Node(
                package="ghost_game_orchestrator",
                executable="ghost_game_web_monitor",
                name="ghost_game_web_monitor",
                output="screen",
                condition=IfCondition(enable_web_monitor),
            ),
        ]
    )
