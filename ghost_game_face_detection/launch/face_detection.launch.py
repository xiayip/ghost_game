from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(
        Path(get_package_share_directory("ghost_game_face_detection"))
        / "config"
        / "gemini305.yaml"
    )
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=config),
        DeclareLaunchArgument(
            "model_path",
            description="YuNet ONNX file or directory containing versioned YuNet models",
        ),
        Node(package="ghost_game_face_detection", executable="face_detection_node", name="nearest_face",
             output="screen", parameters=[
                 LaunchConfiguration("config"),
                 {"model_path": LaunchConfiguration("model_path")},
             ]),
    ])
