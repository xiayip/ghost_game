"""Independent shared-camera recognizer with an optional dry-run action router."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path


def generate_launch_description():
    config = str(Path(get_package_share_directory("ghost_game_gestures")) / "config" / "gestures.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=config),
        DeclareLaunchArgument("model_path", default_value="~/.local/share/ghost_game/gesture_recognizer.task"),
        DeclareLaunchArgument("enable_action_router", default_value="true"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        Node(package="ghost_game_gestures", executable="gesture_node",
             name="gesture_recognition", output="screen",
             parameters=[LaunchConfiguration("config"),
                         {"model_path": LaunchConfiguration("model_path")}]),
        Node(package="ghost_game_gestures", executable="gesture_action_router",
             name="gesture_action_router", output="screen",
             condition=IfCondition(LaunchConfiguration("enable_action_router")),
             parameters=[LaunchConfiguration("config"),
                         {"dry_run": ParameterValue(LaunchConfiguration("dry_run"), value_type=bool)}]),
    ])
