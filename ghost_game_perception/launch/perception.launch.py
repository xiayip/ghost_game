"""Run the unified face/gesture perception node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = str(
        Path(get_package_share_directory("ghost_game_perception"))
        / "config"
        / "ghost_game.yaml"
    )
    face_models = str(Path(get_package_share_directory("ghost_game")) / "models")
    router_config = str(
        Path(get_package_share_directory("ghost_game_perception"))
        / "config"
        / "gesture_router.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=config),
            DeclareLaunchArgument("initial_mode", default_value="OFF"),
            DeclareLaunchArgument("face_enabled", default_value="true"),
            DeclareLaunchArgument("gesture_enabled", default_value="true"),
            DeclareLaunchArgument("enable_gesture_router", default_value="true"),
            DeclareLaunchArgument("gesture_dry_run", default_value="true"),
            DeclareLaunchArgument(
                "gesture_router_config", default_value=router_config
            ),
            DeclareLaunchArgument("face_model_path", default_value=face_models),
            DeclareLaunchArgument(
                "gesture_model_path",
                default_value=str(
                    Path.home()
                    / ".local/share/ghost_game/gesture_recognizer.task"
                ),
            ),
            Node(
                package="ghost_game_perception",
                executable="perception_node",
                name="ghost_game_perception",
                output="screen",
                additional_env={"MPLCONFIGDIR": "/tmp/ghost-game-matplotlib"},
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "initial_mode": ParameterValue(
                            LaunchConfiguration("initial_mode"), value_type=str
                        ),
                        "face_enabled": ParameterValue(
                            LaunchConfiguration("face_enabled"), value_type=bool
                        ),
                        "gesture_enabled": ParameterValue(
                            LaunchConfiguration("gesture_enabled"), value_type=bool
                        ),
                        "face_model_path": ParameterValue(
                            LaunchConfiguration("face_model_path"), value_type=str
                        ),
                        "gesture_model_path": ParameterValue(
                            LaunchConfiguration("gesture_model_path"), value_type=str
                        ),
                    },
                ],
            ),
            Node(
                package="ghost_game_perception",
                executable="gesture_action_router",
                name="gesture_action_router",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("enable_gesture_router")
                ),
                parameters=[
                    LaunchConfiguration("gesture_router_config"),
                    {
                        "dry_run": ParameterValue(
                            LaunchConfiguration("gesture_dry_run"),
                            value_type=bool,
                        )
                    },
                ],
            ),
        ]
    )
