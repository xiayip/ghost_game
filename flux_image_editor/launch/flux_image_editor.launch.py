from launch import LaunchDescription
from launch_ros.actions import Node

from flux_image_editor.config import config_dir


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='flux_image_editor',
            executable='flux_image_editor_node',
            name='flux_image_editor',
            parameters=[str(config_dir() / 'flux_image_editor.yaml')],
            output='screen',
        ),
    ])

