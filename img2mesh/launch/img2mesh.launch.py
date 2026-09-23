from launch import LaunchDescription
from launch_ros.actions import Node
from img2mesh.config import config_dir


def generate_launch_description():
    config = config_dir()
    return LaunchDescription([
        Node(
            package='img2mesh', executable='img2mesh_node', name='img2mesh',
            parameters=[str(config / 'img2mesh.yaml')],
        ),
    ])
