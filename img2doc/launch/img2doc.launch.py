from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    config = Path(get_package_share_directory("img2doc")) / "config" / "img2doc.yaml"
    return LaunchDescription([
        Node(
            package="img2doc",
            executable="img2doc_node",
            name="img2doc",
            output="screen",
            parameters=[str(config)],
        )
    ])
