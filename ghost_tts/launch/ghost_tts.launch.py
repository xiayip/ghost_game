from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = str(Path(get_package_share_directory('ghost_tts'))/'config/ghost.yaml')
    model = str(Path.home()/'.local/share/ghost_tts/zh_CN-huayan-medium.onnx')
    return LaunchDescription([
        DeclareLaunchArgument('config',default_value=config),
        DeclareLaunchArgument('model',default_value=model),
        DeclareLaunchArgument('preset',default_value='ghost'),
        DeclareLaunchArgument('audio_device',default_value='pulse'),
        Node(package='ghost_tts',executable='ghost_tts_node',name='ghost_tts',output='screen',
             parameters=[LaunchConfiguration('config'),{
                 'model_path':ParameterValue(LaunchConfiguration('model'),value_type=str),
                 'preset':ParameterValue(LaunchConfiguration('preset'),value_type=str),
                 'audio_device':ParameterValue(LaunchConfiguration('audio_device'),value_type=str)}])])
