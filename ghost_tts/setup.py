from glob import glob
from setuptools import setup,find_packages
setup(name='ghost_tts',version='0.2.0',packages=find_packages(),python_requires='>=3.12',
    data_files=[('share/ament_index/resource_index/packages',['resource/ghost_tts']),
                ('share/ghost_tts',[
                    'package.xml','LICENSE','README.md','THIRD_PARTY.md',
                    'VALIDATION.md','requirements.txt']),
                ('share/ghost_tts/config',glob('config/*.yaml')),
                ('share/ghost_tts/launch',glob('launch/*.launch.py'))],
    install_requires=['setuptools'],zip_safe=False,
    maintainer='nvidia',maintainer_email='yipeng.xia@hotmail.com',
    description='Offline cyberpunk TTS for ROS2 Jazzy',license='MIT',tests_require=['pytest'],
    entry_points={'console_scripts':['ghost_tts_node=ghost_tts.node:main','ghost_tts_preview=ghost_tts.preview:main']})
