from glob import glob
from setuptools import find_packages, setup


setup(
    name="ghost_game_gestures",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/ghost_game_gestures"]),
        ("share/ghost_game_gestures", ["package.xml"]),
        ("share/ghost_game_gestures/launch", glob("launch/*.launch.py")),
        ("share/ghost_game_gestures/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="nvidia",
    maintainer_email="yipeng.xia@hotmail.com",
    description="Latest-frame hand gesture labels and gated interaction requests",
    license="MIT",
    entry_points={"console_scripts": [
        "gesture_node = ghost_game_gestures.node:main",
        "gesture_preview = ghost_game_gestures.preview:main",
        "download_gesture_model = ghost_game_gestures.download_model:main",
        "gesture_action_router = ghost_game_gestures.action_router:main",
    ]},
)
