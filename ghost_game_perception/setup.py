from glob import glob

from setuptools import find_packages, setup


setup(
    name="ghost_game_perception",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/ghost_game_perception"],
        ),
        ("share/ghost_game_perception", ["package.xml", "LICENSE"]),
        ("share/ghost_game_perception/launch", glob("launch/*.launch.py")),
        ("share/ghost_game_perception/config", glob("config/*.yaml")),
        ("share/ghost_game_perception", ["requirements.txt"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="nvidia",
    maintainer_email="yipeng.xia@hotmail.com",
    description="Phase-aware face and gesture perception over one RGB subscription",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception_node = ghost_game_perception.node:main",
            "gesture_action_router = ghost_game_perception.action_router:main",
            "download_gesture_model = ghost_game_perception.download_model:main",
            "gesture_preview = ghost_game_perception.preview:main",
        ],
    },
)
