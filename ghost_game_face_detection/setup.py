from glob import glob
from setuptools import find_packages, setup

setup(
    name="ghost_game_face_detection",
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/ghost_game_face_detection"],
        ),
        ("share/ghost_game_face_detection", ["package.xml"]),
        (
            "share/ghost_game_face_detection",
            ["LICENSE", "README.md", "VALIDATION.md"],
        ),
        ("share/ghost_game_face_detection/launch", glob("launch/*.launch.py")),
        ("share/ghost_game_face_detection/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="nvidia",
    maintainer_email="yipeng.xia@hotmail.com",
    description="Nearest-face RGB bounding-box detector",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "face_detection_node = ghost_game_face_detection.node:main",
    ]},
)
