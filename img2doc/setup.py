from pathlib import Path

from setuptools import find_packages, setup

package_name = "img2doc"
config_files = ["config/img2doc.yaml", "config/local_api.example.yaml"]
if Path("config/local_api.yaml").is_file():
    config_files.append("config/local_api.yaml")

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", ["launch/img2doc.launch.py"]),
        ("share/" + package_name + "/config", config_files),
    ],
    install_requires=["setuptools", "PyYAML", "requests"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="ddw",
    maintainer_email="ddw@example.com",
    description="ROS 2 image-to-character-card node using DeepSeek vision",
    license="Apache-2.0",
    entry_points={"console_scripts": ["img2doc_node = img2doc.node:main"]},
)
