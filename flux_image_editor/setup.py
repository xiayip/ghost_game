from glob import glob

from setuptools import find_packages, setup


package_name = 'flux_image_editor'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', ['launch/flux_image_editor.launch.py']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/docker', [
            'docker/Dockerfile.dgx_spark', 'docker/compose.yaml',
        ]),
    ],
    install_requires=['setuptools', 'PyYAML', 'requests'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='ddw',
    maintainer_email='ddw@example.com',
    description='ROS 2 bridge for FLUX.2 Klein image editing on DGX Spark',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'flux_image_editor_node = flux_image_editor.ros_node:main',
            'flux_inference_server = flux_image_editor.inference_server:main',
            'flux_test_file = flux_image_editor.file_cli:main',
        ],
    },
)
