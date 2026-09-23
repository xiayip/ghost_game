from setuptools import find_packages, setup

package_name = 'img2mesh'
config_files = [
    'config/img2mesh.yaml',
    'config/ghost_game.yaml',
    'config/img2mesh_style.yaml',
    'config/local_api.example.yaml',
]
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', [
            'launch/img2mesh.launch.py',
            'launch/img2mesh_style.launch.py',
        ]),
        ('share/' + package_name + '/config', config_files),
    ],
    install_requires=['setuptools', 'PyYAML'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='yipeng.xia@hotmail.com',
    description='ROS 2 bridge from prepared Ghost portraits to Tripo mesh URLs',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'img2mesh_node = img2mesh.node:main',
        ],
    },
)
