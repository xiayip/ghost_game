from setuptools import find_packages, setup

package_name = 'ghost_game_orchestrator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/ghost_game.yaml']),
        ('share/' + package_name + '/web',
            ['web/index.html', 'web/style.css', 'web/app.js', 'web/viewer.js']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='yipeng.xia@hotmail.com',
    description=(
        '"Find the Ghost" interaction demo built on the '
        'mit_impedance_controller chain'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ghost_game_node = ghost_game_orchestrator.ghost_game_node:main',
            'ghost_game_mock_solve = ghost_game_orchestrator.mock_solve:main',
            'ghost_game_monitor = ghost_game_orchestrator.monitor:main',
            'ghost_game_web_monitor = ghost_game_orchestrator.web_monitor:main',
        ],
    },
)
