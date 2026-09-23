cmake_minimum_required(VERSION 3.10)
project(ghost_game_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/Speak.action"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
