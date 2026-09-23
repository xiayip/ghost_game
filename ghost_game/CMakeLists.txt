cmake_minimum_required(VERSION 3.10)
project(ghost_game)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
