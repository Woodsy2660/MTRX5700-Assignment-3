cmake_minimum_required(VERSION 3.8)
project(turtlebot_landmark_slam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(PROGRAMS
  scripts/ekf_pipeline_node.py
  scripts/landmark_publisher_sim.py
  scripts/odom_to_control_republisher.py
  scripts/map_writer.py
  scripts/evaluate_map.py
  scripts/path_publisher.py
  scripts/landmark_publisher_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
