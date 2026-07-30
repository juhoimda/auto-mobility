cmake_minimum_required(VERSION 3.8)
project(auto_mobility)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/common.sh
    scripts/record.sh
    scripts/play.sh
    scripts/run_live.sh
    scripts/run_bag.sh
    scripts/check.sh
    scripts/export_pointcloud.sh
    scripts/generate_mesh.sh
    scripts/process_mesh_open3d.py
    scripts/validate_dataset.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
