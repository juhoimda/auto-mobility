cmake_minimum_required(VERSION 3.8)
project(auto_mobility)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config rviz docs
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY src/auto_mobility
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/common.sh
    scripts/pipeline/run_bag.sh
    scripts/pipeline/run_live.sh
    scripts/pipeline/record.sh
    scripts/pipeline/play.sh
    scripts/pipeline/mesh.sh
    scripts/utils/check.sh
    scripts/utils/export_ply.sh
    scripts/utils/benchmark.sh
    src/auto_mobility/nodes/republish.py
    src/auto_mobility/processing/mesh_open3d.py
    src/auto_mobility/processing/validate.py
    src/auto_mobility/processing/benchmark_hw.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
