cmake_minimum_required(VERSION 3.8)
project(auto_mobility)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(auto_mobility PACKAGE_DIR src/auto_mobility)

install(
  DIRECTORY launch config rviz docs
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/common.sh
    scripts/pipeline/run_bag.sh
    scripts/pipeline/run_live.sh
    scripts/pipeline/record.sh
    scripts/pipeline/play.sh
    scripts/pipeline/mesh.sh
    scripts/pipeline/isaac.sh
    scripts/pipeline/run_pipeline_all.sh
    scripts/utils/check.sh
    scripts/utils/export_ply.sh
    scripts/utils/benchmark.sh
    scripts/utils/view_mesh.sh
    src/auto_mobility/nodes/republish.py
    src/auto_mobility/processing/mesh_open3d.py
    src/auto_mobility/processing/validate.py
    src/auto_mobility/processing/benchmark_hw.py
    src/auto_mobility/processing/inspect_system.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

