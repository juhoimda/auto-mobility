cmake_minimum_required(VERSION 3.8)
project(auto_mobility)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(auto_mobility PACKAGE_DIR src/auto_mobility)

install(
  DIRECTORY launch config docs
  DESTINATION share/${PROJECT_NAME}
)

# extract_db_rgbd.cpp: TSDF 전용 DB→RGB-D 추출기 소스 (build_extractor.sh 로 컴파일)
install(
  FILES src/auto_mobility/mesh/extract_db_rgbd.cpp
  DESTINATION share/${PROJECT_NAME}/mesh
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
    scripts/utils/run_tests.sh
    scripts/utils/build_extractor.sh
    scripts/utils/fix_imu_permissions.sh
    src/auto_mobility/nodes/republish.py
    src/auto_mobility/nodes/cloud_throttle.py
    src/auto_mobility/nodes/camera_tf_pub.py
    src/auto_mobility/mesh/mesh_open3d.py
    src/auto_mobility/mesh/reconstruct_tsdf.py
    src/auto_mobility/mesh/view_mesh.py
    src/auto_mobility/isaac/load_isaac_mesh.py
    src/auto_mobility/slam/benchmark_slam.py
    src/auto_mobility/utils/benchmark_hw.py
    src/auto_mobility/utils/inspect_system.py
    src/auto_mobility/utils/validate.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

