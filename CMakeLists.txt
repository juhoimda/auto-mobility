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
    scripts/pipeline/capture_safe.sh
    scripts/pipeline/run_bag.sh
    scripts/pipeline/run_slam.sh
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
    scripts/utils/view_pointcloud.sh
    scripts/utils/run_tests.sh
    scripts/utils/build_extractor.sh
    scripts/utils/fix_imu_permissions.sh
    src/auto_mobility/nodes/republish.py
    src/auto_mobility/nodes/cloud_throttle.py
    src/auto_mobility/nodes/camera_tf_pub.py
    src/auto_mobility/mesh/mesh_open3d.py
    src/auto_mobility/mesh/reconstruct_tsdf.py
    src/auto_mobility/mesh/worker.py
    src/auto_mobility/mesh/worker_surface.py
    src/auto_mobility/mesh/view_mesh.py
    src/auto_mobility/mesh/view_pointcloud.py
    src/auto_mobility/isaac/load_isaac_mesh.py
    src/auto_mobility/slam/benchmark_slam.py
    src/auto_mobility/slam/compare_algorithms.py
    src/auto_mobility/benchmark/cli.py
    src/auto_mobility/utils/benchmark_hw.py
    src/auto_mobility/utils/inspect_system.py
    src/auto_mobility/utils/validate.py
    src/auto_mobility/utils/validate_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

set(ORB_SLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ORB_SLAM3)
set(PANGOLIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/installed)

if(EXISTS "${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so")
  add_executable(orbslam3_rgbd_node src/auto_mobility/slam/orbslam3_rgbd_node.cpp)
  target_compile_features(orbslam3_rgbd_node PRIVATE cxx_std_14)
  target_include_directories(orbslam3_rgbd_node PRIVATE
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${PANGOLIN_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
  )
  target_link_directories(orbslam3_rgbd_node PRIVATE
    ${ORB_SLAM3_DIR}/lib
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib
    ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib
    ${PANGOLIN_DIR}/lib
  )
  target_link_libraries(orbslam3_rgbd_node
    ORB_SLAM3
    DBoW2
    g2o
    pangolin
    OpenGL::GL
    ${GLEW_LIBRARIES}
    pthread
    ${OpenCV_LIBS}
  )
  ament_target_dependencies(orbslam3_rgbd_node
    rclcpp
    sensor_msgs
    geometry_msgs
    nav_msgs
    cv_bridge
    message_filters
  )
  install(TARGETS orbslam3_rgbd_node DESTINATION lib/${PROJECT_NAME})
endif()

set(STELLA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/installed)
if(EXISTS "${STELLA_DIR}/lib/libstella_vslam.so")
  add_executable(stella_rgbd_node src/auto_mobility/slam/stella_rgbd_node.cpp)
  target_compile_features(stella_rgbd_node PRIVATE cxx_std_17)
  target_include_directories(stella_rgbd_node PRIVATE
    ${STELLA_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_directories(stella_rgbd_node PRIVATE
    ${STELLA_DIR}/lib
  )
  target_link_libraries(stella_rgbd_node
    stella_vslam
    fbow
    pthread
    ${OpenCV_LIBS}
  )
  ament_target_dependencies(stella_rgbd_node
    rclcpp
    sensor_msgs
    geometry_msgs
    nav_msgs
    cv_bridge
    message_filters
  )
  install(TARGETS stella_rgbd_node DESTINATION lib/${PROJECT_NAME})
endif()

ament_package()

