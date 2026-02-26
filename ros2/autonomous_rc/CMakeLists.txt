cmake_minimum_required(VERSION 3.10)
project(autonomous_rc LANGUAGES CXX CUDA)

# ─── C++17 (ROS 2 Foxy 호환) ───
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Release 최적화 (Jetson 에서 필수) ───
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# ═════════════════════════════════════════════
# ROS 2 / ament 의존 (ament_cmake 가장 먼저)
# ═════════════════════════════════════════════
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# ★ cv_bridge는 더 이상 사용하지 않으므로 find_package 불필요

# ═════════════════════════════════════════════
# CUDA
# ═════════════════════════════════════════════
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# ═════════════════════════════════════════════
# TensorRT 8.x  (Xavier NX JetPack 5.x 기본 경로)
# ═════════════════════════════════════════════
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu" CACHE PATH "TensorRT include")
set(TENSORRT_LIB_DIR     "/usr/lib/aarch64-linux-gnu"     CACHE PATH "TensorRT lib")

include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIB_DIR})

# ═════════════════════════════════════════════
# OpenCV (시스템 패키지 — ament 패키지 아님)
# ═════════════════════════════════════════════
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ═════════════════════════════════════════════
# 프로젝트 헤더
# ═════════════════════════════════════════════
include_directories(include)

# ═════════════════════════════════════════════
# 실행 파일
# ═════════════════════════════════════════════
add_executable(autonomous_rc_node
  src/main.cpp
  src/trt_engine.cpp
  src/control_logic.cpp
  src/motor_driver.cpp
)

# ─── 시스템 라이브러리 직접 링크 ───
target_link_libraries(autonomous_rc_node
  ${CUDA_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
  nvinfer
  ${OpenCV_LIBS}
)

# ─── ROS 2 ament 패키지 의존성만 여기에 ───
# 주의: OpenCV는 ament 패키지가 아니므로 절대 여기 넣지 않음
# 주의: cv_bridge 제거됨 (직접 변환 방식 사용)
ament_target_dependencies(autonomous_rc_node
  rclcpp
  std_msgs
  sensor_msgs
)

# ═════════════════════════════════════════════
# Install
# ═════════════════════════════════════════════
install(TARGETS autonomous_rc_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
