cmake_minimum_required(VERSION 3.8)

set(CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples")

project(crowd_pomdp_planner CXX CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")

add_definitions(-std=c++0x -O3)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  msg_builder
  tf
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs msg_builder car_hyp_despot  roscpp
  # DEPENDS CUDA
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ../car_hyp_despot/src/gamma/
  ../car_hyp_despot/src/planner/
  ../car_hyp_despot/src/HypDespot/include/
  ../car_hyp_despot/src/HypDespot/include/despot
  ../crowd_pomdp_planner/src/
)

add_library(car_hyp_despot STATIC IMPORTED)
set_target_properties(car_hyp_despot PROPERTIES
  IMPORTED_LOCATION ${CATKIN_DEVEL_PREFIX}/lib/libcar_hyp_despot.a 
)
set_target_properties(car_hyp_despot PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES "CUDA"
)

add_executable(ped_pomdp src/context_pomdp_node.cpp src/controller.cpp src/world_simulator.cpp)
add_executable(local_frame src/local_frame.cpp)
add_executable(vel_publisher src/vel_publisher.cpp)

set_target_properties( ped_pomdp 
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( ped_pomdp
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(ped_pomdp
  car_hyp_despot
  ${catkin_LIBRARIES}
)

target_link_libraries(local_frame
  car_hyp_despot
  ${catkin_LIBRARIES}
)

target_link_libraries(vel_publisher
  car_hyp_despot
  ${catkin_LIBRARIES}
)
