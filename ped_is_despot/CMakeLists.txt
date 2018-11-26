cmake_minimum_required(VERSION 3.8)


set(CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples")


project(ped_is_despot CXX CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()


# set(CMAKE_C_COMPILER "/usr/bin/gcc-6")

# set(CMAKE_CXX_COMPILER "/usr/bin/gcc-6")

# set(CUDAHOSTCXX "g++-6")


set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/${PROJECT_NAME}/cmake")

set(DESPOT_BUILD_EXAMPLES ON CACHE BOOL "Build C++ model examples")
set(DESPOT_BUILD_POMDPX ON CACHE BOOL "Build POMDPX example")
set(DESPOT_BUILD_GPUS ON CACHE BOOL "Build GPU despot")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")


## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++0x -O3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(Torch REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  std_msgs
  sensor_msgs
  message_generation
  cluster_assoc
  pnc_msgs
  car_hyp_despot
  peds_unity_system
)


#add_subdirectory(src/HypDespot/)
#add_subdirectory(src/planner/)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

add_message_files(
    DIRECTORY msg
    FILES
    ped_belief.msg
    ped_local_frame.msg
    ped_local_frame_vector.msg
    peds_believes.msg
    imitation_data.msg
    car_info.msg
    ped_info.msg
    peds_info.msg
)

generate_messages(
    DEPENDENCIES
        std_msgs
        geometry_msgs
        nav_msgs
        peds_unity_system
)

## Generate messages in the 'msg' folder
#add_message_files(
#  FILES
#  cluster.msg
#  pedestrian.msg
#  pedestrian_array.msg
#)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
#generate_messages(
#  DEPENDENCIES
#  geometry_msgs
#  std_msgs
#)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this

## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ped_path_predictor
 CATKIN_DEPENDS geometry_msgs roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ####src/RVO2/src
  ../car_hyp_despot/src/RVO2/src/
  ../car_hyp_despot/src/planner/
  ../car_hyp_despot/src/HypDespot/include/
  ../car_hyp_despot/src/HypDespot/include/despot
)



####FILE(GLOB LIBRVO_SRCS  src/RVO2/src/*.cpp)
#FILE(GLOB LIBDESPOT_SRCS src/despot/src/*.cpp src/despot/src/core/*.cpp src/despot/src/ippc/*.cpp src/despot/src/pomdpx/*.cpp src/despot/src/pomdpx/parser/*.cpp src/despot/src/solver/*.cpp src/despot/src/test/*.cpp src/despot/src/util/*.cpp src/despot/src/util/tinyxml/*.cpp src/planner/*.cpp)

## Declare a C++ library
####add_library(rvo
####   ${LIBRVO_SRCS}
####)

#add_library(despot
#   ${LIBDESPOT_SRCS}
#)


## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(ped_pomdp src/PedPomdpNode.cpp src/controller.cpp src/world_simulator.cpp src/pomdp_simulator.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_executable(local_frame src/local_frame.cpp)
add_executable(vel_publisher src/VelPublisher.cpp)

## Specify libraries to link a library or executable target against
#target_link_libraries(
   #GPUmodel
   #/home/panpan/workspace/catkin_ws/src/ped_is_despot/src/planner/build/libGPUmodel.a
   #${catkin_LIBRARIES}
   #/home/panpan/workspace/catkin_ws/src/ped_is_despot/src/HypDespot/build/libHypDespot.a
   #HypDespot
#)

# LINK_LIST = "torch;caffe2_library;caffe2_gpu_library;/usr/lib/x86_64-linux-gnu/libcuda.so;/usr/local/cuda-9.0/lib64/libnvrtc.so;/usr/local/cuda-9.0/lib64/libnvToolsExt.so;/usr/local/cuda-9.0/lib64/libcudart_static.a;-lpthread;dl;/usr/lib/x86_64-linux-gnu/librt.so"
message("@@@@@@@@@@@@@@ ped_is_despot linking @@@@@@@@@@@@@@")

# set(LINK_LIST "torch;/usr/lib/x86_64-linux-gnu/libcuda.so;/usr/local/cuda-9.0/lib64/libnvrtc.so;/usr/local/cuda-9.0/lib64/libnvToolsExt.so;/usr/local/cuda-9.0/lib64/libcudart_static.a;-lpthread;dl;/usr/lib/x86_64-linux-gnu/librt.so")
 
# target_link_libraries(ped_pomdp "${LINK_LIST}")
# target_link_libraries(local_frame "${LINK_LIST}")


# target_link_libraries(ped_pomdp "${TORCH_LIBRARIES}")
# target_link_libraries(local_frame "${TORCH_LIBRARIES}")

target_link_libraries(ped_pomdp
   ${catkin_LIBRARIES}
   ####rvo
   car_hyp_despot
   # "${TORCH_LIBRARIES}"
   #/home/panpan/workspace/catkin_ws/src/ped_is_despot/src/planner/build/libGPUmodel.a
)

target_link_libraries(local_frame
   ${catkin_LIBRARIES}
   car_hyp_despot
)

target_link_libraries(vel_publisher
   ${catkin_LIBRARIES}
)


add_dependencies(ped_pomdp caffe2)
add_dependencies(ped_pomdp caffe2_gpu_library)
add_dependencies(local_frame caffe2_library)
add_dependencies(local_frame caffe2_gpu_library)

add_dependencies(ped_pomdp ped_pathplan_generate_messages_cpp)

add_dependencies(ped_pomdp ped_is_despot_generate_messages_cpp)
add_dependencies(ped_pomdp cluster_assoc_generate_messages_cpp)

add_dependencies(local_frame ped_is_despot_generate_messages_cpp)
add_dependencies(local_frame pnc_msgs_generate_messages_cpp)
add_dependencies(vel_publisher ped_is_despot_generate_messages_cpp)

message("@@@@@@@@@@@@@@ ped_is_despot end @@@@@@@@@@@@@@")

#############
## Install ##
#############


# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_ped_path_predictor.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
