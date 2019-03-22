cmake_minimum_required(VERSION 2.8.3)
project(ped_pathplan)

## Add support for C++0x, supported in ROS Kinetic and newer
add_definitions(-std=c++0x)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
  std_msgs
  geometry_msgs
  message_filters
  # ped_costmap
  costmap_2d
  message_generation
  nav_core
  pluginlib
  pcl_conversions
  pcl_ros
  pluginlib
  tf
  visualization_msgs
)

find_package(Eigen)




add_definitions(${EIGEN_DEFINITIONS})

# messages
add_message_files(
    FILES
    StartGoal.msg
)

generate_messages(
    DEPENDENCIES
        std_msgs
        geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
# LIBRARIES strands_gazing
  CATKIN_DEPENDS std_msgs geometry_msgs visualization_msgs
# DEPENDS system_lib
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    #${PCL_INCLUDE_DIRS}
    #${Boost_INCLUDE_DIRS}
)

add_library(ped_pathplan src/pathplan.cpp src/pathplan_ros.cpp)
add_dependencies(ped_pathplan ped_pathplan_generate_messages_cpp)
add_dependencies(ped_pathplan cluster_assoc_generate_messages_cpp)


add_executable(path_planner src/PathPlanner.cpp)
add_executable(send_start_goal src/SendStartGoal.cpp)

target_link_libraries(path_planner
   ${catkin_LIBRARIES}
   ped_pathplan
)

target_link_libraries(send_start_goal
   ${catkin_LIBRARIES}
   ped_pathplan
)



add_dependencies(path_planner ped_pathplan_generate_messages_cpp)
add_dependencies(send_start_goal ped_pathplan_generate_messages_cpp)
