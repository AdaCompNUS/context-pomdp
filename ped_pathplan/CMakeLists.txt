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
  ped_costmap
  message_filters
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


include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    #${PCL_INCLUDE_DIRS}
    #${Boost_INCLUDE_DIRS}
)

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

add_library(ped_pathplan src/pathplan.cpp src/pathplan_ros.cpp)

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
