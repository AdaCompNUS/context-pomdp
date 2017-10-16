cmake_minimum_required(VERSION 2.8.3)
project(ped_costmap)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  dynamic_reconfigure
  geometry_msgs
  laser_geometry
  map_msgs
  message_filters
  message_generation
  nav_msgs
  pcl_conversions
  pcl_ros
  pluginlib
  roscpp
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
  voxel_grid
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
    DIRECTORY msg
    FILES
    VoxelGrid.msg
)

generate_messages(
    DEPENDENCIES
        std_msgs
        geometry_msgs
        map_msgs
)

# dynamic reconfigure
generate_dynamic_reconfigure_options(
  cfg/Costmap2D.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ped_costmap
#  CATKIN_DEPENDS geometry_msgs roscpp std_msgs
#  DEPENDS system_lib
)

add_library(ped_costmap src/costmap_2d.cpp src/observation_buffer.cpp src/costmap_2d_ros.cpp src/costmap_2d_publisher.cpp src/voxel_costmap_2d.cpp)

add_executable(costmap_2d_markers src/costmap_2d_markers.cpp)
add_executable(costmap_2d_cloud src/costmap_2d_cloud.cpp)
add_executable(costmap_2d_node src/costmap_2d_node.cpp)

add_dependencies(ped_costmap ped_costmap_generate_messages_cpp)
add_dependencies(costmap_2d_markers ped_costmap_generate_messages_cpp)
add_dependencies(costmap_2d_cloud ped_costmap_generate_messages_cpp)
add_dependencies(costmap_2d_node ped_costmap_generate_messages_cpp)

target_link_libraries(ped_costmap
   ${catkin_LIBRARIES}
)

target_link_libraries(costmap_2d_node
   ${catkin_LIBRARIES}
   ped_costmap
)

target_link_libraries(costmap_2d_markers
   ${catkin_LIBRARIES}
)

target_link_libraries(costmap_2d_cloud
   ${catkin_LIBRARIES}
)
