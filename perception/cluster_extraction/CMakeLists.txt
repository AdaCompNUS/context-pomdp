cmake_minimum_required(VERSION 2.4.6)
project(cluster_extraction)

find_package(PCL REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf
  message_filters
  laser_geometry
  pcl_conversions
  message_generation
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


add_message_files(
  FILES
  cluster.msg
  clusters.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
   CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs tf message_filters laser_geometry pcl_conversions message_runtime
)

add_executable(ped_clustering src/ped_clustering.cpp)
add_dependencies(ped_clustering cluster_extraction_generate_messages_cpp)

add_dependencies(ped_clustering ${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ped_clustering
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
)



