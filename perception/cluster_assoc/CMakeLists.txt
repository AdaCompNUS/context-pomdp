cmake_minimum_required(VERSION 2.4.6)
project(cluster_assoc)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  message_filters
  tf
  dynamic_reconfigure
  message_generation
  cluster_extraction
)

include_directories(
  ${catkin_INCLUDE_DIRS}  
)

add_message_files(
  FILES
  pedestrian.msg
  pedestrian_array.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  cluster_extraction
)

generate_dynamic_reconfigure_options(
    cfg/ClusterAssoc.cfg
)

catkin_package(
    CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs tf message_runtime message_filters cluster_extraction dynamic_reconfigure
)
 
add_executable(clusterAssoc src/cluster_assoc.cpp)
add_dependencies(clusterAssoc
    ${PROJECT_NAME}_gencfg
    ${catkin_EXPORTED_TARGETS}
)

add_executable(clusterAssoc src/cluster_assoc.cpp)
add_dependencies(clusterAssoc ${catkin_EXPORTED_TARGETS})
target_link_libraries(clusterAssoc 
#  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)


