cmake_minimum_required(VERSION 2.4.6)
project(cluster_assoc)

set(CMAKE_CXX_STANDARD 11)


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
  Landmark.msg
  LandmarksInfo.msg
  SkeletonJoint.msg
  User.msg
  UserInfo.msg
  Gesture.msg
  Gestures.msg
  Pointing.msg
  Wave.msg
  Frame.msg
  FrameTest.msg
  RectWithConfidence.msg
  EulerAngles.msg
  EulerAnglesWithConfidence.msg
  PersonModuleState.msg
  pedestrian.msg
  pedestrian_array.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
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
  cluster_assoc_generate_messages_cpp
    ${PROJECT_NAME}_gencfg
    ${catkin_EXPORTED_TARGETS}
)

add_executable(clusterAssoc src/cluster_assoc.cpp)
add_dependencies(clusterAssoc ${catkin_EXPORTED_TARGETS})
target_link_libraries(clusterAssoc 
#  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)


