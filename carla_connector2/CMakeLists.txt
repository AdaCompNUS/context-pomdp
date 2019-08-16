cmake_minimum_required(VERSION 2.8.3)
project(carla_connector2)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  peds_unity_system
  cluster_assoc
)

add_message_files(
  DIRECTORY msg
  FILES 
  NetworkRoutePoint.msg 
  CrowdNetworkAgent.msg
  CrowdNetworkAgentArray.msg
  SidewalkRoutePoint.msg
  CrowdSidewalkAgent.msg
  CrowdSidewalkAgentArray.msg
  TrafficAgent.msg
  TrafficAgentArray.msg)

 generate_messages(
   DEPENDENCIES
   nav_msgs
 )

catkin_package(
  CATKIN_DEPENDS 
  rospy 
  std_msgs 
  geometry_msgs
  nav_msgs 
  peds_unity_system 
  cluster_assoc
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
