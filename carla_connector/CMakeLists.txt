cmake_minimum_required(VERSION 2.8.3)
project(carla_connector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  ped_is_despot
  nav_msgs
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
  TrafficAgentArray.msg
  agent_array.msg
  traffic_agent.msg
 )

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
  ped_is_despot 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
