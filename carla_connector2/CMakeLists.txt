cmake_minimum_required(VERSION 2.8.3)
project(carla_connector2)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  rospy
  std_msgs
  cluster_assoc
  peds_unity_system
)

add_message_files(
  DIRECTORY msg
  FILES 
  agent_array.msg 
  NetworkRoutePoint.msg 
  CrowdNetworkAgent.msg
  CrowdNetworkAgentArray.msg
  SidewalkRoutePoint.msg
  CrowdSidewalkAgent.msg
  CrowdSidewalkAgentArray.msg
  traffic_agent.msg)

 generate_messages(
   DEPENDENCIES
   nav_msgs
 )

catkin_package(
  CATKIN_DEPENDS nav_msgs rospy std_msgs peds_unity_system cluster_assoc
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
