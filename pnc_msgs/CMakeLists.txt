cmake_minimum_required(VERSION 2.8.3)
project(pnc_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  std_msgs
  nav_msgs
  message_generation
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_message_files(
    DIRECTORY msg
    FILES
    local_map.msg
    move_status.msg
    poi.msg
    sig_pt.msg
    speed_contribute.msg
)

generate_messages(
    DEPENDENCIES
        std_msgs
        geometry_msgs
        nav_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs roscpp std_msgs nav_msgs
#  DEPENDS system_lib
)