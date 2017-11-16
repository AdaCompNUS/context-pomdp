#!/bin/sh
rosbag record /camera/depth_registered/sw_registered/image_rect /camera/rgb/image_rect_color /odom /tf_static /tf /ORB_SLAM/pose /scan -e '/RosAria/.*' /cmd_vel

