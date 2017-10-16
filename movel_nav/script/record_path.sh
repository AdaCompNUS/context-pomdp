#!/bin/sh
rosbag record /tf /ORB_SLAM/pose /vslam2d_pose /cmd_vel /odometry/filtered -o path

