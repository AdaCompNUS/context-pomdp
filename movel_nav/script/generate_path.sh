#rostopic echo -b $1 -p /vslam2d_pose > $2
rostopic echo -b $1 -p /odometry/filtered > $2
