# Collection of points where the car needs to activate the blinkers
pnc_msgs/sig_pt[] sig_pts

# location of interactive points: places where the passenger must
# click to let the car move.
# defined as the waypoint number on the path
int64[] int_pts

int64 cur_node
float64 node_tracking_dist
geometry_msgs/Pose robot_pose