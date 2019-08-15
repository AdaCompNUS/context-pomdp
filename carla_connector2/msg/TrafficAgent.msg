# update time
time last_update

# general
int32 id
string type

# geometric information
geometry_msgs/Pose pose
geometry_msgs/Polygon bbox

# intention (paths)
bool reset_intention
nav_msgs/Path[] path_candidates
bool[] cross_dirs







