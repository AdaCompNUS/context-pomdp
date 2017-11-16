#include"param.h"

namespace ModelParams {
    double CRASH_PENALTY = -1000;
    double REWARD_FACTOR_VEL = 0.5;
    double REWARD_BASE_CRASH_VEL=0.5;
    double BELIEF_SMOOTHING = 0.05;
    double NOISE_ROBVEL = 0.1; //0; //0.1;
    double NOISE_GOAL_ANGLE = 3.14*0.25;//3.14 * 0.25; //use 0 for debugging
    double COLLISION_DISTANCE = 1.5;
    double IN_FRONT_ANGLE_DEG = 70;
    int DRIVING_PLACE = 0;

	//double VEL_MAX=1.5;
    double VEL_MAX=1.2;

    double LASER_RANGE = 12.0;

	std::string rosns="";
    std::string laser_frame="/scan";
}

