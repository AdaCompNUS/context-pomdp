#include"param.h"

namespace ModelParams {
	double GOAL_TRAVELLED=20.0;
    double CRASH_PENALTY = -1000;
	double REWARD_FACTOR_VEL = 0.5/*2.0*//*1.0*/;
	double REWARD_BASE_CRASH_VEL=0.5;
	double BELIEF_SMOOTHING = 0.05;
	double NOISE_ROBVEL = 0.01;
	double NOISE_GOAL_ANGLE = 3.14*0.1; //use 0 for debugging
    double NOISE_PED_POS = 0.2; // 0.6;
    double COLLISION_DISTANCE = 1.5;
	double IN_FRONT_ANGLE_DEG = 181;
    int DRIVING_PLACE = 0;

	double VEL_MAX=1.5;

	double LASER_RANGE = 10.0;
 
	std::string rosns="";
	std::string laser_frame="/laser_frame";
}

