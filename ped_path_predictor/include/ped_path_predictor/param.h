#ifndef MODELPARAMS_PED_PATH_PREDICTOR_H
#define MODELPARAMS_PED_PATH_PREDICTOR_H
#include <string>

namespace ModelParams {

	const int N_PED_IN=6;
    const int N_PED_WORLD=30;
    const int N_HISTORY_STEP=10;
    const int N_SIM_STEP=20; // number of pedestrian simulation steps

    const double OVERLAP_THRESHOLD=0.1;
    const double OUTDATED_THRESHOLD=1.0;
    const double VEL_ZERO_THRESHOLD=0.01; // velocity smaller than this value will be treated as 0.

    const double RVO2_TIME_PER_STEP=0.33;
    const double RVO2_PREVEL_WEIGHT=0.8;

    const double WEIGHT_DISCOUNT = 0.95;

    const double control_freq=3;

    

	const double pos_rln=0.5; // position resolution
	const double vel_rln=0.03; // velocity resolution

    const double PATH_STEP = 0.05;

    const double GOAL_TOLERANCE = 2;

    const double PED_SPEED = 1.2;

	const bool debug=false;

	
	const double AccSpeed=0.5;


    const double GOAL_REWARD = 0.0;


    const double CRASH_PENALTY = -1000;
    const double REWARD_FACTOR_VEL = 0.5;
    const double REWARD_BASE_CRASH_VEL=0.5;
    const double BELIEF_SMOOTHING = 0.05;
    const double NOISE_ROBVEL = 0.1; //0; //0.1;
    const double NOISE_GOAL_ANGLE = 3.14*0.25;//3.14 * 0.25; //use 0 for debugging
    const double COLLISION_DISTANCE = 1.5;
    const double IN_FRONT_ANGLE_DEG = 70;
    const int DRIVING_PLACE = 0;
    
    const double VEL_MAX=1.2;
    const double LASER_RANGE = 12.0;

    const std::string rosns="/scooter";
    const std::string laser_frame="/front_top_lidar";

};

#endif

