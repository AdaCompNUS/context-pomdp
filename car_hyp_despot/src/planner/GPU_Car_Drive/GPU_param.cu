#include"GPU_param.h"

namespace Dvc_ModelParams {
	DEVICE double GOAL_TRAVELLED;
	DEVICE int N_PED_IN;
    DEVICE int N_PED_WORLD;

	DEVICE double VEL_MAX;
    DEVICE double NOISE_GOAL_ANGLE;
    DEVICE double CRASH_PENALTY;
    DEVICE double REWARD_FACTOR_VEL;
    DEVICE double REWARD_BASE_CRASH_VEL;
    DEVICE double BELIEF_SMOOTHING;
    DEVICE double NOISE_ROBVEL;
    DEVICE double COLLISION_DISTANCE;

    DEVICE double IN_FRONT_ANGLE_DEG;

    DEVICE double LASER_RANGE;

	DEVICE double pos_rln; // position resolution
	DEVICE double vel_rln; // velocity resolution

    DEVICE double PATH_STEP;

    DEVICE double GOAL_TOLERANCE;

    DEVICE double PED_SPEED;

	DEVICE bool debug;

	DEVICE double control_freq;
	DEVICE double AccSpeed;

    // deprecated params
	DEVICE double GOAL_REWARD;
}

