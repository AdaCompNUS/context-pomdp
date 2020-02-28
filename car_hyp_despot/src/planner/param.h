#ifndef MODELPARAMS_H
#define MODELPARAMS_H
#include<string>
#include <math.h>

namespace ModelParams {
const int N_PED_WORLD = 200;
const int N_PED_IN = 20;

extern double GOAL_TRAVELLED;
extern double VEL_MAX;
extern double NOISE_GOAL_ANGLE;
extern double NOISE_PED_VEL;
extern double CRASH_PENALTY;
extern double REWARD_FACTOR_VEL;
extern double REWARD_BASE_CRASH_VEL;
extern double BELIEF_SMOOTHING;
extern double NOISE_ROBVEL;
extern double COLLISION_DISTANCE;
extern double IN_FRONT_ANGLE_DEG;
extern double LASER_RANGE;
extern int DRIVING_PLACE;
extern double NOISE_PED_POS;
const double PATH_STEP = 0.05;
const double GOAL_TOLERANCE = 2;
const double OBS_LINE_STEP = 0.5; // only check end points

extern double CAR_WIDTH;
extern double CAR_LENGTH;
extern double CAR_WHEEL_DIST;
extern double CAR_FRONT; // car pos may not be measured at rear wheel
extern double CAR_REAR; // car pos may not be measured at rear wheel
extern double MAX_STEER_ANGLE;

const double POS_RLN = 0.4; // position resolution
const double VEL_RLN = 0.3; // velocity resolution

const double CONTROL_FREQ = 3;
const double ACC_SPEED = 3.0;
const double NUM_ACC = 1;

const double NUM_STEER_ANGLE = 7;
const double ANGLE_RLN = MAX_STEER_ANGLE / NUM_STEER_ANGLE; // velocity resolution

const double GOAL_REWARD = 0.0;
const double TIME_REWARD = 0.1;

extern std::string ROS_NS;
extern std::string LASER_FRAME;
extern bool ROS_BRIDG;

inline void InitParams(bool in_simulation) {
	if (in_simulation) {
		ROS_NS = "";
		LASER_FRAME = "/laser_frame";
	} else {
		ROS_NS = "";
		LASER_FRAME = "/laser_frame";
	}
}

const bool CPUDoPrint = false;

void PrintParams();
};

//#define CAR_SIDE_MARGIN 0.0f
//#define CAR_FRONT_MARGIN 0.0f

#define CAR_SIDE_MARGIN 0.8f
#define CAR_FRONT_MARGIN 3.0f
#define PED_SIZE 0.25f
#define CAR_EXPAND_SIZE 0.0f

#endif

