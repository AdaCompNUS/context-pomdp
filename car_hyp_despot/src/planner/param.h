
#ifndef MODELPARAMS_H
#define MODELPARAMS_H
#include<string>
#include <math.h>

namespace ModelParams {
	const int N_PED_WORLD=200;

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
	const double PED_SPEED = 1.0;

	const double pos_rln=0.4; // position resolution
	const double vel_rln=0.3; // velocity resolution

	const bool debug=false;

	const double control_freq=3;
	const double AccSpeed=1.5;
	const double NumAcc=1;

	const double MaxSteerAngle=35/180.0*M_PI;
	const double NumSteerAngle=7;
	const double angle_rln=MaxSteerAngle/NumSteerAngle; // velocity resolution

	extern std::string rosns;
	extern std::string laser_frame;

	extern std::string car_model;

    inline void init_params(bool in_simulation) {
        if(in_simulation) {
            rosns="";
            laser_frame="/laser_frame";
        } else {
            rosns="";
            laser_frame="/laser_frame";
        }
    }

    // deprecated params
    const double GOAL_REWARD = 0;
    const double TIME_REWARD = 0.1;
    const bool CPUDoPrint=false;

    const bool is_simulation=true;
};


//#define CAR_WIDTH 1.2f
//#define CAR_LENGTH 2.2f
//#define CAR_SIDE_MARGIN 1.2f
//#define CAR_FRONT_MARGIN 1.5f
//#define PED_SIZE 0.25f
//#define CAR_EXPAND_SIZE 0.2f

#define CAR_WIDTH 2.0f
#define CAR_LENGTH 1.68f
#define CAR_SIDE_MARGIN 1.2f
#define CAR_FRONT_MARGIN 1.5f
#define PED_SIZE 0.25f
#define CAR_EXPAND_SIZE 0.0f


#endif

