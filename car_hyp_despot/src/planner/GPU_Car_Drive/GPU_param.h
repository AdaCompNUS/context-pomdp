
#ifndef GPUMODELPARAMS_H
#define GPUMODELPARAMS_H
#include <string>
#include <despot/GPUcore/CudaInclude.h>


namespace Dvc_ModelParams {

	DEVICE extern double GOAL_TRAVELLED;
	DEVICE extern int N_PED_IN;
    DEVICE extern int N_PED_WORLD;

	DEVICE extern double VEL_MAX;
    DEVICE extern double NOISE_GOAL_ANGLE;
    DEVICE extern double CRASH_PENALTY;
    DEVICE extern double REWARD_FACTOR_VEL;
    DEVICE extern double REWARD_BASE_CRASH_VEL;
    DEVICE extern double BELIEF_SMOOTHING;
    DEVICE extern double NOISE_ROBVEL;
    DEVICE extern double COLLISION_DISTANCE;

    DEVICE extern double IN_FRONT_ANGLE_DEG;

    DEVICE extern double LASER_RANGE;

	DEVICE extern double pos_rln; // position resolution
	DEVICE extern double vel_rln; // velocity resolution

    DEVICE extern double PATH_STEP;

    DEVICE extern double GOAL_TOLERANCE;

    DEVICE extern double PED_SPEED;

	DEVICE extern bool debug;

	DEVICE extern double control_freq;
	DEVICE extern double AccSpeed;
	DEVICE extern double NumAcc;

	DEVICE extern double MaxSteerAngle;
	DEVICE extern double NumSteerAngle;
//	DEVICE extern double angle_rln=MaxSteerAngle/NumSteerAngle; // velocity resolution

    // deprecated params
	DEVICE extern double GOAL_REWARD;
	DEVICE extern double TIME_REWARD;

    DEVICE extern double CAR_WIDTH;
    DEVICE extern double CAR_LENGTH;
    DEVICE extern double CAR_WHEEL_DIST;
    DEVICE extern double CAR_FRONT; // car pos may not be measured at rear wheel
    DEVICE extern double CAR_REAR; // car pos may not be measured at rear wheel
};

#endif

