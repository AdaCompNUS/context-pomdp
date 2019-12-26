#include"param.h"

namespace ModelParams {
	double GOAL_TRAVELLED=20.0;
    double CRASH_PENALTY = -1000;
	double REWARD_FACTOR_VEL = 0.5/*2.0*//*1.0*/;
	double REWARD_BASE_CRASH_VEL=0.5;
	double BELIEF_SMOOTHING = 0.05;
	double NOISE_ROBVEL = 0.01;
	double NOISE_GOAL_ANGLE = 3.14*0.1; //use 0 for debugging
	double NOISE_PED_VEL = 0.3; //use 0 for debugging
    double NOISE_PED_POS = 0.2; // 0.6;
    double COLLISION_DISTANCE = 1.5;
	double IN_FRONT_ANGLE_DEG = 60;
    int DRIVING_PLACE = 0;

	double VEL_MAX=/*1.5*/ 6.0;
	double LASER_RANGE = 50.0;

	double CAR_WIDTH = 2.0;
	double CAR_LENGTH = 2.68;
	double CAR_WHEEL_DIST = 2.68;
	double CAR_FRONT = 1.34;
	double CAR_REAR = 1.34;
	
	std::string rosns="";
	std::string laser_frame="/laser_frame";

	std::string car_model = "summit";// summit, audi_r8 or pomdp_car

	void print_params(){
		printf("ModelParams:\n");
		printf("=> GOAL_TRAVELLED=%f\n", GOAL_TRAVELLED);
		printf("=> CRASH_PENALTY=%f\n", CRASH_PENALTY);
		printf("=> REWARD_FACTOR_VEL=%f\n", REWARD_FACTOR_VEL);
		printf("=> REWARD_BASE_CRASH_VEL=%f\n", REWARD_BASE_CRASH_VEL);
		printf("=> BELIEF_SMOOTHING=%f\n", BELIEF_SMOOTHING);
		printf("=> NOISE_ROBVEL=%f\n", NOISE_ROBVEL);
		printf("=> NOISE_GOAL_ANGLE=%f\n", NOISE_GOAL_ANGLE);
		printf("=> NOISE_PED_VEL=%f\n", NOISE_PED_VEL);
		printf("=> NOISE_PED_POS=%f\n", NOISE_PED_POS);
		printf("=> COLLISION_DISTANCE=%f\n", COLLISION_DISTANCE);
		printf("=> IN_FRONT_ANGLE_DEG=%f\n", IN_FRONT_ANGLE_DEG);
		printf("=> DRIVING_PLACE=%d\n", DRIVING_PLACE);
		printf("=> VEL_MAX=%f\n", VEL_MAX);
		printf("=> LASER_RANGE=%f\n", LASER_RANGE);

		printf("=> rosns=%s\n", rosns.c_str());
		printf("=> laser_frame=%s\n", laser_frame.c_str());
		printf("=> car_model=%s\n", car_model.c_str());
	}
}

