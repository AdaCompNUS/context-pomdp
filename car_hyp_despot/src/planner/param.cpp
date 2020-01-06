#include"param.h"

namespace ModelParams {
double GOAL_TRAVELLED = 20.0;
double CRASH_PENALTY = -1000;
double REWARD_FACTOR_VEL = 0.5;
double REWARD_BASE_CRASH_VEL = 0.5;
double BELIEF_SMOOTHING = 0.05;
double NOISE_ROBVEL = 0.01;
double NOISE_GOAL_ANGLE = 3.14 * 0.1; //use 0 for debugging
double NOISE_PED_VEL = 0.3; //use 0 for debugging
double NOISE_PED_POS = 0.2;
double COLLISION_DISTANCE = 1.5;
double IN_FRONT_ANGLE_DEG = 60;
int DRIVING_PLACE = 0;

double VEL_MAX = 8.0;
double LASER_RANGE = 50.0;

double CAR_WIDTH = 2.0;
double CAR_LENGTH = 2.68;
double CAR_WHEEL_DIST = 2.68;
double CAR_FRONT = 1.34;
double CAR_REAR = 1.34;
double MAX_STEER_ANGLE = 35 / 180.0 * M_PI;

std::string ROS_NS = "";
std::string LASER_FRAME = "/laser_frame";
bool ROS_BRIDG = false;

void PrintParams() {
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

	printf("=> ROS_NS=%s\n", ROS_NS.c_str());
	printf("=> LASER_FRAME=%s\n", LASER_FRAME.c_str());
}
}

