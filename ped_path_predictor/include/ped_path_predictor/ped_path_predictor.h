#ifndef PED_PATH_PREDICTOR_PATH_PREDICTOR_H
#define PED_PATH_PREDICTOR_PATH_PREDICTOR_H

#include <ped_path_predictor/world_state.h>
#include "RVO.h"
#include <sensor_msgs/PointCloud.h>

class PedPathPredictor{
public:
	WorldStateHistroy* world_his;
	WorldState state_cur;
	RVO::RVOSimulator* sim;
	ros::Publisher ped_prediction_pub;
	string global_frame_id;
	ros::Timer timer;
	ros::NodeHandle nh;

	void Predict(const ros::TimerEvent &e);
	PedPathPredictor();
	~PedPathPredictor();
	void UpdatePed(PedStruct &ped, double cur_pos_x, double cur_pos_y);
	void AddObstacle();//adding obstables to the map for the RVO simulator
};

#endif