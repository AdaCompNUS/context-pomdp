#ifndef WORLD_STATE_PATH_PREDICTOR_H
#define WORLD_STATE_PATH_PREDICTOR_H

#include <ped_path_predictor/state.h>
#include "cluster_assoc/pedestrian_array.h"
#include <sys/time.h>
#include <mutex>          // std::mutex
#include <ros/ros.h>


using namespace std;

class WorldState{
public:
	CarStruct car;
	std::vector<PedStruct> peds;
	WorldState() {}
};

class WorldStateHistroy{
public:
	double weights[ModelParams::N_HISTORY_STEP];

	vector<WorldState> history;
	WorldState state_cur; //current world state; the same with the last (latest) elem in history but with preferred velcoity set to average velocity

	double starttime;//program start time in second.
	double last_time;

	std::mutex mtx;           // mutex for updating state_cur

	ros::Subscriber ped_sub_;

	WorldStateHistroy(ros::NodeHandle &nh);

	void UpdatePrefVel(); // update the preferred velocities of the pedestrians in current state.

	double Timestamp();

	void CleanPed();
	void UpdatePed(const PedStruct& ped);
	void UpdatePed(WorldState& state, const PedStruct& ped);
	void AddState(WorldState new_state);
	void PedCallback(cluster_assoc::pedestrian_arrayConstPtr ped_array);

	void PrintState(WorldState& state);

	double get_time_second();
	WorldState get_current_state();
};

#endif