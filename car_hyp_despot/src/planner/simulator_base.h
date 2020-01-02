#pragma once

#include <core/globals.h>
#include <ros/node_handle.h>
#include <string>

class WorldBeliefTracker;
class WorldModel;
class WorldStateTracker;

using namespace despot;

class SimulatorBase  {
protected:
	double target_speed_;
	double time_scale_;
    double steering_;

    ACT_TYPE buffered_action_;

public:
	ros::NodeHandle& nh;

	double real_speed;
	std::string global_frame_id;
   	WorldBeliefTracker* beliefTracker;


    static WorldModel worldModel;
   	static WorldStateTracker* stateTracker;

	static bool agents_data_ready;
	static bool agents_path_data_ready;

public:
	SimulatorBase(ros::NodeHandle&_nh):
		nh(_nh), time_scale_(1.0), buffered_action_(0),
		beliefTracker(NULL), target_speed_(0), real_speed(0), steering_(0) {}
};


static double marker_colors[20][3] = {
    	{0.0,1.0,0.0},  //green
		{1.0,0.0,0.0},  //red
		{0.0,0.0,1.0},  //blue
		{1.0,1.0,0.0},  //sky blue
		{0.0,1.0,1.0},  //yellow
		{0.5,0.25,0.0}, //brown
		{1.0,0.0,1.0}   //pink
};

static int action_map[3]={2,0,1};
