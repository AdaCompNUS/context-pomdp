#ifndef CARPOMDPSIMULATOR_H
#define CARPOMDPSIMULATOR_H
#include <despot/planner.h>

#include "coord.h"
#include "Path.h"
#include "state.h"
#include "WorldModel.h"
#include "ped_pomdp.h"
#include "param.h"

#include <despot/interface/pomdp.h>
#include <despot/core/pomdp_world.h>


#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Twist.h>
#include <rosgraph_msgs/Clock.h>
#include <ped_is_despot/imitation_data.h>

#include "simulator_base.h"


using namespace despot;

class WorldBeliefTracker;

class POMDPSimulator:public POMDPWorld, public SimulatorBase {
public:
    typedef WorldStateTracker::AgentDistPair AgentDistPair;

    //POMDPSimulator(DSPOMDP* model, unsigned seed=0);
    POMDPSimulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed,  std::string obstacle_file_name);

    ~POMDPSimulator();

    double StepReward(PomdpStateWorld& state, ACT_TYPE action);


    int numPedInArea(AgentStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world);

    int numPedInCircle(AgentStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world, double car_x, double car_y);
    int run(int argc, char *argv[]);

    AgentStruct randomPed();
    AgentStruct randomFarPed(double car_x, double car_y);
    AgentStruct randomPedAtCircleEdge(double car_x, double car_y);

    void generateFixedPed(PomdpState &s);

    //virtual DSPOMDP* InitializeModel(option::Option* options) ;
	//virtual void InitializeDefaultParameters();

	void ImportPeds(std::string filename, PomdpStateWorld& world_state);
	void ExportPeds(std::string filename, PomdpStateWorld& world_state);

	void PrintWorldState(PomdpStateWorld state, ostream& out = cout);

    void AddObstacle();

    COORD start, goal;

    Path path;
    //static WorldModel worldModel;

    //WorldStateTracker* stateTracker;
    //WorldBeliefTracker* beliefTracker;

	PomdpStateWorld world_state;
	int num_of_peds_world;

    //double real_speed_;
    //double target_speed_;
   // double steering_;
public:

   // std::string global_frame_id;

    //std::string obstacle_file_name_;



public:
    // for imitation learning

    //ros::Subscriber carSub_;
    ros::Subscriber steerSub_;

    //ros::Publisher IL_pub; 
    //bool b_update_il;
    //ped_is_despot::imitation_data p_IL_data;


    void publishImitationData(PomdpStateWorld& planning_state, ACT_TYPE safeAction, float reward, float vel);


public:

	virtual bool Connect();

	virtual State* Initialize();

	virtual State* GetCurrentState() const;

	virtual bool ExecuteAction(ACT_TYPE action, OBS_TYPE& obs);

};

#endif
