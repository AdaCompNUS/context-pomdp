#ifndef CUSTOM_PARTICLE_BELIEF_H
#define CUSTOM_PARTICLE_BELIEF_H

#include "state.h"
#include <despot/interface/belief.h>
#include <despot/core/particle_belief.h>

#include <limits>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud.h>

#include <car_hyp_despot/peds_believes.h>
//#include <ped_is_despot/ped_local_frame.h>
//#include <ped_is_despot/ped_local_frame_vector.h>

#include <rosgraph_msgs/Clock.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <nav_msgs/GetPlan.h>

class MaxLikelihoodScenario: public ParticleBelief{
public:
	MaxLikelihoodScenario(vector<State*> particles, const DSPOMDP* model,
		Belief* prior = NULL, bool split = true);

	vector<State*> SampleCustomScenarios(int num, vector<State*> particles,
	const DSPOMDP* model) const;

	virtual void Debug() const{}
};

class WorldModel;
class PedPomdp;
class WorldStateTracker;
class WorldBeliefTracker;
class AgentBelief;

class PedPomdpBelief: public ParticleBelief{
public:

	PedPomdpBelief(vector<State*> particles,const DSPOMDP* model);

	virtual Belief* MakeCopy() const;
	/*Update function to be used in SOlver::BeliefUpdate*/
	virtual bool DeepUpdate(const std::vector<const State*>& state_history,
			std::vector<State*>& state_history_for_search,
			const State* cur_state,
			State* cur_state_for_search,
			ACT_TYPE action);
	virtual bool DeepUpdate(const State* cur_state);

	void ResampleParticles(const PedPomdp* model);

	virtual void Update(despot::ACT_TYPE, despot::OBS_TYPE);

	//long double TransProb(const State* state1, const State* state2, ACT_TYPE action) const;
	WorldModel& world_model_;


public:     

	WorldStateTracker* stateTracker;
	WorldBeliefTracker* beliefTracker;
	ros::Publisher agentPredictionPub_,believesPub_, markers_pub, plannerAgentsPub_;
	ros::NodeHandle nh;

	visualization_msgs::MarkerArray markers;

	void UpdateState(const PomdpStateWorld* src_world_state, WorldModel& world_model);
	void SortAgents(PomdpState* sorted_search_state, const PomdpStateWorld* src_world_state);
	void ReorderAgents(PomdpState* target_search_state,
		const PomdpState* ref_search_state,
		const PomdpStateWorld* src_history_state);

	void publishAgentsPrediciton();
	void publishBelief();
	void publishPlannerPeds(const State &s);

	void publishMarker(int , AgentBelief & ped);

	State* GetParticle(int i);

	//inline int GetThreadID() const{return 0/*MapThread(this_thread::get_id())*/;}
};

const double dummy_pos_value=std::numeric_limits<float>::max();

#endif
