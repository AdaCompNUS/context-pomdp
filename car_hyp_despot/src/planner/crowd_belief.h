#ifndef CUSTOM_PARTICLE_BELIEF_H
#define CUSTOM_PARTICLE_BELIEF_H

#include "state.h"
#include "world_model.h"
#include <despot/interface/belief.h>
#include <despot/core/particle_belief.h>

#include <limits>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud.h>

#include <msg_builder/peds_believes.h>

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


class HiddenStateBelief {
	std::vector<std::vector<double>> probs_;
public:
	HiddenStateBelief(int num_intentions, int num_modes);
	void Resize(int new_intentions);
	void Resize(int new_intentions, int new_modes);
	void Reset();
	void Update(WorldModel&, AgentStruct& past_agent, const AgentStruct& cur_agent,
			int intention_id, int mode_id);
	void Normalize();
	void Sample(int& intention, int& mode);
	void Text(std::ostream& out) {
		int mode = 0;
		out << "b: ";
		for (auto& intention_probs: probs_) {
			out << "mode "<< mode << ":";
			for (double prob: intention_probs)
				out << " " << prob;
			mode++;
		}
		out << endl;
	}

	int size(int dim) {
		if (dim ==0)
			return probs_.size();
		else
			return probs_[0].size();
	}
};

struct AgentBelief {
	AgentStruct observable_;
	HiddenStateBelief* belief_;
	std::vector<Path> possible_paths;
	float time_stamp;

	AgentBelief(int num_intentions, int num_modes) {
		belief_ = new HiddenStateBelief(num_intentions, num_modes);
		time_stamp = -1;
	}

	~AgentBelief() {
		delete belief_;
	}

	void Sample(int& goal, int& mode) const;
	void Reset(int new_intention_size);

	void Update(WorldModel& model, const AgentStruct& cur_agent, int num_intentions);
	bool OutDated(double cur_time_stamp) {
		return abs(time_stamp - cur_time_stamp) > 2.0;
	}

	void Text(std::ostream& out) {
		out << "Belief ";
		observable_.ShortText(out);
		belief_->Text(out);
	}
};


class CrowdBelief: public Belief {
	std::map<double, AgentBelief*> sorted_belief_;
	CarStruct car_;

public:
	CrowdBelief(const DSPOMDP* model);
	CrowdBelief(const DSPOMDP* model, History history,
			std::map<double, AgentBelief*> sorted_belief_);

	~CrowdBelief();

	/**
	 * Sample states from a belief.
	 * Returns a set of sampled states.
	 *
	 * @param num Number of states to be sampled
	 */
	std::vector<State*> Sample(int num) const;

	/**
	 * Update the belief.
	 *
	 * @param action The action taken in the last step
	 * @param obs    The observation received in the last step
	 */
	void Update(ACT_TYPE action, OBS_TYPE obs);

	void Update(ACT_TYPE action, const State* obs);

	/**
	 * Returns a copy of this belief.
	 */
	Belief* MakeCopy() const;

	void Text(std::ostream& out) {
		if (logging::level() >= logging::VERBOSE) {
			auto& sorted_beliefs = sorted_belief_;
			for (int i = 0;
					i < sorted_beliefs.size() && i < min(20, ModelParams::N_PED_IN);
					i++) {
				auto& p = *sorted_beliefs[i];
				p.Text(out);
				out << endl;
			}
		}
	}


private:

	WorldModel& world_model_;
};

const double dummy_pos_value = std::numeric_limits<float>::max();


#endif
