#ifndef NODE_H
#define NODE_H

#include <despot/interface/pomdp.h>
#include <despot/util/util.h>
#include <despot/random_streams.h>
#include <despot/util/logging.h>

namespace despot {

class QNode;

/* =============================================================================
 * VNode class
 * =============================================================================*/

/**
 * A belief/value/AND node in the search tree.
 */
class VNode: public MemoryObject {
protected:
  std::vector<State*> particles_; // Used in DESPOT
    std::vector<int> particleIDs_; //Used in GPUDESPOT
    Dvc_State* GPU_particles_; // Used in GPUDESPOT
	Belief* belief_; // Used in AEMS
	int depth_;
	QNode* parent_;
	OBS_TYPE edge_;

	std::vector<QNode*> children_;

	ValuedAction default_move_; // Value and action given by default policy
	double lower_bound_;
	double upper_bound_;

	// For POMCP
	int count_; // Number of visits on the node
	double value_; // Value of the node

public:
	VNode* vstar;
	double likelihood; // Used in AEMS
	double utility_upper_bound_;

	double weight_;
  	int num_GPU_particles_;  // Used in GPUDESPOT

// lets_drive
protected:
  	const double DUMMY_VALUE = 10000000;
  	std::map<ACT_TYPE, double> prior_action_probs_;
  	std::vector<double> prior_steer_probs_;

  	std::vector<ACT_TYPE> legal_actions_;

  	double prior_value_;

  	bool prior_initialized_;

public:
  	double prior_action_probs(int i){return prior_action_probs_[i];}
  	void prior_action_probs(int i, double v){
//  		if (prior_action_probs_.size() > i)
//  			prior_action_probs_[i]=v;
//  		else if (prior_action_probs_.size() == i)
//  			prior_action_probs_.push_back(v);
//  		else
//  			assert(false);

  		prior_action_probs_[i]=v;
  	}
  	std::map<ACT_TYPE, double>& prior_action_probs(){
  		return prior_action_probs_;
  	}

  	double prior_steer_probs(int i){return prior_steer_probs_[i];}
  	std::vector<double>& prior_steer_probs(){return prior_steer_probs_;}

  	void prior_steer_probs(int i, double v){
		if (prior_steer_probs_.size() > i)
			prior_steer_probs_[i]=v;
		else if (prior_steer_probs_.size() == i)
			prior_steer_probs_.push_back(v);
		else
			assert(false);
	}

  	void legal_actions(std::vector<ACT_TYPE> actions){
  		legal_actions_ = actions;
  	}

  	std::vector<ACT_TYPE>& legal_actions(){
  		return legal_actions_;
  	}

	double prior_value();
  	void prior_value(double v){prior_value_ = v;}

  	void print_action_probs();

  	void prior_initialized(bool v){
  		prior_initialized_ = v;
  	}
  	bool prior_initialized(){
  		return prior_initialized_;
  	}
  	ACT_TYPE max_prob_action();
// lets_drive

public:
	VNode():prior_value_(DUMMY_VALUE){	prior_initialized_ = false; }
	VNode(std::vector<State*>& particles, std::vector<int> particleIDs, int depth = 0, QNode* parent = NULL,
		OBS_TYPE edge = (OBS_TYPE)-1);
	VNode(Belief* belief, int depth = 0, QNode* parent = NULL, OBS_TYPE edge =
			(OBS_TYPE)-1);
	VNode(int count, double value, int depth = 0, QNode* parent = NULL,
		OBS_TYPE edge = (OBS_TYPE)-1);
	~VNode();

	void Initialize(std::vector<State*>& particles, std::vector<int> particleIDs,int depth = 0, QNode* parent = NULL,
			OBS_TYPE edge = (OBS_TYPE)-1);
	Belief* belief() const;
	const std::vector<State*>& particles() const;
	const std::vector<int>& particleIDs() const;
	void depth(int d);
	int depth() const;
	void parent(QNode* parent);
	QNode* parent();
	OBS_TYPE edge();

	double Weight();
	bool PassGPUThreshold();
	const std::vector<QNode*>& children() const;
	std::vector<QNode*>& children();
	const QNode* Child(int action) const;
	QNode* Child(int action);
	int Size() const;
	int PolicyTreeSize() const;

	void default_move(ValuedAction move);
	ValuedAction default_move() const;
	void lower_bound(double value);
	double lower_bound() const;
	void upper_bound(double value);
	double upper_bound() const;
	void utility_upper_bound(double value);
	double utility_upper_bound() const;

	bool IsLeaf();

	void Add(double val);
	void count(int c);
	int count() const;
	void value(double v);
	double value() const;

	void PrintTree(int depth = -1, std::ostream& os = std::cout);
	void PrintPolicyTree(int depth = -1, std::ostream& os = std::cout);

	void Free(const DSPOMDP& model);
	/*GPU particle functions*/
	void AssignGPUparticles( Dvc_State* src, int size);
	Dvc_State* GetGPUparticles(){return GPU_particles_;};

	double GPUWeight();
	void ResizeParticles(int i);
	void ReadBackCPUParticles(const DSPOMDP* model);
	void ReconstructCPUParticles(const DSPOMDP* model, RandomStreams& streams, History& history);
};

/* =============================================================================
 * QNode class
 * =============================================================================*/

/**
 * A Q-node/AND-node (child of a belief node) of the search tree.
 */
class QNode : public MemoryObject{
protected:
	VNode* parent_;
	int edge_;
	std::map<OBS_TYPE, VNode*> children_;
	double lower_bound_;
	double upper_bound_;

	// For POMCP
	int count_; // Number of visits on the node
	double value_; // Value of the node

	// Let's drive
	bool prior_values_ready_;
	//


public:
	double default_value;
	double utility_upper_bound_;
	double step_reward;
	double likelihood;

	// Let's drive
	double prior_probability_; // prior probability for choosing the node
	//

	VNode* vstar;

	double weight_;
	QNode();//{;}
	QNode(VNode* parent, int edge);
	QNode(int count, double value);
	~QNode();

	void parent(VNode* parent);
	VNode* parent();
	int edge() const;
	void edge(int edge) {edge_=edge;};
	std::map<OBS_TYPE, VNode*>& children();
	VNode* Child(OBS_TYPE obs);
	int Size() const;
	int PolicyTreeSize() const;

	double Weight() /*const*/;

	void lower_bound(double value);
	double lower_bound() const;
	void upper_bound(double value);
	double upper_bound() const;
	void utility_upper_bound(double value);
	double utility_upper_bound() const;

	void Add(double val);
	void count(int c);
	int count() const;
	void value(double v);
	double value() const;

	// Let's drive
	void prior_probability(double p);
	double prior_probability() const;

	void prior_values_ready(bool v){prior_values_ready_ = v; };
	bool prior_values_ready() const {return prior_values_ready_;};
	//
};


} // namespace despot

#endif
