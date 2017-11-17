#ifndef GPUNODE_H
#define GPUNODE_H

#include <despot/GPUcore/GPUpomdp.h>
#include <despot/util/util.h>
//#include <despot/GPUrandom_streams.h>
#include <despot/util/logging.h>
#include <despot/core/node.h>

#include <despot/GPUcore/CudaInclude.h>
#include <despot/GPUutil/GPUmap.h>
#include <despot/GPUutil/GPUFile.h>


namespace despot {
//using namespace archaeopteryx::util;
class Dvc_QNode;

/* =============================================================================
 * Dvc_VNode class
 * =============================================================================*/

/**
 * A belief/value/AND node in the search tree.
 */
class Dvc_VNode {
public:
	Dvc_State* particles_; // Used in DESPOT
	int num_particles_;
	Dvc_Belief* belief_; // Used in AEMS
	int depth_;
	Dvc_QNode* parent_;
	OBS_TYPE edge_;

	Dvc_QNode* children_;
	int num_children_;

	Dvc_ValuedAction default_move_; // Value and action given by default policy
	double lower_bound_;
	double upper_bound_;

	// For POMCP
	int count_; // Number of visits on the node
	double value_; // Value of the node

public:
	Dvc_VNode* vstar;
	double likelihood; // Used in AEMS
	double utility_upper_bound;

	/*DEVICE Dvc_VNode(Dvc_State**& particles, int depth = 0, Dvc_QNode* parent = NULL,
		OBS_TYPE edge = -1);
	DEVICE Dvc_VNode(Dvc_Belief* belief, int depth = 0, Dvc_QNode* parent = NULL, OBS_TYPE edge =
		-1);
	DEVICE Dvc_VNode(int count, double value, int depth = 0, Dvc_QNode* parent = NULL,
		OBS_TYPE edge = -1);
	DEVICE ~Dvc_VNode();

	DEVICE Dvc_Belief* belief() const;
	DEVICE Dvc_State **const& particles() const;
	DEVICE void depth(int d);
	DEVICE int depth() const;
	DEVICE void parent(Dvc_QNode* parent);
	DEVICE Dvc_QNode* parent();
	DEVICE OBS_TYPE edge();

	DEVICE double Weight() const;

	DEVICE Dvc_QNode**const & children() const;
	DEVICE Dvc_QNode**& children();
	DEVICE const Dvc_QNode* Child(int action) const;
	DEVICE Dvc_QNode* Child(int action);
	DEVICE int Size() const;
	DEVICE int PolicyTreeSize() const;

	DEVICE void default_move(Dvc_ValuedAction move);
	DEVICE Dvc_ValuedAction default_move() const;
	DEVICE void lower_bound(double value);
	DEVICE double lower_bound() const;
	DEVICE void upper_bound(double value);
	DEVICE double upper_bound() const;

	DEVICE bool IsLeaf();

	DEVICE void Add(double val);
	DEVICE void count(int c);
	DEVICE int count() const;
	DEVICE void value(double v);
	DEVICE double value() const;

	//DEVICE void PrintTree(int depth = -1, std::ostream& os = std::cout);
	//DEVICE void PrintPolicyTree(int depth = -1, std::ostream& os = std::cout);

	DEVICE void Free(const Dvc_DSPOMDP& model);

	HOST void assign(VNode* host_node);*/
};

/* =============================================================================
 * Dvc_QNode class
 * =============================================================================*/

/**
 * A Q-node/AND-node (child of a belief node) of the search tree.
 */
class Dvc_QNode {
public:
	Dvc_VNode* parent_;
	int edge_;
	//archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*> children_;
	double lower_bound_;
	double upper_bound_;

	// For POMCP
	int count_; // Number of visits on the node
	double value_; // Value of the node

public:
	double default_value;
	double utility_upper_bound;
	double step_reward;
	double likelihood;
	Dvc_VNode* vstar;

	/*DEVICE Dvc_QNode(Dvc_VNode* parent, int edge);
	DEVICE Dvc_QNode(int count, double value);
	DEVICE ~Dvc_QNode();

	DEVICE void parent(Dvc_VNode* parent);
	DEVICE Dvc_VNode* parent();*/
	DEVICE int edge();
	/*DEVICE archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>& children();
	DEVICE Dvc_VNode* Child(OBS_TYPE obs);
	DEVICE int Size() const;
	DEVICE int PolicyTreeSize() const;

	DEVICE double Weight() const;

	DEVICE void lower_bound(double value);
	DEVICE double lower_bound() const;
	DEVICE void upper_bound(double value);
	DEVICE double upper_bound() const;

	DEVICE void Add(double val);
	DEVICE void count(int c);
	DEVICE int count() const;
	DEVICE void value(double v);
	DEVICE double value() const;*/

	HOST void assign(QNode* host_node);
	HOST static void CopyToGPU(Dvc_QNode* Dvc, const QNode* Hst);
};

} // namespace despot

#endif
