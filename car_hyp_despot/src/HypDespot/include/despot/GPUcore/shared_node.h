/*
 * shard_node.h
 *
 *  Created on: 19 Jul, 2017
 *      Author: panpan
 */

#ifndef SHARD_NODE_H_
#define SHARD_NODE_H_

#include <despot/GPUcore/thread_globals.h>
#include <despot/core/node.h>
#include <despot/core/pomdp.h>
#include <despot/util/util.h>
#include <despot/random_streams.h>
#include <despot/util/logging.h>

#include <despot/GPUcore/msg_queue.h>

namespace despot {

class Shared_QNode;

/* =============================================================================
 * Shared_VNode class
 * =============================================================================*/

/**
 * A belief/value/AND node in the search tree.
 */
class Shared_VNode:public VNode {
protected:
	mutable std::mutex _mutex;
	//mutable bool waiting_change;
public:
	float exploration_bonus;
	std::atomic<int> visit_count_;
	mutable bool is_waiting_;

	std::mutex& GetMutex(){return _mutex;}
	void lock(){_mutex.lock();}
	void unlock(){_mutex.unlock();}
	Shared_VNode(std::vector<State*>& particles, std::vector<int> particleIDs,int depth = 0, Shared_QNode* parent = NULL,
		OBS_TYPE edge = (OBS_TYPE)-1);
	Shared_VNode(Belief* belief, int depth = 0, Shared_QNode* parent = NULL, OBS_TYPE edge =
			(OBS_TYPE)-1);
	Shared_VNode(int count, double value, int depth = 0, Shared_QNode* parent = NULL,
		OBS_TYPE edge = (OBS_TYPE)-1);
	~Shared_VNode();

	Belief* belief() const;
	const std::vector<State*>& particles() const;
	const std::vector<int>& particleIDs() const;
	void depth(int d);
	int depth() const;
	void parent(Shared_QNode* parent);
	Shared_QNode* parent();
	OBS_TYPE edge();

	double Weight();

	//const std::vector<Shared_QNode*>& children() const;
	//std::vector<Shared_QNode*>& children();
	const Shared_QNode* Child(int action) const;
	Shared_QNode* Child(int action);
	int Size() const;
	int PolicyTreeSize() const;

	void default_move(ValuedAction move);
	ValuedAction default_move() const;
	void lower_bound(double value);
	double lower_bound() const;
	void upper_bound(double value);
	double upper_bound(bool use_Vloss) const;
	void utility_upper_bound(double value);
	double utility_upper_bound() const;

	bool IsLeaf();

	void Add(double val);
	void count(int c);
	int count() const;
	void value(double v);
	double value() const;

	//void PrintTree(int depth = -1, std::ostream& os = std::cout);
	void PrintPolicyTree(int depth = -1, std::ostream& os = std::cout);

	void Free(const DSPOMDP& model);

	//double GPUWeight();

	void AddVirtualLoss(float v);
	void RemoveVirtualLoss(float v);
	float GetVirtualLoss();

	void ResizeParticles(int i);
	void ReconstructCPUParticles(const DSPOMDP* model, RandomStreams& streams, History& history);
};

/* =============================================================================
 * Shared_QNode class
 * =============================================================================*/

/**
 * A Q-node/AND-node (child of a belief node) of the search tree.
 */
class Shared_QNode:public QNode  {
protected:
	mutable std::mutex _mutex;
public:
	float exploration_bonus;
	std::atomic<int> visit_count_;
	std::mutex& GetMutex(){return _mutex;}

	void lock(){_mutex.lock();}
	void unlock(){_mutex.unlock();}
	Shared_QNode(Shared_VNode* parent, int edge);
	Shared_QNode(int count, double value);
	~Shared_QNode();

	void parent(Shared_VNode* parent);
	Shared_VNode* parent();
	int edge() const;
	//std::map<OBS_TYPE, Shared_VNode*>& children();
	Shared_VNode* Child(OBS_TYPE obs);
	int Size() const;
	int PolicyTreeSize() const;

	double Weight() /*const*/;

	void lower_bound(double value);
	double lower_bound() const;
	void upper_bound(double value);
	double upper_bound(bool use_Vloss) const;
	void utility_upper_bound(double value);
	double utility_upper_bound() const;

	void Add(double val);
	void count(int c);
	int count() const;
	void value(double v);
	double value() const;

	void AddVirtualLoss(float v);
	void RemoveVirtualLoss(float v);
	float GetVirtualLoss();
};



extern MsgQueque<Shared_VNode> Expand_queue, Print_queue;
} // namespace despot



#endif /* SHARD_NODE_H_ */
