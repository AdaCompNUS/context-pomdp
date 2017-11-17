#ifndef DESPOT_H
#define DESPOT_H

#include <despot/core/solver.h>
#include <despot/core/pomdp.h>
#include <despot/core/belief.h>
#include <despot/core/node.h>
#include <despot/core/globals.h>
#include <despot/core/history.h>
#include <despot/random_streams.h>
#include <despot/GPUcore/shared_node.h>
#include <despot/GPUcore/shared_solver.h>

namespace despot {
class Dvc_RandomStreams;

class DESPOT: public Solver {
friend class VNode;

	static void CPU_MakeNodes(double start, int NumParticles,
			const std::vector<int>& particleIDs, const DSPOMDP* model,
			const std::vector<State*>& particles, QNode* qnode, VNode* parent,
			History& history, ScenarioLowerBound* lb, ScenarioUpperBound* ub,
			RandomStreams& streams);
	static void GPU_MakeNodes(double start, int NumParticles,
			const std::vector<int>& particleIDs, const DSPOMDP* model,
			const std::vector<State*>& particles, QNode* qnode, VNode* parent,
			History& history, ScenarioLowerBound* lb, ScenarioUpperBound* ub,
			RandomStreams& streams);
	static int CalSharedMemSize();

protected:
	VNode* root_;
	Shared_SearchStatistics statistics_;
	//SearchStatistics statistics_;

	ScenarioLowerBound* lower_bound_;
	ScenarioUpperBound* upper_bound_;

	/* GPU despot */
	std::vector<double> Initial_upper;
	std::vector<double> Initial_lower;
	std::vector<double> Final_upper;
	std::vector<double> Final_lower;
	/* GPU despot */


public:
	DESPOT(const DSPOMDP* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, Belief* belief = NULL, bool use_GPU=false);
	virtual ~DESPOT();

	ValuedAction Search();

	void belief(Belief* b);
	void Update(int action, OBS_TYPE obs);

	ScenarioLowerBound* lower_bound() const;
	ScenarioUpperBound* upper_bound() const;

	static VNode* ConstructTree(std::vector<State*>& particles, RandomStreams& streams,
		ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, History& history, double timeout,
		SearchStatistics* statistics = NULL);

protected:
	static VNode* Trial(VNode* root, RandomStreams& streams,
		ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, History& history, SearchStatistics* statistics =
			NULL);
	static Shared_VNode* Trial(Shared_VNode* root, RandomStreams& streams,
		ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, History& history, bool & Expansion_done, Shared_SearchStatistics* statistics =
			NULL);
	static void InitLowerBound(VNode* vnode, ScenarioLowerBound* lower_bound,
		RandomStreams& streams, History& history);
	static void InitUpperBound(VNode* vnode, ScenarioUpperBound* upper_bound,
		RandomStreams& streams, History& history);
	static void InitBounds(VNode* vnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound, RandomStreams& streams, History& history);

	static void Expand(VNode* vnode,
		ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, RandomStreams& streams, History& history);
	static void Expand(Shared_VNode* vnode,
		ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, RandomStreams& streams, History& history);
	static void Backup(VNode* vnode, bool real);

	static double Gap(VNode* vnode);
	static double Gap(Shared_VNode* vnode, bool use_Vloss);

	double CheckDESPOT(const VNode* vnode, double regularized_value);
	double CheckDESPOTSTAR(const VNode* vnode, double regularized_value);
	void Compare();

	static void ExploitBlockers(VNode* vnode);
	static void ExploitBlockers(Shared_VNode* vnode);
	static VNode* FindBlocker(VNode* vnode);
	static void Expand(QNode* qnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound, const DSPOMDP* model,
		RandomStreams& streams, History& history);
	static void Update(VNode* vnode, bool real);
	static void Update(QNode* qnode, bool real);
	static void Update(Shared_VNode* vnode, bool real);
	static void Update(Shared_QNode* qnode, bool real);
	static VNode* Prune(VNode* vnode, int& pruned_action, double& pruned_value);
	static QNode* Prune(QNode* qnode, double& pruned_value);
	static double WEU(VNode* vnode);
	static double WEU(Shared_VNode* vnode);
	static double WEU(VNode* vnode, double epsilon);
	static double WEU(Shared_VNode* vnode, double xi);
	static VNode* SelectBestWEUNode(QNode* qnode);
	static QNode* SelectBestUpperBoundNode(VNode* vnode);
	static Shared_QNode* SelectBestUpperBoundNode(Shared_VNode* vnode);
	static ValuedAction OptimalAction(VNode* vnode);

	/*Debug*/
	static void OutputWeight(QNode* qnode);
	static void OptimalAction2(VNode* vnode);
	/*Debug*/

	static ValuedAction Evaluate(VNode* root, std::vector<State*>& particles,
		RandomStreams& streams, POMCPPrior* prior, const DSPOMDP* model);

	static void GPU_Expand(QNode* qnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound, const DSPOMDP* model,
		RandomStreams& streams, History& history);
	static void GPU_Expand1(QNode* qnode, ScenarioLowerBound* lb,
		ScenarioUpperBound* ub, const DSPOMDP* model,
		RandomStreams& streams,
		History& history);
	static void UpdateData(VNode* vnode,int ThreadID,
			const DSPOMDP* model,RandomStreams& streams);
	static void ReadBackData(int ThreadID);

	static void DataReadBack(VNode* vnode,int ThreadID);
	static void MCSimulation(VNode* vnode, int ThreadID,
			const DSPOMDP* model, RandomStreams& streams,History& history, bool Do_rollout=true);
	static void GPU_Expand_Action(VNode* vnode, ScenarioLowerBound* lb,
		ScenarioUpperBound* ub, const DSPOMDP* model,
		RandomStreams& streams,
		History& history);
	static void GPU_InitBounds(VNode* vnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound,const DSPOMDP* model, RandomStreams& streams,
		History& history);
	static void GPU_UpdateParticles(VNode* vnode, ScenarioLowerBound* lb,
		ScenarioUpperBound* ub, const DSPOMDP* model, RandomStreams& streams,
		History& history);
	void PrepareGPUMemory( const Config& config, int num_actions, int num_obs);
	void PrepareGPUStreams(const RandomStreams& streams, const Config& config, int NumParticles);
	void ClearGPUMemory();
	void PrintGPUData(int num_searches);
	void PrintCPUTime(int num_searches);

	static void GPUHistoryAdd(int action, OBS_TYPE obs, bool for_all_threads=false);
	static void GPUHistoryTrunc(int size, bool for_all_threads=false);

	virtual void initGPUHistory();
	virtual void clearGPUHistory();

	static void PrepareGPUMemory_root( const DSPOMDP* model, const std::vector<int>& particleIDs, std::vector<State*>& particles, VNode* node);

	double AverageInitLower() const;
	double StderrInitLower() const;
	double AverageFinalLower() const;
	double StderrFinalLower() const;

	double AverageInitUpper() const;
	double StderrInitUpper() const;
	double AverageFinalUpper() const;
	double StderrFinalUpper() const;

	void PrintStatisticResult();


	static void ExpandTreeServer(RandomStreams streams,
			ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
			const DSPOMDP* model, History history, Shared_SearchStatistics* statistics,
			double& used_time,double& explore_time,double& backup_time,int& num_trials,double timeout,
			MsgQueque<Shared_VNode>& node_queue, MsgQueque<Shared_VNode>& print_queue, int threadID);

	static void Validate_GPU(int line);

	static float CalExplorationValue(int depth);
	static void CalExplorationValue(Shared_QNode* node);
	static void CalExplorationValue(Shared_VNode* node);

	static void ValidateGPU(const char* file, int line);

	static void ChooseGPUForThread();


	//static bool PassGPUThreshold(VNode* vnode);
public:
	static bool use_GPU_;
	static int num_Obs_element_in_GPU;

	static double Initial_root_gap;
	static bool Debug_mode;

};

} // namespace despot

#endif
