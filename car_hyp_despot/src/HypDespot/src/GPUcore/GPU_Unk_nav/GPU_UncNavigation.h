#ifndef GPUUNCNAVIGATION_H
#define GPUUNCNAVIGATION_H

#include <despot/GPUcore/GPUpomdp.h>
#include <despot/GPUcore/GPUmdp.h>
#include "GPU_base_unc_navigation.h"
#include <despot/GPUutil/GPUcoord.h>
//#include <despot/util/grid.h>
#include <despot/GPUcore/CudaInclude.h>
namespace despot {

enum RollOut
{
	INDEPENDENT_ROLLOUT,
	GRAPH_ROLLOUT
};

//#define
/* =============================================================================
 * Dvc_UncNavigation class
 * =============================================================================*/

class Dvc_UncNavigation {

	/*friend class UncNavigationENTScenarioLowerBound;
	friend class UncNavigationMMAPStateScenarioLowerBound;
	friend class UncNavigationEastScenarioLowerBound;
	friend class UncNavigationParticleUpperBound1;
	friend class UncNavigationParticleUpperBound2;
	friend class UncNavigationMDPParticleUpperBound;
	friend class UncNavigationApproxParticleUpperBound;
	friend class UncNavigationEastBeliefPolicy;
	friend class UncNavigationMDPBeliefUpperBound;
	friend class UncNavigationPOMCPPrior;*/

protected:
	//Grid<int> grid_; // the map
	//std::vector<DvcCoord> obstacle_pos_;// obstacles' pos on the map
	//int size_, num_obstacles_;// size of map, number of obstacles
	//DvcCoord start_pos_,goal_pos_;// robot start pos, goal pos
	//double half_efficiency_distance_;

	//Dvc_UncNavigationState* obstacle_state_;//rock good or bad
	//mutable Dvc_MemoryPool<Dvc_UncNavigationState> Dvc_memory_pool_;

	//std::vector<Dvc_UncNavigationState*> Dvc_states_;

protected:
	//DEVICE void InitGeneral();
	/*void Init_4_4();
	void Init_5_5();
	void Init_5_7();
	void Init_7_8();
	void Init_11_11();*/
	//DEVICE void InitStates();
	DEVICE static OBS_TYPE Dvc_GetObservation(double rand_num, const Dvc_UncNavigationState& navstate);
	DEVICE static OBS_TYPE Dvc_GetObservation_parallel(double rand_num,const Dvc_UncNavigationState& nav_state);

	//std::vector<std::vector<std::vector<Dvc_State> > > transition_probabilities_;
	//std::vector<std::vector<double> > alpha_vectors_; // For blind policy
	//mutable std::vector<ValuedAction> mdp_policy_;

public:
	enum OBS_enum{ //Dvc_State flag of cells in the map. FRAGILE: Don't change!
		E_FN_FE_FS_FW = 0,
		E_FN_FE_FS_OW = 1,
		E_FN_FE_OS_FW = 2,
		E_FN_FE_OS_OW = 3,
		E_FN_OE_FS_FW = 4,
		E_FN_OE_FS_OW = 5,
		E_FN_OE_OS_FW = 6,
		E_FN_OE_OS_OW = 7,
		E_ON_FE_FS_FW = 8,
		E_ON_FE_FS_OW = 9,
		E_ON_FE_OS_FW = 10,
		E_ON_FE_OS_OW = 11,
		E_ON_OE_FS_FW = 12,
		E_ON_OE_FS_OW = 13,
		E_ON_OE_OS_FW = 14,
		E_ON_OE_OS_OW = 15,
	};

	/*enum { //Actions of the robot. FRAGILE: Don't change!
		E_STAY = 4,
		E_SOUTH = 0,
		E_EAST = 1,
		E_WEST = 2,
		E_NORTH = 3,
	};*/

	/*enum { //Actions of the robot. FRAGILE: Don't change!
		E_STAY = 4,
		E_SOUTH = 2,
		E_EAST = 1,
		E_WEST = 3,
		E_NORTH = 0,
	};*/

	enum { //Actions of the robot. FRAGILE: Don't change!
		E_STAY = 8,
		E_SOUTH = 2,
		E_EAST = 1,
		E_WEST = 3,
		E_NORTH = 0,
		//"NE", "SE", "SW", "NW"
		E_SOUTH_EAST = 5,
		E_SOUTH_WEST = 6,
		E_NORTH_EAST = 4,
		E_NORTH_WEST = 7,
	};
public:
	//DEVICE Dvc_BaseUncNavigation(std::string map);
	//Dvc_UncNavigation(std::string map);
	DEVICE Dvc_UncNavigation(/*int size, int obstacles*/);
	DEVICE ~Dvc_UncNavigation();
	DEVICE static bool Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
		OBS_TYPE& obs);
	DEVICE int NumActions() const;
	DEVICE static float ObsProb(OBS_TYPE obs, const Dvc_State& state, int action);
	DEVICE virtual bool Step(Dvc_State& state, double rand_num, int action,
		double& reward, OBS_TYPE& obs) const = 0;
	DEVICE static int Dvc_NumObservations();

	DEVICE Dvc_State* Allocate(int state_id, double weight) const;
	DEVICE static Dvc_State* Dvc_Get(Dvc_State* particles, int pos);
	DEVICE static Dvc_State* Dvc_Alloc( int num);
	DEVICE static Dvc_State* Dvc_Copy(const Dvc_State* particle, int pos);
	DEVICE static void Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des=true);
	DEVICE static void Dvc_Free(Dvc_State* particle);

	DEVICE static Dvc_ValuedAction Dvc_GetMinRewardAction() {
		//return Dvc_ValuedAction(E_NORTH, -1);
		return Dvc_ValuedAction(E_STAY, -0.1);
	}
	//DEVICE static void Dvc_FreeList(Dvc_State* particle);


	//const std::vector<Dvc_State>& TransitionProbability(int s, int a) const;
	//DEVICE Dvc_UncNavigationState NextState(Dvc_UncNavigationState& s, int a) const;
	//DEVICE double Reward(Dvc_UncNavigationState& s, int a) const;

	//Dvc_State* CreateStartState(std::string type = "DEFAULT") const;
	//DEVICE Dvc_State* CreateStartState(std::string type = "DEFAULT") const;
	//DEVICE void RandGate(Dvc_UncNavigationState* nav_state) const;
	//DEVICE void RandMap(Dvc_UncNavigationState* nav_state, float ObstacleProb, int skip) const;
	//DEVICE void RandMap(Dvc_UncNavigationState* nav_state) const;

	/*std::vector<Dvc_State*> InitialParticleSet() const;
	std::vector<Dvc_State*> NoisyInitialParticleSet() const;*/
	// Belief* InitialBelief(const Dvc_State* start, std::string type = "PARTICLE") const;

	//DEVICE inline double GetMaxReward() const {
	//	return 10;
	//}
	//DEVICE Dvc_ScenarioUpperBound* CreateScenarioUpperBound(std::string name = "DEFAULT",
	//	std::string particle_bound_name = "DEFAULT") const;
	/*BeliefUpperBound* CreateBeliefUpperBound(std::string name = "DEFAULT") const;
*/

	//DEVICE Dvc_ScenarioLowerBound* CreateScenarioLowerBound(std::string name = "DEFAULT",
	//	std::string particle_bound_name = "DEFAULT") const;
	/*BeliefLowerBound* CreateBeliefLowerBound(std::string name = "DEFAULT") const;

	POMCPPrior* CreatePOMCPPrior(std::string name = "DEFAULT") const;*/

	//DEVICE void AllocBeliefMap(float**& map) const;
	//DEVICE void ClearBeliefMap(float**& map) const;
	//DEVICE void PrintState(const Dvc_State& state, std::ostream& out = std::cout) const;
	//DEVICE void PrintBelief(const Belief& belief, std::ostream& out = std::cout) const;
	//DEVICE void PrintBeliefMap(float** map,std::ostream& out = std::cout) const;
	//DEVICE virtual void PrintObs(const Dvc_State& state, OBS_TYPE observation, std::ostream& out = std::cout) const = 0;
	//DEVICE void PrintAction(int action, std::ostream& out = std::cout) const;


	//DEVICE int NumActiveParticles() const;

	//DEVICE Belief* Tau(const Belief* belief, int action, OBS_TYPE obs) const;
	//void Observe(const Belief* belief, int action, std::map<OBS_TYPE, double>& obss) const;
	//double StepReward(const Belief* belief, int action) const;

	//const Dvc_State* GetState(int index) const;
	//int GetIndex(const Dvc_State* state) const;

	//inline int GetAction(const Dvc_State& navstate) const {
	//	return 0;
	//}

	//int GetRobPosIndex(const Dvc_State* state) const;
	//DEVICE DvcCoord GetRobPos(const Dvc_State* state) const;
	//bool Getobstacle(const Dvc_State* state, int obstacle) const;
	//void Sampleobstacle(Dvc_State* state, int obstacle) const;
	//DEVICE int GetX(const Dvc_UncNavigationState* state) const;
	//DEVICE void IncX(Dvc_UncNavigationState* state) const;
	//DEVICE void DecX(Dvc_UncNavigationState* state) const;
	//DEVICE int GetY(const Dvc_UncNavigationState* state) const;
	//DEVICE void IncY(Dvc_UncNavigationState* state) const;
	//DEVICE void DecY(Dvc_UncNavigationState* state) const;

	//DEVICE void UniformRobPos(Dvc_UncNavigationState* startState) const;
	//DEVICE void AreaRobPos(Dvc_UncNavigationState* startState, int area_size) const;
	//DEVICE void CalObstacles(float prob) const;
	//DEVICE void FreeObstacles() const;
protected:
	//void InitializeTransitions();
	//DvcCoord IndexToCoord(int pos) const;
	//int CoordToIndex(DvcCoord c) const;
	//std::vector<ValuedAction>& ComputeOptimalSamplingPolicy() const;
	//Dvc_UncNavigationState* MajorityUncNavigationState(const std::vector<Dvc_State*>& particles) const;
public:

	//DEVICE void PrintObs(const Dvc_State& state, OBS_TYPE observation,
	//	std::ostream& out = std::cout) const;
	//DEVICE void TestObsProb(const Dvc_State& state) const;

};

} // namespace despot

#endif
