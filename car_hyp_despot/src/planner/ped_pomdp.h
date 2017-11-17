#ifndef PED_POMDP_H
#define PED_POMDP_H


//#include "util/util.h"
#include "disabled_util.h"
#include "despot/core/pomdp.h"
#include <despot/core/mdp.h>
#include "despot/core/globals.h"
#include "despot/util/coord.h"
//#include "lower_bound.h"
//#include "upper_bound.h"
//#include "string.h"

#include "param.h"
#include "state.h"
#include "WorldModel.h"
#include <cmath>
#include <utility>
#include <string>
#include "math_utils.h"

using namespace std;
using namespace despot;

class PedPomdp : public DSPOMDP {
public:
	PedPomdp(WorldModel &);
	void UpdateVel(int& vel, int action, Random& random) const;
	void RobStep(int &robY,int &rob_vel, int action, Random& random) const;
	void PedStep(PomdpState& state, Random& random) const;

	bool Step(State& state_, double rNum, int action, double& reward, uint64_t& obs) const;
	bool Step(PomdpStateWorld& state, double rNum, int action, double& reward, uint64_t& obs) const;

	bool ImportanceSamplingStep(State& state_, double rNum, int action, double& reward, uint64_t& obs) const;
    std::vector<double> ImportanceWeight(std::vector<State*> particles) const;
    double ImportanceScore(PomdpState* state) const;

    State* CreateStartState(string type = "DEFAULT") const {
		return 0;	
	}
	double TransitionProbability(const PomdpState& curr, const PomdpState& next, int action) const;

    double CrashPenalty(const PomdpState& state) const; //, int closest_ped, double closest_dist) const;
	double CrashPenalty(const PomdpStateWorld& state) const; //, int closest_ped, double closest_dist) const;

    double ActionPenalty(int action) const;

    double MovementPenalty(const PomdpState& state) const;
    double MovementPenalty(const PomdpStateWorld& state) const;

	uint64_t Observe(const State& ) const;
	const std::vector<int>& ObserveVector(const State& )   const;
	double ObsProb(uint64_t z, const State& s, int action) const;

	inline int NumActions() const { return 3; }

	PomdpState* GreateStartState(string type) const;

	std::vector<std::vector<double>> GetBeliefVector(const std::vector<State*> particles) const;
	Belief* InitialBelief(const State* start, string type) const;

	ValuedAction GetMinRewardAction() const;

	double GetMaxReward() const;

	ParticleUpperBound* CreateParticleUpperBound(string name = "DEFAULT") const;
	ScenarioUpperBound* CreateScenarioUpperBound(string name = "DEFAULT",
		string particle_bound_name = "DEFAULT") const;

	ScenarioLowerBound* CreateScenarioLowerBound(string name = "DEFAULT",
		string particle_bound_name = "DEFAULT") const;

	void Statistics(const std::vector<PomdpState*> particles) const;

	void PrintState(const State& state, ostream& out = cout) const;
	void PrintWorldState(PomdpStateWorld state, ostream& out = cout);
	void PrintObs(const State & state, uint64_t obs, ostream& out = cout) const;
	void PrintAction(int action, ostream& out = cout) const;
	void PrintBelief(const Belief& belief, ostream& out = cout) const;

	State* Allocate(int state_id, double weight) const;
	State* Copy(const State* particle) const;
	void Free(State* particle) const;

	std::vector<State*> ConstructParticles(std::vector<PomdpState> & samples);
	int NumActiveParticles() const;
	void PrintParticles(const std::vector<State*> particles, ostream& out) const;


	int NumObservations() const;
	int ParallelismInStep() const;
	void ExportState(const State& state, std::ostream& out = std::cout) const;
	State* ImportState(std::istream& in) const;
	void ImportStateList(std::vector<State*>& particles, std::istream& in) const;

	Dvc_State* AllocGPUParticles(int numParticles,int alloc_mode) const;

	void CopyGPUParticles(Dvc_State* des,Dvc_State* src,int src_offset,int* IDs,
			int num_particles,bool interleave,
			Dvc_RandomStreams* streams, int stream_pos,
			void* CUDAstream=NULL, int shift=0) const;

	void CopyGPUWeight1(void* cudaStream=NULL, int shift=0) const;
	float CopyGPUWeight2(void* cudaStream=NULL, int shift=0) const;
	Dvc_State* GetGPUParticles() const;

	void CopyToGPU(const std::vector<int>& particleIDs,int* Dvc_ptr, void* CUDAstream=NULL) const;
	Dvc_State* CopyToGPU(const std::vector<State*>& particles , bool copy_cells) const;
	void ReadBackToCPU(std::vector<State*>& particles ,const Dvc_State* parent_particles,
				bool copycells) const;

	void DeleteGPUParticles( int num_particles) const;
	void DeleteGPUParticles(Dvc_State* particles, int num_particles) const;
	void CreateMemoryPool(int chunk_size) const;
	void DestroyMemoryPool(int mode) const;


	WorldModel &world;

 // protected:
	enum {
		ACT_CUR,
		ACT_ACC,
		ACT_DEC
	};
private:
	int** map;
	PomdpState startState;
	mutable MemoryPool<PomdpState> memory_pool_;
	mutable Random random_;
};
#endif

