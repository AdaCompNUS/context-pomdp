#ifndef PED_POMDP_H
#define PED_POMDP_H


//#include "util/util.h"
#include "disabled_util.h"
#include "core/pomdp.h"
#include "core/globals.h"
#include "util/coord.h"
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
// #include "SFM.h"

/*
struct PathNode { // TODO
	vector<PathNode> children;
	int x,y;
};
*/

class PedPomdp : public DSPOMDP {
public:
	PedPomdp(WorldModel &);
	void UpdateVel(int& vel, int action, Random& random) const;
	void RobStep(int &robY,int &rob_vel, int action, Random& random) const;
	void PedStep(PomdpState& state, Random& random) const;

	bool Step(State& state_, double rNum, int action, double& reward, uint64_t& obs) const;
	bool Step(PomdpStateWorld& state, double rNum, int action, double& reward, uint64_t& obs) const;

	bool ImportanceSamplingStep(State& state_, double rNum, int action, double& reward, uint64_t& obs) const;
    vector<double> ImportanceWeight(vector<State*> particles) const;
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
	const vector<int>& ObserveVector(const State& )   const;
	double ObsProb(uint64_t z, const State& s, int action) const;

	inline int NumActions() const { return 3; }

	PomdpState* GreateStartState(string type) const;

	vector<vector<double>> GetBeliefVector(const vector<State*> particles) const;
	Belief* InitialBelief(const State* start, string type) const;

	ValuedAction GetMinRewardAction() const;
	// void InitializeScenarioLowerBound(string name, RandomStreams& streams);

	double GetMaxReward() const;
	// void InitializeParticleUpperBound(string name, RandomStreams& streams);
	// void InitializeScenarioUpperBound(string name, RandomStreams& streams);

	ParticleUpperBound* CreateParticleUpperBound(string name = "DEFAULT") const;
	ScenarioUpperBound* CreateScenarioUpperBound(string name = "DEFAULT",
		string particle_bound_name = "DEFAULT") const;

	// ParticleLowerBound* CreateParticleLowerBound(string name = "DEFAULT") const;
	ScenarioLowerBound* CreateScenarioLowerBound(string name = "DEFAULT",
		string particle_bound_name = "DEFAULT") const;

	void Statistics(const vector<PomdpState*> particles) const;

	void PrintState(const State& state, ostream& out = cout) const;
	void PrintWorldState(PomdpStateWorld state, ostream& out = cout);

	void PrintObs(const State & state, uint64_t obs, ostream& out = cout) const;
	void PrintAction(int action, ostream& out = cout) const;
	void PrintBelief(const Belief& belief, ostream& out = cout) const;

	State* Allocate(int state_id, double weight) const;
	State* Copy(const State* particle) const;
	void Free(State* particle) const;

	vector<State*> ConstructParticles(vector<PomdpState> & samples); 
	int NumActiveParticles() const;
	void PrintParticles(const vector<State*> particles, ostream& out) const;

	WorldModel &world;

	bool use_rvo;

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

