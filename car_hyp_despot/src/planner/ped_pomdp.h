#ifndef PED_POMDP_H
#define PED_POMDP_H


//#include "util/util.h"
#include "disabled_util.h"
#include "despot/interface/pomdp.h"
#include <despot/core/mdp.h>
#include "despot/core/globals.h"
#include "despot/util/coord.h"
#include <despot/core/prior.h>

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
#include <nav_msgs/OccupancyGrid.h>
#include <vector>
#include "despot/core/node.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;

//#include <solver/despot.h>

using namespace std;
using namespace despot;


//#ifndef __CUDACC__

class PedNeuralSolverPrior:public SolverPrior{
	WorldModel& world_model;

	COORD point_to_indices(COORD pos, COORD origin, double resolution, int dim) const;
	void add_in_map(cv::Mat map_tensor, COORD indices, double map_intensity, double map_intensity_scale);
	std::vector<COORD> get_transformed_car(CarStruct car, COORD origin, double resolution);
	void Process_states(const vector<PomdpState*>& hist_states, const vector<int> hist_ids);


	enum UPDATE_MODES { FULL=0, PARTIAL=1 };

	enum {
		VALUE, ACC_PI, ACC_MU, ACC_SIGMA, ANG, VEL_PI, VEL_MU, VEL_SIGMA,
		                   CAR_VALUE_0, RES_IMAGE_0
	};

	struct map_properties{

		double resolution;
		COORD origin;
		int dim;
		double map_intensity;
		int new_dim; // down sampled resolution

		double map_intensity_scale;
		double downsample_ratio;
	};

private:
	at::Tensor empty_map_tensor_;
	at::Tensor map_tensor_;
	std::vector<at::Tensor> map_hist_tensor_;
	std::vector<at::Tensor> car_hist_tensor_;
	at::Tensor goal_tensor;

	cv::Mat map_image_;
	std::vector<cv::Mat> map_hist_images_;
	std::vector<cv::Mat> car_hist_images_;
	cv::Mat goal_image_;

	vector<cv::Point3f> car_shape;
public:

	PedNeuralSolverPrior(const DSPOMDP* model, WorldModel& world);
//	virtual const std::vector<double>& ComputePreference();
//
//	virtual double ComputeValue();

	void Compute(vector<torch::Tensor>& images, map<OBS_TYPE, despot::VNode*>& vnode);

	void Process_history(int);
	std::vector<torch::Tensor> Process_history_input();

	std::vector<torch::Tensor> Process_nodes_input(const std::vector<State*>& vnode_states);
//	at::Tensor Combine_images(const at::Tensor& node_image, const at::Tensor& hist_images){return torch::zeros({1,1,1});}
	torch::Tensor Combine_images();

public:
	int num_hist_channels;
	int num_peds_in_NN;
	nav_msgs::OccupancyGrid raw_map_;

	map_properties map_prop_;
	std::shared_ptr<torch::jit::script::Module> drive_net;

};
//#endif

class PedPomdp : public DSPOMDP {
public:

	PedPomdp();
	PedPomdp(WorldModel &);
	void UpdateVel(int& vel, int action, Random& random) const;
	void RobStep(int &robY,int &rob_vel, int action, Random& random) const;
	void PedStep(PomdpState& state, Random& random) const;

	bool Step(State& state_, double rNum, int action, double& reward, uint64_t& obs) const;
	bool Step(PomdpStateWorld& state, double rNum, int action, double& reward, uint64_t& obs) const;

	bool ImportanceSamplingStep(State& state_, double rNum, int action, double& reward, uint64_t& obs) const;
    std::vector<double> ImportanceWeight(std::vector<State*> particles, ACT_TYPE last_action) const;
    double ImportanceScore(PomdpState* state, ACT_TYPE last_action) const;

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

	inline int NumActions() const { return (int)(2*ModelParams::NumAcc+1); }

	PomdpState* GreateStartState(string type) const;

	std::vector<std::vector<double>> GetBeliefVector(const std::vector<State*> particles) const;
	Belief* InitialBelief(const State* start, string type) const;

	ValuedAction GetBestAction() const;

	double GetMaxReward() const;

	ParticleUpperBound* CreateParticleUpperBound(string name = "DEFAULT") const;
	ScenarioUpperBound* CreateScenarioUpperBound(string name = "DEFAULT",
		string particle_bound_name = "DEFAULT") const;

	ScenarioLowerBound* CreateScenarioLowerBound(string name = "DEFAULT",
		string particle_bound_name = "DEFAULT") const;

	void Statistics(const std::vector<PomdpState*> particles) const;

	void PrintState(const State& state, ostream& out = cout) const;
	void PrintWorldState(const PomdpStateWorld& state, ostream& out = cout) const;
	void PrintObs(const State & state, uint64_t obs, ostream& out = cout) const;
	void PrintAction(int action, ostream& out = cout) const;
	void PrintBelief(const Belief& belief, ostream& out = cout) const;

	State* Allocate(int state_id, double weight) const;
	State* Copy(const State* particle) const;
	void Free(State* particle) const;

	std::vector<State*> ConstructParticles(std::vector<PomdpState> & samples) const;
	int NumActiveParticles() const;
	void PrintParticles(const std::vector<State*> particles, ostream& out) const;


	int NumObservations() const;
	int ParallelismInStep() const;
	void ExportState(const State& state, std::ostream& out = std::cout) const;
	State* ImportState(std::istream& in) const;
	void ImportStateList(std::vector<State*>& particles, std::istream& in) const;


	/* HyP-DESPOT GPU model */
	Dvc_State* AllocGPUParticles(int numParticles, MEMORY_MODE mode,  Dvc_State*** particles_all_a = NULL ) const;

	void DeleteGPUParticles( MEMORY_MODE mode, Dvc_State** particles_all_a = NULL) const;

	void CopyParticleIDsToGPU(int* dvc_IDs, const std::vector<int>& particleIDs, void* CUDAstream=NULL) const;

	Dvc_State* CopyParticlesToGPU(Dvc_State* dvc_particles, const std::vector<State*>& particles , bool deep_copy) const;

	void ReadParticlesBackToCPU(std::vector<State*>& particles ,const Dvc_State* parent_particles,
				bool deepcopy) const;

	void CopyGPUParticlesFromParent(Dvc_State* des,Dvc_State* src,int src_offset,int* IDs,
		int num_particles,bool interleave,
		Dvc_RandomStreams* streams, int stream_pos,
			void* CUDAstream=NULL, int shift=0) const;

	void CreateMemoryPool() const;
	
	void DestroyMemoryPool(MEMORY_MODE mode) const;

	void InitGPUModel();
	void InitGPUUpperBound(string name,	string particle_bound_name) const;
	void InitGPULowerBound(string name,	string particle_bound_name) const ;

	void DeleteGPUModel();
	void DeleteGPUUpperBound(string name, string particle_bound_name);
	void DeleteGPULowerBound(string name, string particle_bound_name);

	virtual OBS_TYPE StateToIndex(const State*) const;

	/* end HyP-DESPOT GPU model */


	SolverPrior* CreateSolverPrior(World* world, std::string name, bool update_prior = true) const;

	State* CopyForSearch(const State* particle) const;

	WorldModel *world_model;


	bool use_rvo_in_search;
	bool use_rvo_in_simulation;
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

	void InitRVOSetting();


public:
	// Let's drive

	double GetAcceleration(ACT_TYPE action, bool debug=false) const;
	double GetSteering(ACT_TYPE action, bool debug=false) const;
	static ACT_TYPE GetActionID(double steering, double acc, bool debug=false);
	static double GetAccfromAccID(int);
};


#endif

