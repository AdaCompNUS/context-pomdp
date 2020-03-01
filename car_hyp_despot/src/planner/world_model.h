#pragma once
#include "state.h"
#include "path.h"
#include <RVO.h>
#include "utils.h"
#include <unordered_map>
#include <msg_builder/LaneSeg.h>
#include <geometry_msgs/Polygon.h>
#include "despot/core/prior.h"

using namespace despot;


class WorldModel {
public:

	WorldModel();
	~WorldModel();

	double freq;
	Path path;
	void SetPath(Path path);

public:
	/// Default policy
	const double in_front_angle_cos;

	int DefaultPolicy(const std::vector<State*>& particles);
	ACT_TYPE DefaultStatePolicy(const State* _state) const;

	bool InFront(const COORD ped_pos, const CarStruct car,
			double in_front_angle_deg = -1) const;
	int MinStepToGoal(const PomdpState& state);

public:
	// step function elements
	void AgentStep(AgentStruct &ped, Random& random);
	void AgentStep(AgentStruct &ped, double& random);

	void GammaAgentStep(AgentStruct peds[], double& random, int num_ped,
			CarStruct car); //pedestrian also need to consider car when moving
	void GammaAgentStep(AgentStruct& agent, int intention_id);
	void AgentStepCurVel(AgentStruct& ped, int step = 1, double noise = 0.0);
	void AgentStepPath(AgentStruct& agent, int step = 1, double noise = 0.0,
			bool doPrint = false);
	void VehStepPath(AgentStruct& agent, int step = 1, double noise = 0.0,
			bool doPrint = false);
	void PedStepPath(AgentStruct& agent, int step = 1, double noise = 0.0,
			bool doPrint = false);

	void RobStep(CarStruct &car, double steering, Random& random);
	void RobStep(CarStruct &car, double steering, double& random);
	void RobStep(CarStruct &car, double& random, double acc, double steering);
	void RobStep(CarStruct &car, Random& random, double acc, double steering);
	void RobVelStep(CarStruct &car, double acc, Random& random);
	void RobVelStep(CarStruct &car, double acc, double& random);
	double ISRobVelStep(CarStruct &car, double acc, Random& random); //importance sampling RobvelStep

	void RobStepCurVel(CarStruct &car);
	void RobStepCurAction(CarStruct &car, double acc, double steering);

public:
	/// Intention-related
	const std::string goal_mode = "path"; // "cur_vel", "path"
	std::unordered_map<int, int> id_map_num_paths;
	std::unordered_map<int, std::vector<Path>> id_map_paths;
	std::unordered_map<int, bool> id_map_belief_reset;

	COORD GetGoalPos(const AgentStruct& ped, int intention_id = -1);
	COORD GetGoalPosFromPaths(int agent_id, int intention_id,
			int pos_along_path, const COORD& agent_pos, const COORD& agent_vel,
			AgentType type, bool agent_cross_dir);
	void RasieUnsupportedGoalMode(std::string function) {
		std::cout << function << ": unsupported goal mode " << goal_mode
				<< std::endl;
		raise (SIGABRT);
	}

	int NumPaths(int agent_id);
	std::vector<Path>& PathCandidates(int agent_id);
	void PrintPathMap();

	int GetNumIntentions(int agent_id);
	void ValidateIntention(int agent_id, int intention_id, const char*, int);
	bool IsStopIntention(int intention, int agent_id);
	bool IsCurVelIntention(int intention, int agent_id);

	bool NeedBeliefReset(int agent_id);

public:
	// Termination-related
	bool IsGlobalGoal(const CarStruct& car) const;

	bool InCollision(const PomdpState& state) const;
	bool InCollision(const PomdpState& state, int &id) const;
//	bool InCollision(const PomdpStateWorld& state) const;
	bool InCollision(const PomdpStateWorld& state, double in_front_angle_deg = -1) const;

	bool InCollision(const PomdpStateWorld& state, int &id) const;
	bool InRealCollision(const PomdpStateWorld& state,
			double in_front_angle_deg = -1) const;
	bool InRealCollision(const PomdpStateWorld& state, int &id,
			double in_front_angle_deg = -1) const;

	bool CheckCarWithObsLine(const CarStruct& car, COORD obs_last_point,
			COORD obs_first_point, int flag) const; // 0 in search, 1 real check
	bool CheckCarWithVehicle(const CarStruct& car, const AgentStruct& veh,
			int flag) const;
	bool CheckCarWithVehicleReal(const CarStruct& car, const AgentStruct& veh,
			int flag) const;

public:
	/// Dynamics
	double GetSteerToPath(const CarStruct& car) const {
		COORD car_goal = path[path.Forward(path.Nearest(car.pos), 5.0)];
		return PurepursuitAngle(car, car_goal);
	}

	template<typename T>
	double PControlAngle(const T& car, COORD& car_goal) const {
		COORD dir = car_goal - car.pos;
		double theta = atan2(dir.y, dir.x);

		if (theta < 0)
			theta += 2 * M_PI;

		theta = theta - car.heading_dir;

		while (theta > 2 * M_PI)
			theta -= 2 * M_PI;
		while (theta < 0)
			theta += 2 * M_PI;

		float guide_steer = 0;
		if (theta < M_PI)
			guide_steer = min(theta, ModelParams::MAX_STEER_ANGLE);
		else if (theta > M_PI)
			guide_steer = max((theta - 2 * M_PI), -ModelParams::MAX_STEER_ANGLE);
		else
			guide_steer = 0;
		return guide_steer;
	}

	// for ego-car
	void BicycleModel(CarStruct &car, double steering, double end_vel);
	double PurepursuitAngle(const CarStruct& car, COORD& pursuit_point) const;

	// for exo-cars
	void BicycleModel(AgentStruct &car, double steering, double end_vel);
	double PurepursuitAngle(const AgentStruct& car, COORD& pursuit_point) const;

public:
	// gamma related
	AgentParams default_car_;
	AgentParams default_bike_;
	AgentParams default_ped_;
	std::vector<RVO::RVOSimulator*> traffic_agent_sim_;
	std::map<int, vector<COORD>> ped_mean_dirs;

	inline int GetThreadID() {
		return Globals::MapThread(this_thread::get_id());
	}

	std::vector<RVO::Vector2> GetBoundingBoxCorners(AgentStruct& agent);
	std::vector<RVO::Vector2> GetBoundingBoxCorners(RVO::Vector2 forward_vec,
			RVO::Vector2 sideward_vec, RVO::Vector2 pos, double forward_len,
			double side_len);

	void InitGamma();
	void AddEgoGammaAgent(int num_peds, const CarStruct& car);
	void AddGammaAgent(const AgentStruct& agent, int id_in_sim);

	// to be used in step function
	void GammaSimulateAgents(AgentStruct agents[], int num_agents,
			CarStruct& car);
	COORD GetGammaVel(AgentStruct& agent, int i);
	void AgentApplyGammaVel(AgentStruct& agent, COORD& rvo_vel);

	// to be used in belief tracking
	void PrepareAttentiveAgentMeanDirs(const State* state);
	void EnsureMeanDirExist(int agent_id, int intention_id);

};

enum PED_MODES {
	AGENT_ATT = 0, AGENT_DIS = 1, NUM_AGENT_TYPES = 2,
};

int ClosestInt(double v);
int FloorIntRobust(double v);

