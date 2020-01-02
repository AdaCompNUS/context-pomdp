#pragma once
#include "state.h"
#include "path.h"
#include <RVO.h>
#include "debug_util.h"
#include <unordered_map>
#include <msg_builder/LaneSeg.h>
#include <geometry_msgs/Polygon.h>
#include "despot/core/prior.h"

using namespace despot;

struct AgentBelief {
	int id;
	COORD pos;
	double speed;
	COORD vel;
	double heading_dir;
	AgentType type;
	bool reset;
	bool cross_dir;

	std::vector<Path> goal_paths;
	std::vector<std::vector<double>> prob_modes_goals;

	std::vector<COORD> bb;

	int SampleGoal() const;
	int MaxLikelyIntention() const;
	void SampleGoalMode(int& goal, int& mode, int use_att_mode = 0) const;
	void ResetBelief(int new_size);
};

class WorldModel {
public:

	AgentParams default_car_;
	AgentParams default_bike_;
	AgentParams default_ped_;

	WorldModel();
	~WorldModel();

	bool MovingAway(const PomdpState& state, int ped);
	void GetClosestPed(const PomdpState& state, int& closest_front_ped,
			double& closest_front_dist, int& closest_side_ped,
			double& closest_side_dist);
	double GetMinCarPedDist(const PomdpState& state);
	double GetMinCarPedDistAllDirs(const PomdpState& state);
	int DefaultPolicy(const std::vector<State*>& particles);
	ACT_TYPE DefaultStatePolicy(const State* _state) const;
	int DefaultGraphEdge(const State* particle, bool print_debug);

	bool InFront(const COORD ped_pos, const CarStruct car,
			double in_front_angle_deg = -1) const;

	bool InCollision(const PomdpState& state);
	bool InCollision(const PomdpStateWorld& state);
	bool InRealCollision(const PomdpStateWorld& state,
			double in_front_angle_deg = -1);

	bool InCollision(const PomdpState& state, int &id);
	bool InCollision(const PomdpStateWorld& state, int &id);
	bool InRealCollision(const PomdpStateWorld& state, int &id,
			double in_front_angle_deg = -1);

	bool IsGlobalGoal(const CarStruct& car);
	int MinStepToGoal(const PomdpState& state);

	void AgentStep(AgentStruct &ped, Random& random);
	void AgentStep(AgentStruct &ped, double& random);
	double ISAgentStep(CarStruct &car, AgentStruct &ped, Random& random); //importance sampling AgentStep
	void GammaAgentStep(AgentStruct peds[], Random& random, int num_ped); //no interaction between car and pedestrian
	void GammaAgentStep(AgentStruct peds[], Random& random, int num_ped,
			CarStruct car); //pedestrian also need to consider car when moving
	void GammaAgentStep(AgentStruct peds[], double& random, int num_ped); //no interaction between car and pedestrian
	void GammaAgentStep(AgentStruct peds[], double& random, int num_ped,
			CarStruct car); //pedestrian also need to consider car when moving
	void GammaAgentStep(PomdpStateWorld& state, Random& random);
	void AgentStepCurVel(AgentStruct& ped, int step = 1, double noise = 0.0);
	void AgentStepPath(AgentStruct& agent, int step = 1, double noise = 0.0,
			bool doPrint = false);
	void VehStepPath(AgentStruct& agent, int step = 1, double noise = 0.0,
			bool doPrint = false);
	void PedStepPath(AgentStruct& agent, int step = 1, double noise = 0.0,
			bool doPrint = false);

	COORD GetGoalPos(const AgentStruct& ped, int intention_id = -1);
	COORD GetGoalPos(const AgentBelief& ped, int intention_id);
	COORD GetGoalPos(const Agent& ped, int intention_id);
	COORD GetGoalPosFromPaths(int agent_id, int intention_id,
			int pos_along_path, const COORD& agent_pos, const COORD& agent_vel,
			AgentType type, bool agent_cross_dir);

	int GetNumIntentions(const AgentStruct& ped);
	int GetNumIntentions(const Agent& ped);
	int GetNumIntentions(int ped_id);

	void FixGPUVel(CarStruct &car);
	void RobStep(CarStruct &car, double steering, Random& random);
	void RobStep(CarStruct &car, double steering, double& random);
	void RobStep(CarStruct &car, double& random, double acc, double steering);
	void RobStep(CarStruct &car, Random& random, double acc, double steering);
	void RobVelStep(CarStruct &car, double acc, Random& random);
	void RobVelStep(CarStruct &car, double acc, double& random);
	double ISRobVelStep(CarStruct &car, double acc, Random& random); //importance sampling RobvelStep

	void RobStepCurVel(CarStruct &car);
	void RobStepCurAction(CarStruct &car, double acc, double steering);

	double AgentMoveProb(const AgentBelief& prev_agent, const Agent& curr,
			int goal_id, int ped_mode);

	bool IsStopIntention(int intention, int agent_id);
	bool IsCurVelIntention(int intention, int agent_id);

	void SetPath(Path path);
	void UpdatePedBelief(AgentBelief& b, const Agent& curr_ped);
	AgentBelief InitPedBelief(const Agent& ped);

	inline int GetThreadID() {
		return Globals::MapThread(this_thread::get_id());
	}
	void InitRVO();

	void CalBBExtents(AgentBelief& bb, AgentStruct& agent);
	void CalBBExtents(AgentStruct& agent, std::vector<COORD>& bb,
			double heading_dir);
	void CalBBExtents(COORD pos, double heading_dir, vector<COORD>& bb,
			double& extent_x, double& extent_y);

	std::vector<RVO::Vector2> GetBoundingBoxCorners(AgentStruct& agent);
	std::vector<RVO::Vector2> GetBoundingBoxCorners(RVO::Vector2 forward_vec,
			RVO::Vector2 sideward_vec, RVO::Vector2 pos, double forward_len,
			double side_len);

	Path path;
	double freq;
	const double in_front_angle_cos;
	std::vector<RVO::RVOSimulator*> traffic_agent_sim_;

	const std::string goal_mode = "path"; // "cur_vel", "path"

	void RasieUnsupportedGoalMode(std::string function) {
		std::cout << function << ": unsupported goal mode " << goal_mode
				<< std::endl;
		raise (SIGABRT);
	}

	std::unordered_map<int, int> id_map_num_paths;
	std::unordered_map<int, std::vector<Path>> id_map_paths;

public:

	int NumPaths(int agent_id) {
		auto it = id_map_num_paths.find(agent_id);
		if (it == id_map_num_paths.end()) {
			return 0;
		}
		return it->second;
	}

	std::vector<Path>& PathCandidates(int agent_id) {
		auto it = id_map_paths.find(agent_id);
		if (it == id_map_paths.end()) {
			std::cout << __FILE__ << ": agent id " << agent_id
					<< " not found in id_map_paths" << std::endl;
			PrintPathMap();
			std::vector<Path> ps;
			id_map_paths[agent_id] = ps;
			it = id_map_paths.find(agent_id);
		}
		return it->second;
	}

	void PrintPathMap() {
		cout << "id_map_paths: ";
		for (auto it = id_map_paths.begin(); it != id_map_paths.end(); ++it) {
			cout << " (" << it->first << ", l_" << it->second.size() << ")";
		}
		cout << endl;
	}
	void ValidateIntention(int agent_id, int intention_id, const char*, int);

public:
	std::string goal_file_name_;
	bool CheckCarWithObsLine(const CarStruct& car, COORD obs_last_point,
			COORD obs_first_point, int flag); // 0 in search, 1 real check
	bool CheckCarWithVehicle(const CarStruct& car, const AgentStruct& veh,
			int flag);
	bool CheckCarWithVehicleReal(const CarStruct& car, const AgentStruct& veh,
			int flag);

public:
	/// lets drive
	std::map<int, vector<COORD>> ped_mean_dirs;
	COORD DistractedPedMeanDir(AgentStruct& ped, int goal_id);
	COORD AttentivePedMeanDir(int ped_id, int goal_id);

	void AddEgoGammaAgent(int num_peds, CarStruct& car);
	void AddGammaAgent(AgentBelief& b, int id_in_sim);

	void PrepareAttentiveAgentMeanDirs(std::map<int, AgentBelief> peds,
			CarStruct& car);
	void PrintMeanDirs(std::map<int, AgentBelief> old_peds,
			map<int, const Agent*>& curr_peds);

	void EnsureMeanDirExist(int ped_id);

public:
	template<typename T>
	double GetSteerToPath(const T& state) const {
		COORD car_goal = path[path.Forward(path.Nearest(state.car.pos), 5.0)];
		return PControlAngle<CarStruct>(state.car, car_goal);
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
			guide_steer = 0; //ahead
		return guide_steer;
	}

	double GetPrefSpeed(const Agent& agent);

	// for ego-car
	void BicycleModel(CarStruct &car, double steering, double end_vel);
	double PurepursuitAngle(const CarStruct& car, COORD& pursuit_point) const;

	// for exo-cars
	void BicycleModel(AgentStruct &car, double steering, double end_vel);
	double PurepursuitAngle(const AgentStruct& car, COORD& pursuit_point) const;

public:

	void GammaSimulateAgents(AgentStruct agents[], int num_agents,
			CarStruct& car);
	COORD GetGammaVel(AgentStruct& agent, int i);

	void AgentApplyGammaVel(AgentStruct& agent, COORD& rvo_vel);

};

class WorldStateTracker {
public:
	typedef pair<float, Agent*> AgentDistPair;

	WorldStateTracker(WorldModel& _model) :
			model(_model) {
		car_heading_dir = 0;
		carvel = 0;
		car_odom_heading = -10;
		detect_time = false;
		latest_path_time_stamp = -1;
		latest_time_stamp = -1;
	}

	void UpdatePedState(const Pedestrian& ped, bool doPrint = false);
	void UpdateVehState(const Vehicle& veh, bool doPrint = false); // this is for exo-vehicles
	void UpdatePedPaths(const Pedestrian& ped, bool doPrint = false);
	void UpdateVehPaths(const Vehicle& veh, bool doPrint = false); // this is for exo-vehicles

	void TrackVel(Agent& des, const Agent& src, bool&, bool);
	void TracPos(Agent& des, const Agent& src, bool);
	void TracIntention(Agent& des, const Agent& src, bool);
	void TracBoundingBox(Vehicle& des, const Vehicle& src, bool);
	void TracCrossDirs(Pedestrian& des, const Pedestrian& src, bool);
	void UpdatePathPool(Agent& des);

	void UpdateCar(const CarStruct car);
	void UpdateVel(double vel);

	void CleanPed();
	void CleanVeh();
	void CleanAgents();

	void RemoveAgents();

	template<typename T>
	bool AgentIsAlive(T& agent, vector<T>& agent_list_new) {
		bool alive = true;
		if (detect_time) {
			if (latest_time_stamp - agent.time_stamp > 1.0) { // agent disappeared for 1 second
				DEBUG(
						string_sprintf(
								"agent %d disappeared for too long (>1.0s).",
								agent.id));
				alive = false;
			}
		}

		return alive;
	}

	template<typename T>
	bool AgentIsUp2Date(T& agent) {
		if (detect_time) {
			if (Globals::ElapsedTime() - agent.time_stamp > 1.0) { // agent disappeared for 1 second
				DEBUG(
						string_sprintf(
								"agent %d disappeared for too long (>1.0s).",
								agent.id));
				return false;
			} else
				return true;
		} else
			return true;
	}

	void ValidateCar(const char* func);

	void SetPomdpCar(CarStruct& car);
	void SetPomdpPed(AgentStruct& agent, const Agent& src);

	bool Emergency();

	void CheckWorldStatus();

	std::vector<AgentDistPair> GetSortedAgents(bool doPrint = false);

	PomdpState GetPomdpState();
	PomdpStateWorld GetPomdpWorldState();

	void Text(const vector<AgentDistPair>& sorted_agents) const;
	void Text(const vector<Pedestrian>& tracked_peds) const;
	void Text(const vector<Vehicle>& tracked_vehs) const;

	// Car state
	COORD carpos;
	double carvel;
	double car_heading_dir;
	double car_odom_heading;

	//Ped states
	std::vector<Pedestrian> ped_list;
	std::vector<Vehicle> veh_list;

	double latest_time_stamp;
	double latest_path_time_stamp;
	bool detect_time;

	WorldModel& model;
};

class WorldBeliefTracker {
public:
	WorldBeliefTracker(WorldModel& model, WorldStateTracker& _stateTracker) :
			model(model), stateTracker(_stateTracker), cur_time_stamp(0), cur_acc(
					-1), cur_steering(0) {
	}

	void Update();
	void UpdateCar();
	PomdpState Sample(bool predict = false, int use_att_mode = 0);
	vector<PomdpState> Sample(int num, bool predict = false, int use_att_mode =
			0);
	vector<AgentStruct> PredictAgents();
	PomdpState PredictPedsCurVel(PomdpState*, double acc, double steering);

	WorldModel& model;
	WorldStateTracker& stateTracker;
	CarStruct car;
	std::map<int, AgentBelief> agent_beliefs;
	std::vector<AgentBelief*> sorted_beliefs;

	void PrintState(const State& s, ostream& out = cout) const;
	void printBelief() const;

	PomdpState Text() const;
	void Text(const std::map<int, AgentBelief>&) const;

	void ValidateCar(const char* func);

public:
	double cur_time_stamp;

	double cur_acc;
	double cur_steering;
};

enum PED_MODES {
	AGENT_ATT = 0, AGENT_DIS = 1, NUM_AGENT_TYPES = 2,
};

int ClosestInt(double v);
int FloorIntRobust(double v);
