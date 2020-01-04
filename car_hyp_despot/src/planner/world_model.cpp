#include<limits>
#include<cmath>
#include<cstdlib>
#include <numeric>

#include <despot/GPUcore/thread_globals.h>
#include <despot/core/globals.h>
#include <despot/solver/despot.h>

#include"math_utils.h"
#include"coord.h"
#include "path.h"
#include "world_model.h"
#include "ped_pomdp.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

using namespace std;
std::chrono::time_point<std::chrono::system_clock> init_time;

bool use_noise_in_rvo = false;
bool use_vel_from_summit = true;
int include_stop_intention = 0;
int include_curvel_intention = 1;
bool initial_update_step = true;
double PURSUIT_LEN = 3.0;


int ClosestInt(double v) {
	if ((v - (int) v) < 0.5)
		return (int) v;
	else
		return (int) v + 1;
}

int FloorIntRobust(double v) {
	if ((v - (int) v) > 1 - 1e-5)
		return (int) v + 1;
	else
		return (int) v;
}

WorldModel::WorldModel() :
		freq(ModelParams::CONTROL_FREQ), in_front_angle_cos(
				cos(ModelParams::IN_FRONT_ANGLE_DEG / 180.0 * M_PI)) {
	goal_file_name_ = "null";

	if (DESPOT::Debug_mode)
		ModelParams::NOISE_GOAL_ANGLE = 0.000001;

	init_time = Time::now();
}

void WorldModel::InitRVO() {
	if (!Globals::config.use_multi_thread_)
		Globals::config.NUM_THREADS = 1;

	int NumThreads = Globals::config.NUM_THREADS;

	traffic_agent_sim_.resize(NumThreads);
	for (int tid = 0; tid < NumThreads; tid++) {
		traffic_agent_sim_[tid] = new RVO::RVOSimulator();
		traffic_agent_sim_[tid]->setTimeStep(1.0f / ModelParams::CONTROL_FREQ);
		traffic_agent_sim_[tid]->setAgentDefaults(5.0f, 5, 1.5f, 1.5f, PED_SIZE,
				2.5f);
	}

	default_car_ = AgentParams::getDefaultAgentParam("Car");
	default_bike_ = AgentParams::getDefaultAgentParam("Bike");
	default_ped_ = AgentParams::getDefaultAgentParam("People");

}

WorldModel::~WorldModel() {
	traffic_agent_sim_.clear();
	traffic_agent_sim_.resize(0);
}

int WorldModel::DefaultPolicy(const std::vector<State*>& particles) {
	const PomdpState *state = static_cast<const PomdpState*>(particles[0]);
	return DefaultStatePolicy(state);
}

double CalculateGoalDir(const CarStruct& car, const COORD& goal) {
	COORD dir = goal - car.pos;
	double theta = atan2(dir.y, dir.x);
	if (theta < 0)
		theta += 2 * M_PI;

	theta = theta - car.heading_dir;
	theta = CapAngle(theta);
	if (theta < M_PI)
		return 1; //CCW
	else if (theta > M_PI)
		return -1; //CW
	else
		return 0; //ahead
}

ACT_TYPE WorldModel::DefaultStatePolicy(const State* _state) const {
	logv << __FUNCTION__ << "] Get state info" << endl;
	const PomdpState *state = static_cast<const PomdpState*>(_state);
	double mindist_along = numeric_limits<double>::infinity();
	double mindist_tang = numeric_limits<double>::infinity();
	const COORD& carpos = state->car.pos;
	double carheading = state->car.heading_dir;
	double carvel = state->car.vel;

	logv << __FUNCTION__ << "] Calculate steering" << endl;

	double steering = GetSteerToPath(static_cast<const PomdpState*>(state)[0]);

	logv << __FUNCTION__ << "] Calculate acceleration" << endl;

	double acceleration;
	// Closest pedestrian in front
	for (int i = 0; i < state->num; i++) {
		const AgentStruct & p = state->agents[i];

		float infront_angle = ModelParams::IN_FRONT_ANGLE_DEG;
		if (Globals::config.pruning_constant > 100.0)
			infront_angle = 60.0;
		if (!InFront(p.pos, state->car, infront_angle))
			continue;

		double d_along = COORD::DirectedDistance(carpos, p.pos, carheading);
		double d_tang = COORD::DirectedDistance(carpos, p.pos,
				carheading + M_PI / 2.0);

		if (p.type == AgentType::car) {
			d_along = d_along - 2.0;
			d_tang = d_tang - 1.0;
		}

		mindist_along = min(mindist_along, d_along);
		mindist_tang = min(mindist_tang, d_tang);
	}

	if (mindist_along < 2 || mindist_tang < 1.0) {
		acceleration = (carvel <= 0.01) ? 0 : -ModelParams::ACC_SPEED;
	} else if (mindist_along < 4 && mindist_tang < 2.0) {
		if (carvel > ModelParams::VEL_MAX / 1.5 + 1e-4)
			acceleration = -ModelParams::ACC_SPEED;
		else if (carvel < ModelParams::VEL_MAX / 2.0 - 1e-4)
			acceleration = ModelParams::ACC_SPEED;
		else
			acceleration = 0.0;
	} else
		acceleration =
				carvel >= ModelParams::VEL_MAX - 1e-4 ?
						0 : ModelParams::ACC_SPEED;

	logv << __FUNCTION__ << "] Calculate action ID" << endl;
	return PedPomdp::GetActionID(steering, acceleration);
}

enum {
	CLOSE_STATIC,
	CLOSE_MOVING,
	MEDIUM_FAST,
	MEDIUM_MED,
	MEDIUM_SLOW,
	FAR_MAX,
	FAR_NOMAX,
};

bool WorldModel::InFront(const COORD ped_pos, const CarStruct car,
		double infront_angle_deg) const {
	if (infront_angle_deg == -1)
		infront_angle_deg = ModelParams::IN_FRONT_ANGLE_DEG;
	if (infront_angle_deg >= 180.0) {
		// inFront check is disabled
		return true;
	}

	double d0 = COORD::EuclideanDistance(car.pos, ped_pos);
	double dot = DotProduct(cos(car.heading_dir), sin(car.heading_dir),
			ped_pos.x - car.pos.x, ped_pos.y - car.pos.y);
	double cosa = (d0 > 0) ? dot / (d0) : 0;
	assert(cosa <= 1.0 + 1E-8 && cosa >= -1.0 - 1E-8);
	return cosa > in_front_angle_cos;
}

/**
 * H: center of the head of the car
 * N: a point right in front of the car
 * M: an arbitrary point
 *
 * Check whether M is in the safety zone
 */
bool InCollision(double Px, double Py, double Cx, double Cy, double Ctheta,
		bool expand = true);
bool inCarlaCollision(double ped_x, double ped_y, double car_x, double car_y,
		double Ctheta, double car_extent_x, double car_extent_y, bool flag = 0);

bool InRealCollision(double Px, double Py, double Cx, double Cy, double Ctheta,
		bool expand = true);
std::vector<COORD> ComputeRect(COORD pos, double heading,
		double ref_to_front_side, double ref_to_back_side,
		double ref_front_side_angle, double ref_back_side_angle);
bool InCollision(std::vector<COORD> rect_1, std::vector<COORD> rect_2);

bool WorldModel::InCollision(const PomdpState& state) {
	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				return true;
			}
		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 0))
				return true;
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}
	}

	return false;
}

bool WorldModel::InRealCollision(const PomdpStateWorld& state, int &id,
		double infront_angle) {
	id = -1;
	if (infront_angle == -1) {
		infront_angle = ModelParams::IN_FRONT_ANGLE_DEG;
	}

	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car, infront_angle))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InRealCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				id = agent.id;
			}
		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 1)) {
				id = agent.id;
			}
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}

		if (id != -1) {
			logv << "[WorldModel::InRealCollision] car_pos: (" << car_pos.x
					<< "," << car_pos.y << "), heading: ("
					<< std::cos(state.car.heading_dir) << ","
					<< std::sin(state.car.heading_dir) << "), agent_pos: ("
					<< agent.pos.x << "," << agent.pos.y << ")\n";
			return true;
		}
	}

	return false;
}

bool WorldModel::InRealCollision(const PomdpStateWorld& state,
		double infront_angle_deg) {
	const COORD& car_pos = state.car.pos;
	int id = -1;
	if (infront_angle_deg == -1) {
		infront_angle_deg = ModelParams::IN_FRONT_ANGLE_DEG;
	}

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car, infront_angle_deg))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InRealCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				id = agent.id;
			}
		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 1)) {
				id = agent.id;
			}
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}

		if (id != -1) {
			logv << "[WorldModel::InRealCollision] car_pos: (" << car_pos.x
					<< "," << car_pos.y << "), heading: ("
					<< std::cos(state.car.heading_dir) << ","
					<< std::sin(state.car.heading_dir) << "), agent_pos: ("
					<< agent.pos.x << "," << agent.pos.y << ")\n";
			return true;
		}
	}

	return false;
}

bool WorldModel::InCollision(const PomdpStateWorld& state) {
	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				return true;
			}
		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 0))
				return true;
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}
	}

	return false;
}

bool WorldModel::InCollision(const PomdpState& state, int &id) {
	id = -1;
	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				id = agent.id;
				return true;
			}

		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 0)) {
				id = agent.id;
				return true;
			}
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}
	}

	if (id != -1) {
		return true;
	}

	return false;
}

bool WorldModel::InCollision(const PomdpStateWorld& state, int &id) {
	id = -1;
	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				id = agent.id;
			}

		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 0)) {
				id = agent.id;
			}
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}

		if (id != -1)
			logv << "[WorldModel::InRealCollision] car_pos: (" << car_pos.x
					<< "," << car_pos.y << "), heading: ("
					<< std::cos(state.car.heading_dir) << ","
					<< std::sin(state.car.heading_dir) << "), agent_pos: ("
					<< agent.pos.x << "," << agent.pos.y << ")\n";
	}

	if (id != -1) {
		return true;
	}

	return false;
}

bool WorldModel::IsGlobalGoal(const CarStruct& car) {
	double d = COORD::EuclideanDistance(car.pos, path.back());
	return (d < ModelParams::GOAL_TOLERANCE);
}

void WorldModel::GetClosestPed(const PomdpState& state, int& closest_front_ped,
		double& closest_front_dist, int& closest_side_ped,
		double& closest_side_dist) {
	closest_front_ped = -1;
	closest_front_dist = numeric_limits<double>::infinity();
	closest_side_ped = -1;
	closest_side_dist = numeric_limits<double>::infinity();
	const COORD& carpos = state.car.pos;

	// Find the closest pedestrian in front
	for (int i = 0; i < state.num; i++) {
		const AgentStruct& p = state.agents[i];
		bool front = InFront(p.pos, state.car);
		double d = COORD::EuclideanDistance(carpos, p.pos);
		if (front) {
			if (d < closest_front_dist) {
				closest_front_dist = d;
				closest_front_ped = i;
			}
		} else {
			if (d < closest_side_dist) {
				closest_side_dist = d;
				closest_side_ped = i;
			}
		}
	}
}

bool WorldModel::MovingAway(const PomdpState& state, int agent) {
	const auto& carpos = state.car.pos;

	const auto& pedpos = state.agents[agent].pos;
	const auto& goalpos = GetGoalPos(state.agents[agent]);

	if (IsStopIntention(state.agents[agent].intention, agent))
		return false;

	return DotProduct(goalpos.x - pedpos.x, goalpos.y - pedpos.y,
			cos(state.car.heading_dir), sin(state.car.heading_dir)) > 0;
}

///get the min distance between car and the agents in its front
double WorldModel::GetMinCarPedDist(const PomdpState& state) {
	double mindist = numeric_limits<double>::infinity();
	const auto& carpos = state.car.pos;

	// Find the closest pedestrian in front
	for (int i = 0; i < state.num; i++) {
		const auto& p = state.agents[i];
		if (!InFront(p.pos, state.car))
			continue;
		double d = COORD::EuclideanDistance(carpos, p.pos);
		if (d >= 0 && d < mindist)
			mindist = d;
	}

	return mindist;
}

///get the min distance between car and the agents
double WorldModel::GetMinCarPedDistAllDirs(const PomdpState& state) {
	double mindist = numeric_limits<double>::infinity();
	const auto& carpos = state.car.pos;

	// Find the closest pedestrian in front
	for (int i = 0; i < state.num; i++) {
		const auto& p = state.agents[i];
		double d = COORD::EuclideanDistance(carpos, p.pos);
		if (d >= 0 && d < mindist)
			mindist = d;
	}

	return mindist;
}

int WorldModel::MinStepToGoal(const PomdpState& state) {
	double d = path.GetLength(path.Nearest(state.car.pos));
	if (d < 0)
		d = 0;
	return int(ceil(d / (ModelParams::VEL_MAX / freq)));
}

void WorldModel::AgentStep(AgentStruct &agent, Random& random) {
	double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
	if (goal_mode == "cur_vel") {
		AgentStepCurVel(agent, 1, noise);
	} else if (goal_mode == "path") {
		AgentStepPath(agent, 1, noise);
	}
}

void WorldModel::AgentStep(AgentStruct &agent, double& random) {

	double noise = sqrt(-2 * log(random));
	if (FIX_SCENARIO != 1 && !CPUDoPrint) {
		random = QuickRandom::RandGeneration(random);
	}
	noise *= cos(2 * M_PI * random) * ModelParams::NOISE_GOAL_ANGLE;

	if (goal_mode == "cur_vel") {
		AgentStepCurVel(agent, 1, noise);
	} else if (goal_mode == "path") {
		AgentStepPath(agent, 1, noise);
	}
}

double gaussian_prob(double x, double stddev) {
	double a = 1.0 / stddev / sqrt(2 * M_PI);
	double b = -x * x / 2.0 / (stddev * stddev);
	return a * exp(b);
}

COORD WorldModel::GetGoalPosFromPaths(int agent_id, int intention_id,
		int pos_along_path, const COORD& agent_pos, const COORD& agent_vel,
		AgentType type, bool agent_cross_dir) {
	auto& path_candidates = PathCandidates(agent_id);
	if (intention_id < path_candidates.size()) {
		auto& path = path_candidates[intention_id];
		COORD pursuit = path[path.Forward(pos_along_path, PURSUIT_LEN)];
		return pursuit;
	} else if (IsCurVelIntention(intention_id, agent_id)) {
		COORD dir(agent_vel);
		dir.AdjustLength(PURSUIT_LEN);
		return agent_pos + dir;
	} else if (IsStopIntention(intention_id, agent_id)) { // stop intention
		return COORD(agent_pos.x, agent_pos.y);
	} else {
		ERR(string_sprintf(
				"Intention ID %d excesses # intentions %d for agent %d of type %d\n",
				intention_id, GetNumIntentions(agent_id), agent_id,
				type));
		return COORD(0.0, 0.0);
	}
}

bool fetch_cross_dir(const Agent& agent) {
	if (agent.type() == AgentType::ped)
		return static_cast<const Pedestrian*>(&agent)->cross_dir;
	else
		return true;
}

std::vector<COORD> fetch_bounding_box(const Agent& agent) {
	if (agent.type() == AgentType::car) {
		return static_cast<const Vehicle*>(&agent)->bb;
	} else {
		std::vector<COORD> bb;
		return bb;
	}
}

double fetch_heading_dir(const Agent& agent) {
	if (agent.type() == AgentType::car) {
		auto* car = static_cast<const Vehicle*>(&agent);
		return static_cast<const Vehicle*>(&agent)->heading_dir;
	} else
		return 0.0;
}

COORD WorldModel::GetGoalPos(const Agent& agent, int intention_id) {
	if (goal_mode == "path") {
		bool cross_dir = fetch_cross_dir(agent);
		return GetGoalPosFromPaths(agent.id, intention_id, 0,
				COORD(agent.w, agent.h), agent.vel, agent.type(), cross_dir); // assuming path is up-to-date
	} else if (goal_mode == "cur_vel")
		return COORD(agent.w, agent.h) + agent.vel.Scale(PURSUIT_LEN);
	else {
		RasieUnsupportedGoalMode(__FUNCTION__);
		return COORD(0.0, 0.0);
	}
}

COORD WorldModel::GetGoalPos(const AgentStruct& agent, int intention_id) {

	if (intention_id == -1)
		intention_id = agent.intention;

	if (goal_mode == "path")
		return GetGoalPosFromPaths(agent.id, intention_id, agent.pos_along_path,
				agent.pos, agent.vel, agent.type, agent.cross_dir);
	else if (goal_mode == "cur_vel")
		return agent.pos + agent.vel.Scale(PURSUIT_LEN);
	else {
		RasieUnsupportedGoalMode(__FUNCTION__);
		return COORD(0.0, 0.0);
	}
}

COORD WorldModel::GetGoalPos(const AgentBelief& agent, int intention_id) {

	int pos_along_path = 0; // assumed that AgentBelied has up-to-date paths
	if (goal_mode == "path")
		return GetGoalPosFromPaths(agent.id, intention_id, pos_along_path,
				agent.pos, agent.vel, agent.type, agent.cross_dir);
	else if (goal_mode == "cur_vel")
		return agent.pos + agent.vel.Scale(PURSUIT_LEN);
	else {
		RasieUnsupportedGoalMode(__FUNCTION__);
		return COORD(0.0, 0.0);
	}
}

int WorldModel::GetNumIntentions(const AgentStruct& agent) {
	if (goal_mode == "path")
		return NumPaths(agent.id) + include_stop_intention
				+ include_curvel_intention;
	else if (goal_mode == "cur_vel")
		return 2;
	else
		return 0;
}

int WorldModel::GetNumIntentions(const Agent& agent) {
	if (goal_mode == "path")
		return NumPaths(agent.id) + include_stop_intention
				+ include_curvel_intention;
	else if (goal_mode == "cur_vel")
		return 2;
	else
		return 0;
}

int WorldModel::GetNumIntentions(int agent_id) {
	if (goal_mode == "path")
		return NumPaths(agent_id) + include_stop_intention
				+ include_curvel_intention;
	else if (goal_mode == "cur_vel")
		return 2;
	else
		return 0;
}

void WorldModel::AgentStepCurVel(AgentStruct& agent, int step, double noise) {
	if (noise != 0) {
		double a = agent.vel.GetAngle();
		a += noise;
		MyVector move(a, step * agent.vel.Length() / freq, 0);
		agent.pos.x += move.dw;
		agent.pos.y += move.dh;
	} else {
		agent.pos.x += agent.vel.x * (float(step) / freq);
		agent.pos.y += agent.vel.y * (float(step) / freq);
	}
}

double WorldModel::PurepursuitAngle(const CarStruct& car,
		COORD& pursuit_point) const {
	COORD rear_pos;
	rear_pos.x = car.pos.x - ModelParams::CAR_REAR * cos(car.heading_dir);
	rear_pos.y = car.pos.y - ModelParams::CAR_REAR * sin(car.heading_dir);

	double offset = (rear_pos - pursuit_point).Length();
	double target_angle = atan2(pursuit_point.y - rear_pos.y,
			pursuit_point.x - rear_pos.x);
	double angular_offset = CapAngle(target_angle - car.heading_dir);

	COORD relative_point(offset * cos(angular_offset),
			offset * sin(angular_offset));
	if (relative_point.x == 0)
		return 0;
	else {
		double turning_radius = relative_point.Length()
				/ (2 * abs(relative_point.y)); // Intersecting chords theorem.
		double steering_angle = atan2(ModelParams::CAR_WHEEL_DIST,
				turning_radius);
		if (relative_point.y < 0)
			steering_angle *= -1;

		return steering_angle;
	}
}

double WorldModel::PurepursuitAngle(const AgentStruct& agent,
		COORD& pursuit_point) const {
	COORD rear_pos;
	rear_pos.x = agent.pos.x
			- agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir);
	rear_pos.y = agent.pos.y
			- agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);

	double offset = (rear_pos - pursuit_point).Length();
	double target_angle = atan2(pursuit_point.y - rear_pos.y,
			pursuit_point.x - rear_pos.x);
	double angular_offset = CapAngle(target_angle - agent.heading_dir);

	COORD relative_point(offset * cos(angular_offset),
			offset * sin(angular_offset));
	if (relative_point.x == 0)
		return 0;
	else {
		double turning_radius = relative_point.Length()
				/ (2 * abs(relative_point.y)); // Intersecting chords theorem.
		double steering_angle = atan2(agent.bb_extent_y * 2 * 0.8,
				turning_radius);
		if (relative_point.y < 0)
			steering_angle *= -1;

		return steering_angle;
	}
}

std::vector<RVO::Vector2> WorldModel::GetBoundingBoxCorners(
		AgentStruct& agent) {
	RVO::Vector2 forward_vec = RVO::Vector2(cos(agent.heading_dir),
			sin(agent.heading_dir));
	RVO::Vector2 sideward_vec = RVO::Vector2(-sin(agent.heading_dir),
			cos(agent.heading_dir)); // rotate 90 degree counter-clockwise

	double side_len = agent.bb_extent_x;
	double forward_len = agent.bb_extent_y;

	if (agent.type == AgentType::ped) {
		side_len = 0.23;
		forward_len = 0.23;
	}

	RVO::Vector2 pos = RVO::Vector2(agent.pos.x, agent.pos.y);

	return std::vector<RVO::Vector2>(
			{ pos + forward_vec * forward_len + sideward_vec * side_len, pos
					- forward_vec * forward_len + sideward_vec * side_len, pos
					- forward_vec * forward_len - sideward_vec * side_len, pos
					+ forward_vec * forward_len - sideward_vec * side_len });
}

std::vector<RVO::Vector2> WorldModel::GetBoundingBoxCorners(
		RVO::Vector2 forward_vec, RVO::Vector2 sideward_vec, RVO::Vector2 pos,
		double forward_len, double side_len) {

	return std::vector<RVO::Vector2>(
			{ pos + forward_vec * forward_len + sideward_vec * side_len, pos
					- forward_vec * forward_len + sideward_vec * side_len, pos
					- forward_vec * forward_len - sideward_vec * side_len, pos
					+ forward_vec * forward_len - sideward_vec * side_len });
}

void WorldModel::CalBBExtents(AgentBelief& b, AgentStruct& agent) {
	if (agent.type == AgentType::ped) {
		agent.bb_extent_x = 0.3;
		agent.bb_extent_y = 0.3;
	} else {
		agent.bb_extent_x = 0.0;
		agent.bb_extent_y = 0.0;
	}
	CalBBExtents(b.pos, b.heading_dir, b.bb, agent.bb_extent_x,
			agent.bb_extent_y);
}

void WorldModel::CalBBExtents(AgentStruct& agent, std::vector<COORD>& bb,
		double heading_dir) {
	if (agent.type == AgentType::ped) {
		agent.bb_extent_x = 0.3;
		agent.bb_extent_y = 0.3;
	} else {
		agent.bb_extent_x = 0.0;
		agent.bb_extent_y = 0.0;
	}
	CalBBExtents(agent.pos, heading_dir, bb, agent.bb_extent_x,
			agent.bb_extent_y);
}

void WorldModel::CalBBExtents(COORD pos, double heading_dir,
		vector<COORD>& bb, double& extent_x, double& extent_y) {
	COORD forward_vec = COORD(cos(heading_dir), sin(heading_dir));
	COORD sideward_vec = COORD(-sin(heading_dir), cos(heading_dir));

	for (auto& point : bb) {
		extent_x = max((point - pos).dot(sideward_vec), extent_x);
		extent_y = max((point - pos).dot(forward_vec), extent_y);
	}
}

void WorldModel::AgentStepPath(AgentStruct& agent, int step, double noise,
		bool doPrint) {
	if (agent.type == AgentType::ped)
		PedStepPath(agent, step, noise, doPrint);
	else if (agent.type == AgentType::car)
		VehStepPath(agent, step, noise, doPrint);
	else
		ERR(string_sprintf("unsupported agent mode %d", agent.type));
}

void WorldModel::PedStepPath(AgentStruct& agent, int step, double noise,
		bool doPrint) {
	auto& path_candidates = PathCandidates(agent.id);

	int intention = doPrint? 0: agent.intention;
	int old_path_pos = agent.pos_along_path;

	if (intention < path_candidates.size()) {
		auto& path = path_candidates[intention];
		agent.pos_along_path = path.Forward(agent.pos_along_path,
				agent.speed * (float(step) / freq));
		COORD new_pos = path[agent.pos_along_path];

		if (noise != 0) {
			COORD goal_vec = new_pos - agent.pos;
			double a = goal_vec.GetAngle() + noise;
			MyVector move(a, step * agent.speed / freq, 0);
			agent.pos.x += move.dw;
			agent.pos.y += move.dh;
		} else {
			agent.pos = new_pos;
		}
		agent.vel = (new_pos - agent.pos) * freq;

		if (doPrint && agent.pos_along_path == old_path_pos)
			logv << "[AgentStepPath]: agent " << agent.id
					<< " no move: path length " << path.size() << " speed "
					<< agent.speed << " forward distance "
					<< agent.speed * (float(step) / freq) << endl;
	} else if (IsCurVelIntention(intention, agent.id)) {
		agent.pos = agent.pos + agent.vel * (1.0 / freq);
	}
}

void WorldModel::VehStepPath(AgentStruct& agent, int step, double noise,
		bool doPrint) {
	auto& path_candidates = PathCandidates(agent.id);
	int intention = doPrint? 0: agent.intention;
	int old_path_pos = agent.pos_along_path;

	if (intention < path_candidates.size()) {
		COORD old_pos = agent.pos;
		auto& path = path_candidates[intention];
		int pursuit_pos = path.Forward(agent.pos_along_path, PURSUIT_LEN);
		COORD pursuit_point = path[pursuit_pos];

		double steering = PControlAngle<AgentStruct>(agent, pursuit_point) + noise;
		BicycleModel(agent, steering, agent.speed);

		agent.pos_along_path = path.Nearest(agent.pos);
		agent.vel = (agent.pos - old_pos) * freq;

		if (doPrint && agent.pos_along_path == old_path_pos)
			logv << "[AgentStepPath]: agent " << agent.id
					<< " no move: path length " << path.size() << " speed "
					<< agent.speed << " forward distance "
					<< agent.speed * (float(step) / freq) << endl;
	} else if (IsCurVelIntention(intention, agent.id)) {
		agent.pos = agent.pos + agent.vel * (1.0 / freq);
	}
}

void WorldModel::EnsureMeanDirExist(int agent_id) {
	auto it = ped_mean_dirs.find(agent_id);
	if (it == ped_mean_dirs.end()) {
		vector<COORD> dirs(GetNumIntentions(agent_id));
		ped_mean_dirs[agent_id] = dirs;
	} else if (it->second.size() != GetNumIntentions(agent_id)) { // path list size has been updated
		ped_mean_dirs[agent_id].resize(GetNumIntentions(agent_id));
	}
}

bool WorldModel::IsStopIntention(int intention, int agent_id) {
	if (include_stop_intention)
		return intention == GetNumIntentions(agent_id) - include_stop_intention;
	else
		return false;
}

bool WorldModel::IsCurVelIntention(int intention, int agent_id) {
	if (include_curvel_intention)
		return intention
				== GetNumIntentions(agent_id) - include_curvel_intention
						- include_stop_intention;
	else
		return false;
}

double WorldModel::AgentMoveProb(const AgentBelief& prev_agent,
		const Agent& agent, int intention_id, int ped_mode) {
	const double K = 0.001;
	COORD cur_pos(agent.w, agent.h);
	int agent_id = agent.id;
	COORD prev_pos = prev_agent.pos;
	COORD prev_goal;
	if (ped_mode == AGENT_ATT) {
		EnsureMeanDirExist(agent_id);

		if (intention_id >= ped_mean_dirs[agent_id].size()) {
			ERR(
					string_sprintf(
							"Encountering overflowed intention id %d at agent %d\n ",
							intention_id, agent_id));
		}
		prev_goal = ped_mean_dirs[agent_id][intention_id] + prev_pos;
	} else {
		prev_goal = GetGoalPos(prev_agent, intention_id);
	}

	double goal_dist = Norm(prev_goal.x - prev_pos.x, prev_goal.y - prev_pos.y);
	double move_dist = Norm(cur_pos.x - prev_pos.x, cur_pos.y - prev_pos.y);

	double sensor_noise = 0.1;
	if (IsStopIntention(intention_id, agent.id)) {
		logv << "stop intention" << endl;
		return (move_dist < sensor_noise) ? 0.4 : 0;
	} else {
		double angle_error = COORD::Angle(prev_pos, cur_pos, prev_goal,
				sensor_noise);
		double angle_prob = gaussian_prob(angle_error,
				ModelParams::NOISE_GOAL_ANGLE) + K;
		double mean_vel =
				ped_mode == AGENT_ATT ?
						ped_mean_dirs[agent_id][intention_id].Length() :
						prev_agent.speed / freq;
		double vel_error = move_dist - mean_vel;
		double vel_prob = gaussian_prob(vel_error,
				ModelParams::NOISE_PED_VEL / freq) + K;

		if (isnan(angle_prob) || isnan(vel_prob))
			ERR("Get likelihood as NAN");

		return angle_prob * vel_prob;
	}
}

void WorldModel::FixGPUVel(CarStruct &car) {
	float tmp = car.vel / (ModelParams::ACC_SPEED / freq);
	car.vel = ((int) (tmp + 0.5)) * (ModelParams::ACC_SPEED / freq);
}

void WorldModel::BicycleModel(CarStruct &car, double steering, double end_vel) {
	if (steering != 0) {
		assert(tan(steering) != 0);
		double TurningRadius = ModelParams::CAR_WHEEL_DIST / tan(steering);
		assert(TurningRadius != 0);
		double beta = end_vel / freq / TurningRadius;

		COORD rear_pos;
		rear_pos.x = car.pos.x - ModelParams::CAR_REAR * cos(car.heading_dir);
		rear_pos.y = car.pos.y - ModelParams::CAR_REAR * sin(car.heading_dir);
		// move and rotate
		rear_pos.x += TurningRadius
				* (sin(car.heading_dir + beta) - sin(car.heading_dir));
		rear_pos.y += TurningRadius
				* (cos(car.heading_dir) - cos(car.heading_dir + beta));
		car.heading_dir = CapAngle(car.heading_dir + beta);
		car.pos.x = rear_pos.x + ModelParams::CAR_REAR * cos(car.heading_dir);
		car.pos.y = rear_pos.y + ModelParams::CAR_REAR * sin(car.heading_dir);
	} else {
		car.pos.x += (end_vel / freq) * cos(car.heading_dir);
		car.pos.y += (end_vel / freq) * sin(car.heading_dir);
	}
}

void WorldModel::BicycleModel(AgentStruct &agent, double steering,
		double end_vel) {
	if (steering != 0) {
		assert(tan(steering) != 0);
		// assuming front-real length is 0.8 * total car length
		double TurningRadius = agent.bb_extent_y * 2 * 0.8 / tan(steering);
		assert(TurningRadius != 0);
		double beta = end_vel / freq / TurningRadius;

		COORD rear_pos;
		rear_pos.x = agent.pos.x
				- agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir);
		rear_pos.y = agent.pos.y
				- agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);
		// move and rotate
		rear_pos.x += TurningRadius
				* (sin(agent.heading_dir + beta) - sin(agent.heading_dir));
		rear_pos.y += TurningRadius
				* (cos(agent.heading_dir) - cos(agent.heading_dir + beta));
		agent.heading_dir = CapAngle(agent.heading_dir + beta);
		agent.pos.x = rear_pos.x
				+ agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir);
		agent.pos.y = rear_pos.y
				+ agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);
	} else {
		agent.pos.x += (end_vel / freq) * cos(agent.heading_dir);
		agent.pos.y += (end_vel / freq) * sin(agent.heading_dir);
	}
}

void WorldModel::RobStep(CarStruct &car, double steering, Random& random) {
	BicycleModel(car, steering, car.vel);
}

void WorldModel::RobStep(CarStruct &car, double steering, double& random) {
	BicycleModel(car, steering, car.vel);
}

void WorldModel::RobStep(CarStruct &car, double& random, double acc,
		double steering) {
	double end_vel = car.vel + acc / freq;
	end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);
	BicycleModel(car, steering, end_vel);
}

void WorldModel::RobStep(CarStruct &car, Random& random, double acc,
		double steering) {
	double end_vel = car.vel + acc / freq;
	end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);
	BicycleModel(car, steering, end_vel);
}

void WorldModel::RobVelStep(CarStruct &car, double acc, Random& random) {
	const double N = ModelParams::NOISE_ROBVEL;
	if (N > 0) {
		double prob = random.NextDouble();
		if (prob > N) {
			car.vel += acc / freq;
		}
	} else {
		car.vel += acc / freq;
	}
	car.vel = max(min(car.vel, ModelParams::VEL_MAX), 0.0);
	return;
}

void WorldModel::RobVelStep(CarStruct &car, double acc, double& random) {
	const double N = ModelParams::NOISE_ROBVEL;
	if (N > 0) {
		if (FIX_SCENARIO != 1 && !CPUDoPrint)
			random = QuickRandom::RandGeneration(random);
		double prob = random;
		if (prob > N) {
			car.vel += acc / freq;
		}
	} else {
		car.vel += acc / freq;
	}
	car.vel = max(min(car.vel, ModelParams::VEL_MAX), 0.0);
	return;
}

void WorldModel::RobStepCurVel(CarStruct &car) {
	car.pos.x += (car.vel / freq) * cos(car.heading_dir);
	car.pos.y += (car.vel / freq) * sin(car.heading_dir);
}

void WorldModel::RobStepCurAction(CarStruct &car, double acc, double steering) {
	double det_prob = 1;
	RobStep(car, steering, det_prob);
	RobVelStep(car, acc, det_prob);
}

double WorldModel::ISRobVelStep(CarStruct &car, double acc, Random& random) {
	const double N = 4 * ModelParams::NOISE_ROBVEL;
	double weight = 1;
	if (N > 0) {
		double prob = random.NextDouble();
		if (prob > N) {
			car.vel += acc / freq;
			weight = (1.0 - ModelParams::NOISE_ROBVEL) / (1.0 - N);
		} else
			weight = ModelParams::NOISE_ROBVEL / N;
	} else {
		car.vel += acc / freq;
	}
	car.vel = max(min(car.vel, ModelParams::VEL_MAX), 0.0);
	return weight;
}

void WorldModel::SetPath(Path path) {
	this->path = path;
	ModelParams::GOAL_TRAVELLED = path.GetLength();
}

void WorldModel::UpdatePedBelief(AgentBelief& b, const Agent& curr_agent) {
	const double ALPHA = 0.1;
	const double SMOOTHING = ModelParams::BELIEF_SMOOTHING;
	bool debug = false;

	b.bb = fetch_bounding_box(curr_agent);
	b.heading_dir = fetch_heading_dir(curr_agent);
	b.cross_dir = fetch_cross_dir(curr_agent);

	if (!curr_agent.reset_intention) {
		b.goal_paths = PathCandidates(b.id);
		for (int i = 0; i < GetNumIntentions(curr_agent); i++) {
			// Attentive mode
			logv << "TEST ATT" << endl;
			double prob = AgentMoveProb(b, curr_agent, i, AGENT_ATT);
			if (debug)
				cout << "attentive likelihood " << i << ": " << prob << endl;
			b.prob_modes_goals[AGENT_ATT][i] *= prob;
			b.prob_modes_goals[AGENT_ATT][i] += SMOOTHING
					/ GetNumIntentions(curr_agent) / 2.0;
			// Detracted mode
			logv << "TEST DIS" << endl;
			prob = AgentMoveProb(b, curr_agent, i, AGENT_DIS);
			if (debug)
				cout << "Detracted likelihood " << i << ": " << prob << endl;
			b.prob_modes_goals[AGENT_DIS][i] *= prob;
			b.prob_modes_goals[AGENT_DIS][i] += SMOOTHING
					/ GetNumIntentions(curr_agent) / 2.0;
		}
		if (debug) {
			for (double w : b.prob_modes_goals[AGENT_ATT]) {
				cout << w << " ";
			}
			cout << endl;
			for (double w : b.prob_modes_goals[AGENT_DIS]) {
				cout << w << " ";
			}
			cout << endl;
		}

		// normalize
		double total_weight_att = accumulate(
				b.prob_modes_goals[AGENT_ATT].begin(),
				b.prob_modes_goals[AGENT_ATT].end(), double(0.0));
		double total_weight_dis = accumulate(
				b.prob_modes_goals[AGENT_DIS].begin(),
				b.prob_modes_goals[AGENT_DIS].end(), double(0.0));
		double total_weight = total_weight_att + total_weight_dis;

		if (total_weight <= 0.01 && GetNumIntentions(curr_agent) > 0)
			ERR(string_sprintf("total_weight too small %f", total_weight));

		if (debug)
			cout << "[updatePedBelief] total_weight = " << total_weight << endl;
		for (double& w : b.prob_modes_goals[AGENT_ATT]) {
			w /= total_weight;
		}
		for (double& w : b.prob_modes_goals[AGENT_DIS]) {
			w /= total_weight;
		}
	}

	COORD cur_pos(curr_agent.w, curr_agent.h);
	double moved_dist = COORD::EuclideanDistance(b.pos, cur_pos);
	b.vel = b.vel * ALPHA
			+ (cur_pos - b.pos) * (ModelParams::CONTROL_FREQ * (1 - ALPHA));
	b.speed = b.vel.Length();

	b.pos = cur_pos;

}

double WorldModel::GetPrefSpeed(const Agent& agent) {
	return agent.vel.Length();
}

AgentBelief WorldModel::InitPedBelief(const Agent& agent) {
	AgentBelief b;
	b.id = agent.id;
	b.type = agent.type();
	b.pos = COORD(agent.w, agent.h);
	b.speed = GetPrefSpeed(agent);
	b.vel = agent.vel;
	b.reset = agent.reset_intention;
	b.heading_dir = fetch_heading_dir(agent);
	b.cross_dir = fetch_cross_dir(agent);
	b.bb = fetch_bounding_box(agent);

	int num_types = NUM_AGENT_TYPES;
	for (int i = 0; i < num_types; i++) {
		b.prob_modes_goals.push_back(vector<double>());
		b.prob_modes_goals.back().reserve(20);
		for (int i = 0; i < GetNumIntentions(agent); i++)
			b.prob_modes_goals.back().push_back(
					1.0 / GetNumIntentions(agent) / num_types);
	}

	return b;
}

void WorldStateTracker::CleanAgents() {
	CleanPed();
	CleanVeh();
	if (logging::level() >= logging::DEBUG)
		model.PrintPathMap();
}

void WorldStateTracker::CleanPed() {
	vector<Pedestrian> ped_list_new;
	for (int i = 0; i < ped_list.size(); i++) {
		bool insert = AgentIsAlive(ped_list[i], ped_list_new);
		if (insert)
			ped_list_new.push_back(ped_list[i]);
		else {
			logi << "Cleaning ped " << ped_list[i].id
					<< " from ped_list, id_map_num_paths, id_map_paths" << endl;
			model.id_map_num_paths.erase(ped_list[i].id);
			model.id_map_paths.erase(ped_list[i].id);
		}
	}
	ped_list = ped_list_new;
}

void WorldStateTracker::CleanVeh() {
	vector<Vehicle> veh_list_new;
	for (int i = 0; i < veh_list.size(); i++) {
		bool insert = AgentIsAlive(veh_list[i], veh_list_new);
		if (insert)
			veh_list_new.push_back(veh_list[i]);
		else {
			logi << "Cleaning veh " << veh_list[i].id
					<< " from veh_list, id_map_num_paths, id_map_paths" << endl;
			model.id_map_num_paths.erase(veh_list[i].id);
			model.id_map_paths.erase(veh_list[i].id);
		}
	}
	veh_list = veh_list_new;
}

void WorldStateTracker::TracPos(Agent& des, const Agent& src, bool doPrint) {
	des.w = src.w;
	des.h = src.h;
}

void WorldStateTracker::TracBoundingBox(Vehicle& des, const Vehicle& src,
		bool doPrint) {
	des.bb = src.bb;
	des.heading_dir = src.heading_dir;
}

void WorldStateTracker::TracCrossDirs(Pedestrian& des, const Pedestrian& src,
		bool doPrint) {
	des.cross_dir = src.cross_dir;
}

void WorldStateTracker::UpdatePathPool(Agent& des) {
	model.id_map_paths[des.id] = des.paths;

	switch (des.type()) {
	case AgentType::ped:
		model.id_map_num_paths[des.id] = des.paths.size();
		break;
	case AgentType::car:
		model.id_map_num_paths[des.id] = des.paths.size();
		break;
	default:
		cout << __FUNCTION__ << ": unsupported agent type " << des.type()
				<< endl;
		raise (SIGABRT);
		break;
	}
}

void WorldStateTracker::TracIntention(Agent& des, const Agent& src,
		bool doPrint) {
	if (des.paths.size() != src.paths.size())
		des.reset_intention = true;
	else
		des.reset_intention = src.reset_intention;
	des.paths.resize(0);

	for (const Path& path : src.paths) {
		des.paths.push_back(path.Interpolate(60.0)); // reset the resolution of the path to ModelParams:PATH_STEP
	}

	UpdatePathPool(des);
}

void WorldStateTracker::TrackVel(Agent& des, const Agent& src, bool& no_move,
		bool doPrint) {
	if (use_vel_from_summit) {
		des.vel = src.vel;
		no_move = false;
	} else {
		double duration = src.ros_time_stamp - des.ros_time_stamp;
		if (duration < 0.001 / Globals::config.time_scale) {
			no_move = false;
			DEBUG( string_sprintf(
					"Update duration too short for agent %d: %f, (%f-%f)",
					src.id, duration, src.ros_time_stamp,
					des.ros_time_stamp));
			return;
		}
		des.vel.x = (src.w - des.w) / duration;
		des.vel.y = (src.h - des.h) / duration;
		if (des.vel.Length() > 1e-4) {
			no_move = false;
		}
	}
}

void WorldStateTracker::UpdateVehState(const Vehicle& veh, bool doPrint) {

	int i = 0;
	bool no_move = true;
	for (; i < veh_list.size(); i++) {
		if (veh_list[i].id == veh.id) {
			logv << "updating agent " << veh.id << endl;
			TrackVel(veh_list[i], veh, no_move, doPrint);
			TracPos(veh_list[i], veh, doPrint);
			TracBoundingBox(veh_list[i], veh, doPrint);
			veh_list[i].time_stamp = veh.time_stamp;
			break;
		}
	}

	if (i == veh_list.size()) {
		logv << "updating new agent " << veh.id << endl;
		no_move = false;
		veh_list.push_back(veh);
		if (!use_vel_from_summit) {
			veh_list.back().vel.x = 0.01;
			veh_list.back().vel.y = 0.01;
		} else {
			veh_list.back().vel = veh.vel;
		}
		veh_list.back().time_stamp = veh.time_stamp;
	}
}

void WorldStateTracker::UpdateVehPaths(const Vehicle& veh, bool doPrint) {

	int i = 0;
	bool no_move = true;
	for (; i < veh_list.size(); i++) {
		if (veh_list[i].id == veh.id) {
			logv << "updating agent path " << veh.id << endl;
			TracIntention(veh_list[i], veh, doPrint);
			break;
		}
	}

}

void WorldStateTracker::UpdatePedState(const Pedestrian& agent, bool doPrint) {
	int i = 0;

	bool no_move = true;
	for (; i < ped_list.size(); i++) {
		if (ped_list[i].id == agent.id) {
			logv << "[UpdatePedState] updating agent " << agent.id << endl;
			TrackVel(ped_list[i], agent, no_move, doPrint);
			TracPos(ped_list[i], agent, doPrint);
			ped_list[i].time_stamp = agent.time_stamp;
			break;
		}
	}

	if (i == ped_list.size()) { // new agent
		no_move = false;
		logv << "[UpdatePedState] updating new agent " << agent.id << endl;
		ped_list.push_back(agent);
		if (!use_vel_from_summit) {
			ped_list.back().vel.x = 0.01;
			ped_list.back().vel.y = 0.01;
		} else
			ped_list.back().vel = agent.vel;
		ped_list.back().time_stamp = agent.time_stamp;
	}

	if (no_move) {
		cout << __FUNCTION__ << " no_move agent " << agent.id << " caught: vel "
				<< ped_list[i].vel.x << " " << ped_list[i].vel.y << endl;
	}
}

void WorldStateTracker::UpdatePedPaths(const Pedestrian& agent, bool doPrint) {
	for (int i = 0; i < ped_list.size(); i++) {
		if (ped_list[i].id == agent.id) {
			logv << "[UpdatePedPaths] updating agent paths" << agent.id << endl;

			TracIntention(ped_list[i], agent, doPrint);
			TracCrossDirs(ped_list[i], agent, doPrint);
			break;
		}
	}
}

void WorldStateTracker::RemoveAgents() {
	ped_list.resize(0);
	veh_list.resize(0);
	model.id_map_num_paths.clear();
	model.id_map_paths.clear();
}

void WorldStateTracker::UpdateCar(const CarStruct car) {
	carpos = car.pos;
	carvel = car.vel;
	car_heading_dir = car.heading_dir;
	ValidateCar(__FUNCTION__);
}

void WorldStateTracker::ValidateCar(const char* func) {
	if (car_odom_heading == -10) // initial value
		return;
	if (fabs(car_heading_dir - car_odom_heading) > 0.1
			&& fabs(car_heading_dir - car_odom_heading) < 2 * M_PI - 0.1) {
		ERR(string_sprintf(
			"%s: car_heading in stateTracker different from odom: %f, %f",
			func, car_heading_dir, car_odom_heading));
	}
}

bool WorldStateTracker::Emergency() {
	double mindist = numeric_limits<double>::infinity();
	for (auto& agent : ped_list) {
		COORD p(agent.w, agent.h);
		double d = COORD::EuclideanDistance(carpos, p);
		if (d < mindist)
			mindist = d;
	}
	cout << "Emergency mindist = " << mindist << endl;
	return (mindist < 0.5);
}

void WorldStateTracker::UpdateVel(double vel) {
	carvel = vel;
}

vector<WorldStateTracker::AgentDistPair> WorldStateTracker::GetSortedAgents(
		bool doPrint) {
	// sort agents
	vector<AgentDistPair> sorted_agents;

	if (doPrint)
		cout << "[GetSortedAgents] state tracker agent_list size "
				<< ped_list.size() + veh_list.size() << endl;

	for (auto& p : ped_list) {
		COORD cp(p.w, p.h);
		float dist = COORD::EuclideanDistance(cp, carpos);

		COORD ped_dir = COORD(p.w, p.h) - carpos;
		COORD car_dir = COORD(cos(car_heading_dir), sin(car_heading_dir));
		double proj = (ped_dir.x * car_dir.x + ped_dir.y * car_dir.y)
				/ ped_dir.Length();
		dist += (int)(proj > 0.6) * -2.0 + (int)(proj < -0.7) * 2.0;
		sorted_agents.push_back(AgentDistPair(dist, &p));

		if (doPrint)
			cout << "[GetSortedAgents] agent id:" << p.id << endl;
	}

	for (auto& veh : veh_list) {
		COORD cp(veh.w, veh.h);
		float dist = COORD::EuclideanDistance(cp, carpos);

		COORD ped_dir = COORD(veh.w, veh.h) - carpos;
		COORD car_dir = COORD(cos(car_heading_dir), sin(car_heading_dir));
		double proj = (ped_dir.x * car_dir.x + ped_dir.y * car_dir.y)
				/ ped_dir.Length();
		dist += (int)(proj > 0.6) * -2.0 + (int)(proj < -0.7) * 2.0;
		sorted_agents.push_back(AgentDistPair(dist, &veh));

		if (doPrint)
			cout << "[GetSortedAgents] veh id:" << veh.id << endl;
	}

	sort(sorted_agents.begin(), sorted_agents.end(),
			[](const AgentDistPair& a, const AgentDistPair& b) -> bool {
				return a.first < b.first;
			});

	return sorted_agents;
}

void WorldStateTracker::SetPomdpCar(CarStruct& car) {
	car.pos = carpos;
	car.vel = carvel;
	car.heading_dir = /*0*/car_heading_dir;
}

void WorldStateTracker::SetPomdpPed(AgentStruct& agent, const Agent& src) {
	agent.pos.x = src.w;
	agent.pos.y = src.h;
	agent.id = src.id;
	agent.type = src.type();
	agent.vel = src.vel;
	agent.speed = src.vel.Length();
	agent.intention = -1; // which path to take
	agent.pos_along_path = 0.0; // travel distance along the path
	agent.cross_dir = fetch_cross_dir(src);
	auto bb = fetch_bounding_box(src);
	agent.heading_dir = fetch_heading_dir(src);
	model.CalBBExtents(agent, bb, agent.heading_dir);
}

PomdpState WorldStateTracker::GetPomdpState() {
	auto sorted_agents = GetSortedAgents();

	PomdpState pomdpState;
	SetPomdpCar(pomdpState.car);

	pomdpState.num = min((int) sorted_agents.size(), ModelParams::N_PED_IN); //current_state.num = num_of_peds_world;
	for (int i = 0; i < pomdpState.num; i++) {
		const auto& agent = *(sorted_agents[i].second);
		SetPomdpPed(pomdpState.agents[i], agent);
	}

	return pomdpState;
}

PomdpStateWorld WorldStateTracker::GetPomdpWorldState() {
	auto sorted_agents = GetSortedAgents();

	PomdpStateWorld worldState;
	SetPomdpCar(worldState.car);
	worldState.num = min((int) sorted_agents.size(), ModelParams::N_PED_WORLD); //current_state.num = num_of_peds_world;

	for (int i = 0; i < worldState.num; i++) {
		const auto& agent = *(sorted_agents[i].second);
		SetPomdpPed(worldState.agents[i], agent);
	}

	return worldState;
}

void AgentBelief::ResetBelief(int new_size) {
	reset = true;
	double accum_prob = 0;

	if (new_size != prob_modes_goals[0].size()) {
		for (auto& prob_goals : prob_modes_goals) {
			logi << "Resizing prob_goals from " << prob_goals.size() << " to "
					<< new_size << endl;
			prob_goals.resize(new_size);
		}
	}

	// initialize and normalize distribution
	for (auto& prob_goals : prob_modes_goals) {
		std::fill(prob_goals.begin(), prob_goals.end(), 1.0);
		accum_prob += accumulate(prob_goals.begin(), prob_goals.end(), 0);
	}
	if(accum_prob == 0)
		ERR("accum_prob == 0");
	for (auto& prob_goals : prob_modes_goals) {
		for (auto& prob : prob_goals)
			prob = prob / accum_prob;
	}
}

void WorldBeliefTracker::UpdateCar() {
	car.pos = stateTracker.carpos;
	car.vel = stateTracker.carvel;
	car.heading_dir = stateTracker.car_heading_dir;
}

void WorldBeliefTracker::Update() {
	auto sorted_agents = stateTracker.GetSortedAgents();

	if (logging::level() >= logging::VERBOSE)
		stateTracker.Text(sorted_agents);

	map<int, const Agent*> cur_agent_map;
	for (WorldStateTracker::AgentDistPair& dp : sorted_agents) {
		auto p = dp.second;
		cur_agent_map[p->id] = p;
	}

	vector<int> agents_to_remove;
	for(const auto& p: agent_beliefs) {
		if (cur_agent_map.find(p.first) == cur_agent_map.end()) {
			logi << "agent "<< p.first << " disappeared" << endl;
			agents_to_remove.push_back(p.first); // erasing element here would trigger error.
		}
	}
	for(const auto& i: agents_to_remove) {
		agent_beliefs.erase(i);
	}

	for (auto& dp : sorted_agents) {
		auto& agent = *dp.second;
		if (agent_beliefs.find(agent.id) != agent_beliefs.end())
			if (agent.reset_intention)
				agent_beliefs[agent.id].ResetBelief(model.GetNumIntentions(agent.id));
			else
				agent_beliefs[agent.id].reset = false;
	}

	if (logging::level() >= logging::VERBOSE)
		Text(agent_beliefs);

	model.PrepareAttentiveAgentMeanDirs(agent_beliefs, car);
	if (logging::level() >= logging::VERBOSE)
		model.PrintMeanDirs(agent_beliefs, cur_agent_map);

	UpdateCar();

	if (logging::level() >= logging::VERBOSE)
		stateTracker.model.PrintPathMap();
	for (auto& kv : agent_beliefs) {
		if (cur_agent_map.find(kv.first) == cur_agent_map.end())
			ERR(string_sprintf("id %d agent_beliefs does not exist any more in newagents", kv.first))
		model.UpdatePedBelief(kv.second, *cur_agent_map[kv.first]);
	}

	for (const auto& kv : cur_agent_map) {
		auto& p = *kv.second;
		if (agent_beliefs.find(p.id) == agent_beliefs.end()) { // new agent
			agent_beliefs[p.id] = model.InitPedBelief(p);
		}
	}

	sorted_beliefs.clear();
	for (const auto& dp : sorted_agents) {
		auto& p = *dp.second;
		sorted_beliefs.push_back(&agent_beliefs[p.id]);
	}

	if (logging::level() >= logging::DEBUG)
		for (auto it = agent_beliefs.begin(); it != agent_beliefs.end(); it++) {
			auto & agent = it->second;
			agent.Text();
		}
	cur_time_stamp = Globals::ElapsedTime();
	return;
}

int AgentBelief::SampleGoal() const {
	double r = Random::RANDOM.NextDouble();

	for (auto prob_goals : prob_modes_goals) {
		int i = 0;
		for (int prob : prob_goals) {
			r -= prob;
			if (r <= 0)
				return i;
			i++;
		}
	}

	return prob_modes_goals[0].size() - 1;
}

void AgentBelief::SampleGoalMode(int& goal, int& mode,
		int use_att_mode) const {
	double r = Random::RANDOM.NextDouble();

	if (use_att_mode == 0) {
		bool done = false;

		double total_prob = 0;
		for (auto& goal_probs : prob_modes_goals) {
			for (auto p : goal_probs)
				total_prob += p;
		}
		if(total_prob == 0)
			ERR("total_prob == 0");

		r = r * total_prob;

		for (int ped_type = 0; ped_type < prob_modes_goals.size(); ped_type++) {
			auto& goal_probs = prob_modes_goals[ped_type];
			for (int ped_goal = 0; ped_goal < goal_probs.size(); ped_goal++) {
				r -= prob_modes_goals[ped_type][ped_goal];
				if (r <= 0.001) {
					goal = ped_goal;
					mode = ped_type;
					done = true;
					break;
				}
			}
			if (done)
				break;
		}

		if (r > 0) {
			logv << "[WARNING]: [AgentBelief::SampleGoalMode] execess probability "
				 << r << endl;
			goal = 0;
			mode = 0;
		}

		if (goal > 10) {
			cout << "the rest r = " << r << endl;
		}
	} else {
		int ped_type = (use_att_mode <= 1) ? AGENT_DIS : AGENT_ATT;
		auto& goal_probs = prob_modes_goals[ped_type];
		double total_prob = 0;
		for (auto p : goal_probs)
			total_prob += p;
		if(total_prob == 0)
			ERR("total_prob == 0");

		r = r * total_prob;
		for (int ped_goal = 0; ped_goal < goal_probs.size(); ped_goal++) {
			r -= prob_modes_goals[ped_type][ped_goal];
			if (r <= 0) {
				goal = ped_goal;
				mode = ped_type;
				break;
			}
		}
	}
}

int AgentBelief::MaxLikelyIntention() const {
	double ml = 0;
	int mode = AGENT_DIS;
	auto& prob_goals = prob_modes_goals[mode];
	int mi = prob_goals.size() - 1;
	for (int i = 0; i < prob_goals.size(); i++) {
		if (prob_goals[i] > ml && prob_goals[i] > 0.5) {
			ml = prob_goals[i];
			mi = i;
		}
	}
	return mi;
}

void WorldBeliefTracker::printBelief() const {
	logi << sorted_beliefs.size() << " entries in sorted_beliefs:" << endl;
	int num = 0;
	for (int i = 0; i < sorted_beliefs.size() && i < ModelParams::N_PED_IN;
			i++) {
		auto& p = *sorted_beliefs[i];
		if (COORD::EuclideanDistance(p.pos, car.pos)
				<= ModelParams::LASER_RANGE) {
			cout << "agent belief " << p.id << ": ";

			for (auto& prob_goals : p.prob_modes_goals) {
				for (int g = 0; g < prob_goals.size(); g++)
					cout << " " << prob_goals[g];
				cout << endl;
			}
		}
	}
}

PomdpState WorldBeliefTracker::Text() const {
	if (logging::level() >= logging::VERBOSE) {
		for (int i = 0;
				i < sorted_beliefs.size() && i < min(20, ModelParams::N_PED_IN);
				i++) {
			auto& p = *sorted_beliefs[i];
			cout << "[WorldBeliefTracker::text] " << this << "->p:" << &p
					<< endl;
			cout << " sorted agent " << p.id << endl;

			cout << "-prob_modes_goals: " << p.prob_modes_goals << endl;
		}
	}
}

void WorldBeliefTracker::Text(
		const std::map<int, AgentBelief>& agent_bs) const {
	if (logging::level() >= logging::VERBOSE) {
		cout << "=> Agent beliefs: " << endl;
		for (auto itr = agent_bs.begin(); itr != agent_bs.end(); ++itr) {
			int id = itr->first;
			auto& b = itr->second;
			fprintf(stderr,
					"==> id / type / pos / vel / heading / reset / cross : %d / %d / (%f %f) / (%f %f) / %f / %d / %d \n",
					b.id, b.type, b.pos.x, b.pos.y, b.vel.x, b.vel.y,
					b.heading_dir, b.reset, b.cross_dir);
		}
	}
}

void WorldBeliefTracker::ValidateCar(const char* func) {
	if (stateTracker.car_odom_heading == -10) // initial value
		return;
	if (fabs(car.heading_dir - stateTracker.car_odom_heading) > 0.1
			&& fabs(car.heading_dir - stateTracker.car_odom_heading)
					< 2 * M_PI - 0.1) {
		ERR(
				string_sprintf(
						"%s: car_heading in stateTracker different from odom: %f, %f",
						func, car.heading_dir, stateTracker.car_odom_heading));
	}
}

PomdpState WorldBeliefTracker::Sample(bool predict, int use_att_mode) {
	ValidateCar(__FUNCTION__);

	PomdpState s;
	s.car = car;
	s.num = 0;

	for (int i = 0; i < sorted_beliefs.size() && i < ModelParams::N_PED_IN;
			i++) {
		auto& p = *sorted_beliefs[i];

		if (p.type >= AgentType::num_values)
			ERR(string_sprintf("non-initialized type in state: %d", p.type));

		if (COORD::EuclideanDistance(p.pos, car.pos)
				< ModelParams::LASER_RANGE) {
			s.agents[s.num].pos = p.pos;

			assert(p.prob_modes_goals.size() == 2);
			for (auto& prob_goals : p.prob_modes_goals) {
				if (prob_goals.size() != model.GetNumIntentions(p.id)) {
					DEBUG(string_sprintf(
							"prob_goals.size()!=model.GetNumIntentions(p.id): %d, %d, id %d",
							prob_goals.size(),
							model.GetNumIntentions(p.id), p.id));
					p.ResetBelief(model.GetNumIntentions(p.id));
				}
			}
			p.SampleGoalMode(s.agents[s.num].intention, s.agents[s.num].mode,
					use_att_mode);
			model.ValidateIntention(p.id, s.agents[s.num].intention,
					__FUNCTION__, __LINE__);

			s.agents[s.num].id = p.id;
			s.agents[s.num].vel = p.vel;
			s.agents[s.num].speed = p.speed;
			s.agents[s.num].pos_along_path = 0; // assuming that paths are up to date here
			s.agents[s.num].cross_dir = p.cross_dir;
			s.agents[s.num].type = p.type;
			s.agents[s.num].heading_dir = p.heading_dir;
			model.CalBBExtents(p, s.agents[s.num]);
			s.num++;
		}
	}

	s.time_stamp = cur_time_stamp;

	if (predict) {
		PomdpState predicted_s = PredictPedsCurVel(&s, cur_acc, cur_steering);
		return predicted_s;
	}

	return s;
}

vector<PomdpState> WorldBeliefTracker::Sample(int num, bool predict,
		int use_att_mode) {

	if (DESPOT::Debug_mode)
		Random::RANDOM.seed(0);

	vector<PomdpState> particles;
	logv << "[WorldBeliefTracker::sample] Sampling" << endl;

	for (int i = 0; i < num; i++) {
		particles.push_back(Sample(predict, use_att_mode));
	}

	cout << "Num agents for planning: " << particles[0].num << endl;

	return particles;
}

vector<AgentStruct> WorldBeliefTracker::PredictAgents() {
	vector<AgentStruct> prediction;

	for (const auto ptr : sorted_beliefs) {
		const auto& p = *ptr;
		double dist = COORD::EuclideanDistance(p.pos, car.pos);
		int step = (p.speed + car.vel > 1e-5) ?
					int(dist / (p.speed + car.vel) * ModelParams::CONTROL_FREQ) :
					100000;

		int intention_id = p.MaxLikelyIntention();
		AgentStruct agent_pred(p.pos, intention_id, p.id);
		agent_pred.vel = p.vel;
		agent_pred.speed = p.speed;
		agent_pred.type = p.type;

		for (int i = 0; i < 4; i++) {
			AgentStruct agent = agent_pred;

			if (model.goal_mode == "cur_vel") {
				model.AgentStepCurVel(agent, step + i);
			} else if (model.goal_mode == "path") {
				model.AgentStepPath(agent, step + i);
			}

			prediction.push_back(agent);
		}
	}

	return prediction;
}

PomdpState WorldBeliefTracker::PredictPedsCurVel(PomdpState* ped_state,
		double acc, double steering) {

	PomdpState predicted_state = *ped_state;
	model.RobStepCurVel(predicted_state.car);
	for (int i = 0; i < predicted_state.num; i++) {
		auto & p = predicted_state.agents[i];
		model.AgentStepCurVel(p);
	}

	predicted_state.time_stamp = ped_state->time_stamp
			+ 1.0 / ModelParams::CONTROL_FREQ;

	return predicted_state;
}

double GenerateGaussian(double rNum) {
	if (FIX_SCENARIO != 1 && !CPUDoPrint)
		rNum = QuickRandom::RandGeneration(rNum);
	double result = sqrt(-2 * log(rNum));
	if (FIX_SCENARIO != 1 && !CPUDoPrint)
		rNum = QuickRandom::RandGeneration(rNum);

	result *= cos(2 * M_PI * rNum);
	return result;
}

void WorldBeliefTracker::PrintState(const State& s, ostream& out) const {
	const PomdpState & state = static_cast<const PomdpState&>(s);
	const COORD& carpos = state.car.pos;

	out << "Rob Pos: " << carpos.x << " " << carpos.y << endl;
	out << "Rob heading direction: " << state.car.heading_dir << endl;
	for (int i = 0; i < state.num; i++) {
		out << "Ped Pos: " << state.agents[i].pos.x << " "
				<< state.agents[i].pos.y << endl;
		out << "Goal: " << state.agents[i].intention << endl;
		out << "id: " << state.agents[i].id << endl;
	}
	out << "Vel: " << state.car.vel << endl;
	out << "num  " << state.num << endl;
	double min_dist = COORD::EuclideanDistance(carpos, state.agents[0].pos);
	out << "MinDist: " << min_dist << endl;
}

void WorldModel::ValidateIntention(int agent_id, int intention_id,
		const char* msg, int line) {
	if (intention_id >= GetNumIntentions(agent_id)) {
		int num_path = NumPaths(agent_id);
		int num_intention = GetNumIntentions(agent_id);
		ERR(
				string_sprintf(
						"%s@%d: Intention ID excess # intentions: %d, # intention = %d, # path = %d",
						msg, line, intention_id, num_intention, num_path));
	}
}

void WorldModel::GammaSimulateAgents(AgentStruct agents[], int num_agents,
		CarStruct& car) {

	int threadID = GetThreadID();

	// Construct a new set of agents every time
	traffic_agent_sim_[threadID]->clearAllAgents();

	// adding pedestrians
	for (int i = 0; i < num_agents; i++) {
		bool frozen_agent = (agents[i].mode == AGENT_DIS);
		if (agents[i].type == AgentType::car) {
			traffic_agent_sim_[threadID]->addAgent(default_car_, i,
					frozen_agent);
		} else if (agents[i].type == AgentType::ped) {
			traffic_agent_sim_[threadID]->addAgent(default_ped_, i,
					frozen_agent);
		} else {
			traffic_agent_sim_[threadID]->addAgent(default_bike_, i,
					frozen_agent);
		}

		traffic_agent_sim_[threadID]->setAgentPosition(i,
				RVO::Vector2(agents[i].pos.x, agents[i].pos.y));
		RVO::Vector2 agt_heading(cos(agents[i].heading_dir),
				sin(agents[i].heading_dir));
		traffic_agent_sim_[threadID]->setAgentHeading(i, agt_heading);
		traffic_agent_sim_[threadID]->setAgentVelocity(i,
				RVO::Vector2(agents[i].vel.x, agents[i].vel.y));

		int intention_id = agents[i].intention;
		ValidateIntention(agents[i].id, intention_id, __FUNCTION__, __LINE__);
		if (IsStopIntention(intention_id, agents[i].id)) { /// stop intention
			traffic_agent_sim_[threadID]->setAgentPrefVelocity(i,
					RVO::Vector2(0.0f, 0.0f));
		} else {
			auto goal_pos = GetGoalPos(agents[i], intention_id);
			RVO::Vector2 goal(goal_pos.x, goal_pos.y);
			if (RVO::abs(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) < 0.5) {
				// Agent is within 0.5 meter of its goal, set preferred velocity to zero
				traffic_agent_sim_[threadID]->setAgentPrefVelocity(i,
						RVO::Vector2(0.0f, 0.0f));
			} else {
				double pref_speed = 0.0;
				pref_speed = agents[i].speed;
				traffic_agent_sim_[threadID]->setAgentPrefVelocity(i,
						normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) * pref_speed);
			}
		}

		traffic_agent_sim_[threadID]->setAgentBoundingBoxCorners(i,
				GetBoundingBoxCorners(agents[i]));
	}

	// adding car as a "special" pedestrian
	AddEgoGammaAgent(num_agents, car);

	traffic_agent_sim_[threadID]->doStep();
}

COORD WorldModel::GetGammaVel(AgentStruct& agent, int i) {
	int threadID = GetThreadID();
	assert(agent.mode == AGENT_ATT);

	COORD new_pos;
	new_pos.x = traffic_agent_sim_[threadID]->getAgentPosition(i).x(); // + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(agent).x() - agents[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
	new_pos.y = traffic_agent_sim_[threadID]->getAgentPosition(i).y(); // + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(agent).y() - agents[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

	return (new_pos - agent.pos) * freq;
}

void WorldModel::AgentApplyGammaVel(AgentStruct& agent, COORD& rvo_vel) {
	COORD old_pos = agent.pos;
	double rvo_speed = rvo_vel.Length();
	if (agent.type == AgentType::car) {
		rvo_vel.AdjustLength(PURSUIT_LEN);
		COORD pursuit_point = agent.pos + rvo_vel;
		double steering = PControlAngle<AgentStruct>(agent, pursuit_point);
		BicycleModel(agent, steering, rvo_speed);
	} else if (agent.type == AgentType::ped) {
		agent.pos = agent.pos + rvo_vel * (1.0 / freq);
	}

	if (!IsStopIntention(agent.intention, agent.id)
			&& !IsCurVelIntention(agent.intention, agent.id)) {
		auto& path = PathCandidates(agent.id)[agent.intention];
		agent.pos_along_path = path.Nearest(agent.pos);
	}
	agent.vel = (agent.pos - old_pos) * freq;
	agent.speed = agent.vel.Length();
}

void WorldModel::GammaAgentStep(PomdpStateWorld& state, Random& random) {
	GammaAgentStep(state.agents, random, state.num, state.car);
}

void WorldModel::GammaAgentStep(AgentStruct agents[], Random& random,
		int num_agents, CarStruct car) {
	GammaSimulateAgents(agents, num_agents, car);

	for (int i = 0; i < num_agents; ++i) {
		auto& agent = agents[i];
		if (agent.mode == AGENT_ATT) {
			COORD rvo_vel = GetGammaVel(agent, i);
			AgentApplyGammaVel(agent, rvo_vel);
			if (use_noise_in_rvo) {
				agent.pos.x += random.NextGaussian()
						* ModelParams::NOISE_PED_POS / freq;
				agent.pos.y += random.NextGaussian()
						* ModelParams::NOISE_PED_POS / freq;
			}
		}
	}
}

void WorldModel::GammaAgentStep(AgentStruct agents[], double& random,
		int num_agents, CarStruct car) {
	GammaSimulateAgents(agents, num_agents, car);

	for (int i = 0; i < num_agents; ++i) {
		auto& agent = agents[i];
		if (agent.mode == AGENT_ATT) {
			COORD rvo_vel = GetGammaVel(agent, i);
			AgentApplyGammaVel(agent, rvo_vel);
			if (use_noise_in_rvo) {
				double rNum = GenerateGaussian(random);
				agent.pos.x += rNum * ModelParams::NOISE_PED_POS / freq;
				rNum = GenerateGaussian(rNum);
				agent.pos.y += rNum * ModelParams::NOISE_PED_POS / freq;
			}
		}
	}
}

COORD WorldModel::DistractedPedMeanDir(AgentStruct& agent, int intention_id) {
	COORD dir(0, 0);
	const COORD& goal = GetGoalPos(agent, intention_id);
	if (IsStopIntention(intention_id, agent.id)) {
		return dir;
	}

	MyVector goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);

	dir.x = goal_vec.dw;
	dir.y = goal_vec.dh;

	return dir;
}

COORD WorldModel::AttentivePedMeanDir(int agent_id, int intention_id) {
	return ped_mean_dirs[agent_id][intention_id];
}

#include "Vector2.h"

void WorldModel::AddEgoGammaAgent(int id_in_sim, CarStruct& car) {
	int threadID = GetThreadID();

	double car_x, car_y, car_yaw;
	car_x = car.pos.x;
	car_y = car.pos.y;
	car_yaw = car.heading_dir;

	traffic_agent_sim_[threadID]->addAgent(default_car_, id_in_sim);
	traffic_agent_sim_[threadID]->setAgentPosition(id_in_sim,
			RVO::Vector2(car_x, car_y));
	RVO::Vector2 agt_heading(cos(car_yaw), sin(car_yaw));
	traffic_agent_sim_[threadID]->setAgentHeading(id_in_sim, agt_heading);
	traffic_agent_sim_[threadID]->setAgentVelocity(id_in_sim,
			car.vel * agt_heading);
	traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim,
			car.vel * agt_heading); // assume that other agents do not know the ego-vehicle's intention and that they also don't infer the intention

	// set agent bounding box corners
	RVO::Vector2 sideward_vec = RVO::Vector2(-agt_heading.y(), agt_heading.x()); // rotate 90 degree counter-clockwise
	traffic_agent_sim_[threadID]->setAgentBoundingBoxCorners(id_in_sim,
			GetBoundingBoxCorners(agt_heading, sideward_vec,
					RVO::Vector2(car_x, car_y), ModelParams::CAR_FRONT, ModelParams::CAR_WIDTH/2.0));
}

void WorldModel::AddGammaAgent(AgentBelief& b, int id_in_sim) {
	int threadID = GetThreadID();

	double car_x, car_y, car_yaw, car_speed;
	car_x = b.pos.x;
	car_y = b.pos.y;
	car_yaw = b.heading_dir;
	car_speed = b.speed;

	if (b.type == AgentType::car)
		traffic_agent_sim_[threadID]->addAgent(default_car_, id_in_sim);
	else if (b.type == AgentType::ped)
		traffic_agent_sim_[threadID]->addAgent(default_ped_, id_in_sim);
	else
		traffic_agent_sim_[threadID]->addAgent(default_bike_, id_in_sim);

	traffic_agent_sim_[threadID]->setAgentPosition(id_in_sim,
			RVO::Vector2(car_x, car_y));
	RVO::Vector2 agt_heading(cos(car_yaw), sin(car_yaw));
	traffic_agent_sim_[threadID]->setAgentHeading(id_in_sim, agt_heading);
	traffic_agent_sim_[threadID]->setAgentVelocity(id_in_sim,
			car_speed * agt_heading);
	// assume that other agents do not know the vehicle's intention
	// and that they also don't infer the intention
	traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim,
			car_speed * agt_heading);
	// rotate 90 degree counter-clockwise
	RVO::Vector2 sideward_vec = RVO::Vector2(-agt_heading.y(), agt_heading.x());
	double bb_x, bb_y;
	CalBBExtents(b.pos, b.heading_dir, b.bb, bb_x, bb_y);
	DEBUG(string_sprintf("bb_x=%f, bb_y=%f", bb_x, bb_y));
	traffic_agent_sim_[threadID]->setAgentBoundingBoxCorners(id_in_sim,
			GetBoundingBoxCorners(agt_heading, sideward_vec,
					RVO::Vector2(car_x, car_y), bb_y, bb_x));
}

void WorldModel::PrepareAttentiveAgentMeanDirs(
		std::map<int, AgentBelief> agents, CarStruct& car) {
	int num_agents = agents.size();
	logi << "num_agents in belief tracker: " << num_agents << endl;

	if (num_agents == 0)
		return;

	for (auto it = agents.begin(); it != agents.end(); it++) {
		auto & agent = it->second;
		EnsureMeanDirExist(agent.id);
	}

	if (initial_update_step){
		initial_update_step = false;

		for (auto it = agents.begin(); it != agents.end(); it++) {
			auto & agent = it->second;
			int id = agent.id;
			for (int intention_id = 0; intention_id < GetNumIntentions(id);	intention_id++)
				ped_mean_dirs[id][intention_id] = COORD(0.0, 0.0);
		}
	}
	else {
		int threadID = GetThreadID();

		traffic_agent_sim_[threadID]->clearAllAgents();

		std::vector<int> agent_ids;
		agent_ids.resize(num_agents);
		int i = 0;
		for (auto it = agents.begin(); it != agents.end(); it++) {
			auto & agent = it->second;
			AddGammaAgent(agent, i);
			agent_ids[i] = agent.id;
			i++;
		}
		AddEgoGammaAgent(num_agents, car);

		traffic_agent_sim_[threadID]->doPreStep();

		for (size_t i = 0; i < num_agents; ++i) {
			int id = agent_ids[i];
			// For each ego agent
			for (int intention_id = 0; intention_id < GetNumIntentions(id);
					intention_id++) {
				RVO::Vector2 ori_pos =
						traffic_agent_sim_[threadID]->getAgentPosition(i);
				// Leave other pedestrians to have default preferred velocity
				ValidateIntention(id, intention_id, __FUNCTION__, __LINE__);
				if (IsStopIntention(intention_id, id)) { /// stop intention
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(i,
							RVO::Vector2(0.0f, 0.0f));
				} else {
					auto goal_pos = GetGoalPos(agents[id], intention_id);
					RVO::Vector2 goal(goal_pos.x, goal_pos.y);
					if (abs(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) < 0.5) {
						// Agent is within 0.5 meters of its goal, set preferred velocity to zero
						traffic_agent_sim_[threadID]->setAgentPrefVelocity(i,
								RVO::Vector2(0.0f, 0.0f));
					} else {
						// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
						double spd = agents[id].speed;
						traffic_agent_sim_[threadID]->setAgentPrefVelocity(i,
								normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) * spd);
					}
				}

				traffic_agent_sim_[threadID]->doStepForOneAgent(i);

				COORD dir;
				dir.x = traffic_agent_sim_[threadID]->getAgentPosition(i).x()
						- agents[id].pos.x;
				dir.y = traffic_agent_sim_[threadID]->getAgentPosition(i).y()
						- agents[id].pos.y;

				logd << "[PrepareAttentiveAgentMeanDirs] ped_mean_dirs len="
						<< ped_mean_dirs.size() << " intention_list len="
						<< ped_mean_dirs[id].size() << "\n";
				ped_mean_dirs[id][intention_id] = dir;

				// reset back agent position
				traffic_agent_sim_[threadID]->setAgentPosition(i, ori_pos);
			}
		}
	}
}

void WorldModel::PrintMeanDirs(std::map<int, AgentBelief> old_agents,
		map<int, const Agent*>& curr_agents) {
	if (logging::level() >= logging::VERBOSE) {
		int num_agents = old_agents.size();

		int count = 0;
		for (std::map<int, const Agent*>::iterator it = curr_agents.begin();
				it != curr_agents.end(); ++it) {

			if (count == 6)
				break;

			int agent_id = it->first;

			if (old_agents.find(agent_id) == old_agents.end())
				continue;

			cout << "agent  " << agent_id << endl;
			auto& cur_agent = *it->second;
			auto& old_agent = old_agents[agent_id];

			cout << "prev pos: " << old_agent.pos.x << "," << old_agent.pos.y
					<< endl;
			cout << "cur pos: " << cur_agent.w << "," << cur_agent.h << endl;

			COORD dir = COORD(cur_agent.w, cur_agent.h) - old_agent.pos;

			cout << "dir: " << endl;
			for (int intention_id = 0;
					intention_id < GetNumIntentions(cur_agent.id);
					intention_id++) {
				cout << intention_id << "," << dir.x << "," << dir.y << " ";
			}
			cout << endl;

			cout << "Meandir: " << endl;
			for (int intention_id = 0;
					intention_id < GetNumIntentions(cur_agent.id);
					intention_id++) {
				cout << intention_id << ","
						<< ped_mean_dirs[agent_id][intention_id].x << ","
						<< ped_mean_dirs[agent_id][intention_id].y << " ";
			}
			cout << endl;

			cout << "Att probs: " << endl;
			for (int intention_id = 0;
					intention_id < GetNumIntentions(cur_agent.id);
					intention_id++) {
				logv << "agent, goal, ped_mean_dirs.size() =" << cur_agent.id
						<< " " << intention_id << " " << ped_mean_dirs.size()
						<< endl;
				double prob = AgentMoveProb(old_agent, cur_agent, intention_id,
						AGENT_ATT);

				cout << "prob: " << intention_id << "," << prob << endl;
			}
			cout << endl;

			count++;
		}
	}
}

bool WorldModel::CheckCarWithVehicle(const CarStruct& car,
		const AgentStruct& veh, int flag) {
	COORD tan_dir(-sin(veh.heading_dir), cos(veh.heading_dir)); // along_dir rotates by 90 degree counter-clockwise
	COORD along_dir(cos(veh.heading_dir), sin(veh.heading_dir));

	COORD test;

	bool result = false;
	double veh_bb_extent_x = veh.bb_extent_x;
	double veh_bb_extent_y = veh.bb_extent_y;

	double car_bb_extent_x = ModelParams::CAR_WIDTH / 2.0;
	double car_bb_extent_y = ModelParams::CAR_FRONT + 0.5;

	test = veh.pos + tan_dir * veh_bb_extent_x + along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos - tan_dir * veh_bb_extent_x + along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos + tan_dir * veh_bb_extent_x - along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos - tan_dir * veh_bb_extent_x - along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}

	tan_dir = COORD(-sin(car.heading_dir), cos(car.heading_dir));
	along_dir = COORD(cos(car.heading_dir), sin(car.heading_dir));

	test = car.pos + tan_dir * (car_bb_extent_x) + along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos - tan_dir * (car_bb_extent_x) + along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos + tan_dir * (car_bb_extent_x) - along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos - tan_dir * (car_bb_extent_x) - along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}

	if (flag == 1 && result) {
		cout << "collision point";
		cout << " " << veh.pos.x << " " << veh.pos.y;
		test = veh.pos + tan_dir * veh_bb_extent_x
				+ along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos - tan_dir * veh_bb_extent_x
				+ along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos + tan_dir * veh_bb_extent_x
				- along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos - tan_dir * veh_bb_extent_x
				- along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;

		cout << " " << car.pos.x << " " << car.pos.y;
		test = car.pos + tan_dir * (car_bb_extent_x)
				+ along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos - tan_dir * (car_bb_extent_x)
				+ along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos + tan_dir * (car_bb_extent_x)
				- along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos - tan_dir * (car_bb_extent_x)
				- along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y << endl;
	}

	return result;
}

bool WorldModel::CheckCarWithVehicleReal(const CarStruct& car,
		const AgentStruct& veh, int flag) {

	COORD tan_dir(-sin(veh.heading_dir), cos(veh.heading_dir)); // along_dir rotates by 90 degree counter-clockwise
	COORD along_dir(cos(veh.heading_dir), sin(veh.heading_dir));

	COORD test;

	bool result = false;
	double veh_bb_extent_x = veh.bb_extent_x * 0.95;
	double veh_bb_extent_y = veh.bb_extent_y * 0.95;

	double car_bb_extent_x = ModelParams::CAR_WIDTH / 2.0;
	double car_bb_extent_y = ModelParams::CAR_FRONT;

	test = veh.pos + tan_dir * veh_bb_extent_x + along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos - tan_dir * veh_bb_extent_x + along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos + tan_dir * veh_bb_extent_x - along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos - tan_dir * veh_bb_extent_x - along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}

	tan_dir = COORD(-sin(car.heading_dir), cos(car.heading_dir));
	along_dir = COORD(cos(car.heading_dir), sin(car.heading_dir));

	test = car.pos + tan_dir * (car_bb_extent_x) + along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos - tan_dir * (car_bb_extent_x) + along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos + tan_dir * (car_bb_extent_x) - along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos - tan_dir * (car_bb_extent_x) - along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}

	if (flag == 1 && result) {
		cout << "collision point";
		cout << " " << veh.pos.x << " " << veh.pos.y;
		test = veh.pos + tan_dir * veh_bb_extent_x
				+ along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos - tan_dir * veh_bb_extent_x
				+ along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos + tan_dir * veh_bb_extent_x
				- along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos - tan_dir * veh_bb_extent_x
				- along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;

		cout << " " << car.pos.x << " " << car.pos.y;
		test = car.pos + tan_dir * (car_bb_extent_x)
				+ along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos - tan_dir * (car_bb_extent_x)
				+ along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos + tan_dir * (car_bb_extent_x)
				- along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos - tan_dir * (car_bb_extent_x)
				- along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y << endl;
	}

	return result;
}

bool WorldModel::CheckCarWithObsLine(const CarStruct& car, COORD start_point,
		COORD end_point, int flag) {
	const double step = ModelParams::OBS_LINE_STEP;

	double d = COORD::EuclideanDistance(start_point, end_point);
	double dx = (end_point.x - start_point.x) / d;
	double dy = (end_point.y - start_point.y) / d;
	double sx = start_point.x;
	double sy = start_point.y;

	double t = 0, ti = 0;
	while (t < ti + d) {
		double u = t - ti;
		double nx = sx + dx * u;
		double ny = sy + dy * u;

		if (flag == 0) {
			if (::InCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir,
					false))
				return true;
		} else if (flag == 1) {
			if (::InRealCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir,
					false))
				return true;
		}

		if (t == ti + d - 0.01)
			break;
		else
			t = min(t + step, ti + d - 0.01);
	}

	return false;
}

void WorldStateTracker::Text(
		const vector<WorldStateTracker::AgentDistPair>& sorted_agents) const {

	if (logging::level() >= logging::VERBOSE) {
		cout << "=> Sorted_agents:" << endl;

		for (auto& dist_agent_pair : sorted_agents) {
			double dist = dist_agent_pair.first;
			auto& agent = *dist_agent_pair.second;

			fprintf(stderr,
					"==> id / type / pos / vel / reset: %d / %d / (%f %f) / (%f %f) / %d \n",
					agent.id, agent.type(), agent.w, agent.h, agent.vel.x,
					agent.vel.y, agent.reset_intention);
		}
	}
}

void WorldStateTracker::Text(const vector<Pedestrian>& tracked_peds) const {
	if (logging::level() >= logging::VERBOSE) {
		cout << "=> ped_list:" << endl;
		for (auto& agent : tracked_peds) {
			fprintf(stderr,
					"==> id / type / pos / vel / cross / reset: %d / %d / (%f %f) / (%f %f) / %d / %d \n",
					agent.id, agent.type(), agent.w, agent.h, agent.vel.x,
					agent.vel.y, agent.cross_dir, agent.reset_intention);
		}
	}
}

void WorldStateTracker::Text(const vector<Vehicle>& tracked_vehs) const {
	if (logging::level() >= logging::VERBOSE) {
		cout << "=> veh_list:" << endl;
		for (auto& agent : tracked_vehs) {
			fprintf(stderr,
					"==> id / type / pos / vel / heading_dir / reset: %d / %d / (%f %f) / (%f %f) / %f / %d \n",
					agent.id, agent.type(), agent.w, agent.h, agent.vel.x,
					agent.vel.y, agent.heading_dir, agent.reset_intention);
		}
	}
}

void WorldStateTracker::CheckWorldStatus() {
	int alive_count = 0;
	for (auto& walker : ped_list) {
		if (AgentIsUp2Date(walker)) {
			alive_count += 1;
		}
	}

	for (auto& vehicle : veh_list) {
		if (AgentIsUp2Date(vehicle)) {
			alive_count += 1;
		}
	}

	if (alive_count == 0)
		DEBUG("No agent alive in the current scene.");
}
