#include<limits>
#include<cmath>
#include<cstdlib>
#include"WorldModel.h"

#include <despot/GPUcore/thread_globals.h>
#include <despot/core/globals.h>

#include"math_utils.h"
#include"coord.h"

#include <despot/solver/despot.h>
#include <numeric>
#include "ped_pomdp.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

using namespace std;
std::chrono::time_point<std::chrono::system_clock> init_time;

bool use_noise_in_rvo = false;

double cap_angle(double angle){

	while (angle>2*M_PI)
		angle-=2*M_PI;
	while (angle<0)
		angle+=2*M_PI;

	return angle;
}

int ClosestInt(double v){
	if ((v-(int)v)<0.5)
		return (int)v;
	else
		return (int)v+1;
}


int FloorIntRobust(double v){
	if ((v-(int)v)>1-1e-5)
		return (int)v+1;
	else
		return (int)v;
}

WorldModel::WorldModel(): freq(ModelParams::control_freq),
    in_front_angle_cos(cos(ModelParams::IN_FRONT_ANGLE_DEG / 180.0 * M_PI)) {
    goal_file_name_ = "null";

//    InitPedGoals();

	if (DESPOT::Debug_mode)
		ModelParams::NOISE_GOAL_ANGLE = 0.000001;

    init_time = Time::now();
}

void WorldModel::InitPedGoals(){

	logi << "WorldModel::InitPedGoals\n";
     if(goal_file_name_ == "null"){
        std::cout<<"Using default goals"<<std::endl;

       /* goals = { // indian cross 2 larger map
            COORD(3.5, 20.0),
            COORD(-3.5, 20.0), 
            COORD(3.5, -20.0), 
            COORD(-3.5, -20.0),
            COORD(20.0  , 3.5),
            COORD( 20.0 , -3.5), 
            COORD(-20.0 , 3.5), 
            COORD( -20.0, -3.5),
            COORD(-1, -1) // stop
        };*/

        goals = {
		   COORD(/*-2*/-20, 7.5),
		   COORD(10, /*17*//*27*/37),
		   COORD(-1,-1)
		};
    }
    else{

        std::cout<<"Using goal file name " << goal_file_name_ <<std::endl;

        goals.resize(0);

        std::ifstream file;
        file.open(goal_file_name_, std::ifstream::in);

        if(file.fail()){
            std::cout<<"open goal file failed !!!!!!"<<std::endl;
            exit(-1);
        }

        std::string line;
        int goal_num = 0;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            
            double x;
            double y;
            while (iss >> x >>y){
                std::cout << x <<" "<< y<<std::endl;
                goals.push_back(COORD(x, y));
            }

            goal_num++;
            if(goal_num > 99) break;
        }

        file.close();
    }

     // for(int i=0;i<ModelParams::N_PED_WORLD; i++){
    	//  vector<COORD> dirs(goals.size());
    	//  ped_mean_dirs.push_back(dirs);
     // }
}

void WorldModel::InitRVO(){
    if(!Globals::config.use_multi_thread_ )
        //NumThreads=Globals::config.NUM_THREADS;
        Globals::config.NUM_THREADS=1;

    int NumThreads=Globals::config.NUM_THREADS;

    traffic_agent_sim_.resize(NumThreads);
    for(int tid=0; tid<NumThreads;tid++)
    {
        traffic_agent_sim_[tid] = new RVO::RVOSimulator();
        
        // Specify global time step of the simulation.
        traffic_agent_sim_[tid]->setTimeStep(1.0f/ModelParams::control_freq);

        traffic_agent_sim_[tid]->setAgentDefaults(5.0f, 5, 1.5f, 1.5f, PED_SIZE, 2.5f);
    }
    
}

WorldModel::~WorldModel(){
    traffic_agent_sim_.clear();
    traffic_agent_sim_.resize(0);
}

//bool WorldModel::isLocalGoal(const PomdpState& state) {
//    return state.car.dist_travelled > ModelParams::GOAL_TRAVELLED-1e-4 || state.car.pos >= path.size()-1;
//}
//
//bool WorldModel::isLocalGoal(const PomdpStateWorld& state) {
//    return state.car.dist_travelled > ModelParams::GOAL_TRAVELLED || state.car.pos >= path.size()-1;
//}

bool WorldModel::isGlobalGoal(const CarStruct& car) {
    double d = COORD::EuclideanDistance(car.pos, car_goal);
    return (d<ModelParams::GOAL_TOLERANCE);
}

int WorldModel::defaultPolicy(const std::vector<State*>& particles)  {
//	const PomdpState *state=static_cast<const PomdpState*>(particles[0]);
//    double mindist = numeric_limits<double>::infinity();
//    auto& carpos = state->car.pos;
//    double carvel = state->car.vel;
//    // Closest pedestrian in front
//    for (int i=0; i<state->num; i++) {
//		auto& p = state->agents[i];
//		if(!inFront(p.pos, state->car)) continue;
//
//        double d = COORD::EuclideanDistance(carpos, p.pos);
//        if (d >= 0 && d < mindist)
//			mindist = d;
//    }
//
//    if(DoPrintCPU &&state->scenario_id ==0) printf("mindist, carvel= %f %f\n",mindist,carvel);
//
//    // TODO set as a param
//    if (mindist < /*2*/3.5) {
//		return (carvel <= 0.01) ? 0 : 2;
//    }
//
//    if (mindist < /*4*/5) {
//		if (carvel > 1.0+1e-4) return 2;
//		else if (carvel < 0.5-1e-4) return 1;
//		else return 0;
//    }
//    return carvel >= ModelParams::VEL_MAX-1e-4 ? 0 : 1;
	const PomdpState *state=static_cast<const PomdpState*>(particles[0]);
	return defaultStatePolicy(state);
}

double CalculateGoalDir(const CarStruct& car, const COORD& goal){
	COORD dir=goal-car.pos;
	double theta=atan2(dir.y, dir.x);
	if(theta<0)
		theta+=2*M_PI;

	theta=theta-car.heading_dir;
	theta=cap_angle(theta);
	if(theta<M_PI)
		return 1;//CCW
	else if(theta>M_PI)
		return -1;//CW
	else
		return 0;//ahead
}

ACT_TYPE WorldModel::defaultStatePolicy(const State* _state) const{
	logv << __FUNCTION__ << "] Get state info"<< endl;
	const PomdpState *state=static_cast<const PomdpState*>(_state);
	double mindist_along = numeric_limits<double>::infinity();
	double mindist_tang = numeric_limits<double>::infinity();
    const COORD& carpos = state->car.pos;
    double carheading = state->car.heading_dir;
	double carvel = state->car.vel;

	logv << __FUNCTION__ <<"] Calculate steering"<< endl;

	double steering = GetSteerToPath(static_cast<const PomdpState*>(state)[0]);

	logv << __FUNCTION__ << "] Calculate acceleration"<< endl;

	double acceleration;
	// Closest pedestrian in front
	for (int i=0; i<state->num; i++) {
		const AgentStruct & p = state->agents[i];

        // if(::inRealCollision(p.pos.x, p.pos.y, state->car.pos.x, state->car.pos.y, state->car.heading_dir)){
        //     mindist_along = 0;
        //     break;
        // }

        float infront_angle = ModelParams::IN_FRONT_ANGLE_DEG;
        if (Globals::config.pruning_constant > 100.0)
            infront_angle = 60.0;
		if(!inFront(p.pos, state->car, infront_angle)) continue;

        double d_along = COORD::DirectedDistance(carpos, p.pos, carheading);
        double d_tang = COORD::DirectedDistance(carpos, p.pos, carheading + M_PI / 2.0);

        if (p.type == AgentType::car){
            d_along = d_along - 2.0; 
            d_tang = d_tang - 1.0;
        }
        
		mindist_along = min(mindist_along, d_along);
        mindist_tang = min(mindist_tang, d_tang);
	}

	// TODO set as a param
	logv << __FUNCTION__ <<"] Calculate min dist"<< endl;
	if (mindist_along < 2/*3.5*/ || mindist_tang < 1.0) {
		acceleration= (carvel <= 0.01) ? 0 : -ModelParams::AccSpeed;
	}
	else if (mindist_along < 4/*5*/ && mindist_tang < 2.0) {
		if (carvel > ModelParams::VEL_MAX/1.5 +1e-4) acceleration= -ModelParams::AccSpeed;
		else if (carvel < ModelParams::VEL_MAX/2.0 -1e-4) acceleration= ModelParams::AccSpeed;
		else acceleration= 0.0;
	}
	else
		acceleration= carvel >= ModelParams::VEL_MAX-1e-4 ? 0 : ModelParams::AccSpeed;

    // acceleration = ModelParams::AccSpeed; // debugging

	logv << __FUNCTION__ <<"] Calculate action ID"<< endl;
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


bool WorldModel::inFront(const COORD ped_pos, const CarStruct car, double infront_angle_deg) const {
    if (infront_angle_deg == -1)
        infront_angle_deg = ModelParams::IN_FRONT_ANGLE_DEG;
    if(infront_angle_deg >= 180.0) {
        // inFront check is disabled
        return true;
    }

	double d0 = COORD::EuclideanDistance(car.pos, ped_pos);
	// if(d0 <= 0.7/*3.5*/) return true;
	double dot = DotProduct(cos(car.heading_dir), sin(car.heading_dir),
			ped_pos.x - car.pos.x, ped_pos.y - car.pos.y);
	double cosa = (d0>0)? dot / (d0): 0;
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
bool inCollision(double Px, double Py, double Cx, double Cy, double Ctheta, bool expand=true);
bool inCarlaCollision(double ped_x, double ped_y, double car_x, double car_y, double Ctheta, double car_extent_x, double car_extent_y, bool flag=0);

bool inRealCollision(double Px, double Py, double Cx, double Cy, double Ctheta, bool expand=true);
std::vector<COORD> ComputeRect(COORD pos, double heading, double ref_to_front_side, double ref_to_back_side, double ref_front_side_angle, double ref_back_side_angle);
bool InCollision(std::vector<COORD> rect_1, std::vector<COORD> rect_2);

bool WorldModel::inCollision(const PomdpState& state) {
	const COORD& car_pos = state.car.pos;

    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;

        if (!inFront(agent.pos, state.car))
            continue;
            
        if (agent.type == AgentType::ped){
            const COORD& pedpos = agent.pos;
            if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
                return true;
            }    
        }
        else if (agent.type == AgentType::car){    
            if (CheckCarWithVehicle(state.car, agent, 0))
                return true;
        }
        else{
            ERR(string_sprintf("unsupported agent type"));
        }
    }

    if(CheckCarWithObstacles(state.car,0))
    	return true;

    return false;
}

bool WorldModel::inRealCollision(const PomdpStateWorld& state, int &id, double infront_angle) {
    id=-1;
    if (infront_angle == -1){
        infront_angle = ModelParams::IN_FRONT_ANGLE_DEG;
    }

    const COORD& car_pos = state.car.pos;

    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;

        if (!inFront(agent.pos, state.car, infront_angle))
            continue;
            
        if (agent.type == AgentType::ped){
            const COORD& pedpos = agent.pos;
            if(::inRealCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
                id=agent.id;
            }
        }
        else if (agent.type == AgentType::car){
            if (CheckCarWithVehicle(state.car, agent, 1)){
                id = agent.id;
            }
        }
        else{
            ERR(string_sprintf("unsupported agent type"));
        }

        if (id != -1){
            logv << "[WorldModel::inRealCollision] car_pos: ("<< car_pos.x <<","<< car_pos.y<<"), heading: ("
                << std::cos(state.car.heading_dir) <<","<< std::sin(state.car.heading_dir)<<"), agent_pos: ("
                << agent.pos.x <<","<< agent.pos.y<<")\n";
            return true;
        }
    }
    
    if(CheckCarWithObstacles(state.car,1))
    	return true;

    return false;
}

bool WorldModel::inRealCollision(const PomdpStateWorld& state, double infront_angle_deg) {
    const COORD& car_pos = state.car.pos;
    int id = -1;
    if (infront_angle_deg == -1){
        infront_angle_deg = ModelParams::IN_FRONT_ANGLE_DEG;
    }

    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;

        if (!inFront(agent.pos, state.car, infront_angle_deg))
            continue;
        
        if (agent.type == AgentType::ped){
            const COORD& pedpos = agent.pos;
            if(::inRealCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
                id=agent.id;
            }
        }
        else if (agent.type == AgentType::car){
            if (CheckCarWithVehicle(state.car, agent, 1)){
                id = agent.id;
            }
        }
        else{
            ERR(string_sprintf("unsupported agent type"));
        }

        if (id != -1){
            logv << "[WorldModel::inRealCollision] car_pos: ("<< car_pos.x <<","<< car_pos.y<<"), heading: ("
                << std::cos(state.car.heading_dir) <<","<< std::sin(state.car.heading_dir)<<"), agent_pos: ("
                << agent.pos.x <<","<< agent.pos.y<<")\n";
            return true;
        }
    }

    if(CheckCarWithObstacles(state.car,1))
    	return true;

    return false;
}

bool WorldModel::inCollision(const PomdpStateWorld& state) {
	const COORD& car_pos = state.car.pos;

    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;

        if (!inFront(agent.pos, state.car))
            continue;
        
        if (agent.type == AgentType::ped){
            const COORD& pedpos = agent.pos;
            if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
                return true;
            }    
        }
        else if (agent.type == AgentType::car){
            if (CheckCarWithVehicle(state.car, agent, 0))
                return true;
        }
        else{
            ERR(string_sprintf("unsupported agent type"));
        }
    }

    if(CheckCarWithObstacles(state.car,0))
    	return true;

    return false;
}

bool WorldModel::inCollision(const PomdpState& state, int &id) {
	id=-1;
	const COORD& car_pos = state.car.pos;

    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;

        if (!inFront(agent.pos, state.car))
            continue;

        if (agent.type == AgentType::ped){
            const COORD& pedpos = agent.pos;
            if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
                id=agent.id;
                return true;
            }

        }
        else if (agent.type == AgentType::car){
            if (CheckCarWithVehicle(state.car, agent, 0)){
                id = agent.id;
                return true;
            }
        }
        else{
            ERR(string_sprintf("unsupported agent type"));
        }
    }

    if(CheckCarWithObstacles(state.car,0)){
    	id = -2;  // with obstacles
        return true;
    }

    if (id != -1){
        return true;
    }

    return false;
}

bool WorldModel::inCollision(const PomdpStateWorld& state, int &id) {
    id=-1;
	const COORD& car_pos = state.car.pos;

    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;

        if (!inFront(agent.pos, state.car))
            continue;

        if (agent.type == AgentType::ped){
            const COORD& pedpos = agent.pos;
            if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
                id=agent.id;
            }

        }
        else if (agent.type == AgentType::car){
            if (CheckCarWithVehicle(state.car, agent, 0)){
                id = agent.id;
            }
        }
        else{
            ERR(string_sprintf("unsupported agent type"));
        }

        if (id != -1)
            logv << "[WorldModel::inRealCollision] car_pos: ("<< car_pos.x <<","<< car_pos.y<<"), heading: ("
                << std::cos(state.car.heading_dir) <<","<< std::sin(state.car.heading_dir)<<"), agent_pos: ("
                << agent.pos.x <<","<< agent.pos.y<<")\n";
    }

    if(CheckCarWithObstacles(state.car,0))
    	id = -2;

    if (id != -1){
        return true;
    }

    return false;
}


void WorldModel::getClosestPed(const PomdpState& state, 
		int& closest_front_ped,
		double& closest_front_dist,
		int& closest_side_ped,
		double& closest_side_dist) {
	closest_front_ped = -1;
	closest_front_dist = numeric_limits<double>::infinity();
	closest_side_ped = -1;
	closest_side_dist = numeric_limits<double>::infinity();
    const COORD& carpos = state.car.pos;

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const AgentStruct& p = state.agents[i];
		bool front = inFront(p.pos, state.car);
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


bool WorldModel::isMovingAway(const PomdpState& state, int agent) {
  const auto& carpos = state.car.pos;

	const auto& pedpos = state.agents[agent].pos;
	const auto& goalpos = GetGoalPos(state.agents[agent]);

	if (state.agents[agent].intention == GetNumIntentions(agent)-1)
		return false;

	return DotProduct(goalpos.x - pedpos.x, goalpos.y - pedpos.y,
			cos(state.car.heading_dir), sin(state.car.heading_dir)) > 0;
}

///get the min distance between car and the agents in its front
double WorldModel::getMinCarPedDist(const PomdpState& state) {
    double mindist = numeric_limits<double>::infinity();
    const auto& carpos = state.car.pos;

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.agents[i];
		if(!inFront(p.pos, state.car)) continue;
        double d = COORD::EuclideanDistance(carpos, p.pos);
        if (d >= 0 && d < mindist) mindist = d;
    }

	return mindist;
}

///get the min distance between car and the agents
double WorldModel::getMinCarPedDistAllDirs(const PomdpState& state) {
    double mindist = numeric_limits<double>::infinity();
    const auto& carpos = state.car.pos;

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.agents[i];
        double d = COORD::EuclideanDistance(carpos, p.pos);
        if (d >= 0 && d < mindist) mindist = d;
    }

	return mindist;
}

int WorldModel::minStepToGoal(const PomdpState& state) {
    double d = COORD::EuclideanDistance(state.car.pos,car_goal);
    if (d < 0) d = 0;
    return int(ceil(d / (ModelParams::VEL_MAX/freq)));
}


int WorldModel::hasMinSteerPath(const PomdpState& state) {

	/// Return: 0 - has path
	///         1 - ccw
	///		    -1 - cw
	///			2 - goal reached

    double d = COORD::EuclideanDistance(state.car.pos,car_goal);
    if (d < 0) d = 0;
    if (d == 0) return 2; // already reach goal

    double theta = abs(COORD::SlopAngle(state.car.pos,car_goal) - state.car.heading_dir);

    if (theta > M_PI) {
         theta = 2 * M_PI - theta;
    }

    if (theta == 0) // can go through straight line
    	return int(ceil(d / (ModelParams::VEL_MAX/freq)));

    double r = ModelParams::CAR_WHEEL_DIST / tan(ModelParams::MaxSteerAngle);
    // double arc_len = min_turning_radii * theta;
    double chord_len = r * sin(theta) * 2;

//    double relax_thresh = ModelParams::GOAL_TOLERANCE;

    float turning_dir = COORD::get_dir(state.car.heading_dir, COORD::SlopAngle(state.car.pos,car_goal));

    if (chord_len >= d){
    	printf("No steering path available: chord_len=%f, d=%f, r=%f, theta=%f\n", chord_len, d, r, theta);

    	return turning_dir? 1:-1;  // no steer path available, should turn ccw or cw
    }
    else
    	return 0; // path available
}


int WorldModel::minStepToGoalWithSteer(const PomdpState& state) {
    double d = COORD::EuclideanDistance(state.car.pos,car_goal);
    if (d < 0) d = 0;
    if (d == 0) return 0;

    double theta = abs(COORD::SlopAngle(state.car.pos,car_goal) - state.car.heading_dir);

    if (theta > M_PI) {
         theta = 2 * M_PI - theta;
    }

    if (theta == 0) // can go through straight line
    	return int(ceil(d / (ModelParams::VEL_MAX/freq)));

    double r = ModelParams::CAR_WHEEL_DIST / tan(ModelParams::MaxSteerAngle);
    // double arc_len = min_turning_radii * theta;
    double chord_len = r * sin(theta) * 2;

    double relax_thresh = ModelParams::GOAL_TOLERANCE;

    if (chord_len >= d + relax_thresh){
//    	printf("No steering path available: chord_len=%f, d=%f, r=%f, theta=%f\n", chord_len, d, r, theta);
    	return Globals::config.sim_len;  // can never reach goal with in the round
    }
    else if (chord_len <  d + relax_thresh && chord_len >= d) {
    	return 2* theta * r; // the arc length of turning to the goal
    }
    else {
    	// calcutate the length of the shortest curve

    	float sin_theta = sin(theta);
    	float cos_theta = cos(theta);
    	float a = d * d + r * r - 2* d *r * sin_theta;
    	float b = 2 * r * (d * sin_theta - r);
    	float c = r * r - d * d * cos_theta * cos_theta;

    	float delta = b * b - 4 * a * c;
    	if (delta <= 0)
    		raise(SIGABRT);

    	float cos_phi_1 = (-b + sqrt(delta)) / (2 *a);
    	float cos_phi_2 = (-b - sqrt(delta)) / (2 *a);

    	float sin_phi_1 = (r * cos_phi_1 + d * sin_theta - r > 0)? sqrt(1 - cos_phi_1 * cos_phi_1): -sqrt(1 - cos_phi_1 * cos_phi_1);
    	float sin_phi_2 = (r * cos_phi_2 + d * sin_theta - r > 0)? sqrt(1 - cos_phi_2 * cos_phi_2): -sqrt(1 - cos_phi_2 * cos_phi_2);

//        float turning_dir = COORD::get_dir(state.car.heading_dir, COORD::SlopAngle(state.car.pos,car_goal));

    	float phi_1 = atan2(sin_phi_1, cos_phi_1); // -pi, - pi
    	float phi_2 = atan2(sin_phi_2, cos_phi_2); // -pi, - pi

        bool phi_1_ok = false; bool phi_2_ok = false;

        if (abs(r + ( d* sin_theta - r) * cos_phi_1 - d * cos_theta * sin_phi_1) < 1e-3)
            phi_1_ok = true;

        if (abs(r + ( d* sin_theta - r) * cos_phi_2 - d * cos_theta * sin_phi_2) < 1e-3)
            phi_2_ok = true;

        if (phi_1 < 0)
			phi_1 = phi_1 + 2 * M_PI;
		if (phi_2 < 0)
			phi_2 = phi_2 + 2 * M_PI;

    	float phi = 0, cos_phi = 0, sin_phi = 0; // the one larger than zero

    	if (!phi_1_ok)
			phi = phi_2;
		else if (!phi_2_ok)
			phi = phi_1;
		else if (phi_2 <= phi_1)
			phi = phi_2;
		else if (phi_1 <= phi_2)
			phi = phi_1;

    	if (phi == phi_1) {
    		cos_phi = cos_phi_1;
    		sin_phi = sin_phi_1;
    	}
    	else if (phi == phi_2){
			cos_phi = cos_phi_2;
			sin_phi = sin_phi_2;
		}
    	else // neither phi_1 or phi_2
    		raise(SIGABRT);

    	float arc_len = phi * r;

    	float line_len = sqrt(pow(r * cos_phi + d * sin_theta - r ,2) + pow(r * sin_phi - d * cos_theta,2));

    	return int(ceil( (arc_len + line_len) / (ModelParams::VEL_MAX/freq)));
    }
}

void WorldModel::PedStep(AgentStruct &agent, Random& random) {

    double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
    if(goal_mode == "goal") {
        AgentStepGoal(agent, 1, noise);
    }
    else if (goal_mode == "cur_vel") {
        PedStepCurVel(agent, 1, noise);
    }
    else if (goal_mode == "path") {
        AgentStepPath(agent, 1, noise);
    }

 //    const COORD& goal = GetGoalPos(agent);
	// if (goal.x == -1 && goal.y == -1) {  //stop intention
	// 	return;
	// }

	// MyVector goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);
 //    double a = goal_vec.GetAngle();
	// double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
 //    a += noise;

	// //TODO noisy speed
 //    MyVector move(a, agent.speed/freq, 0);
 //    agent.pos.x += move.dw;
 //    agent.pos.y += move.dh;
 //    return;
}

void WorldModel::PedStep(AgentStruct &agent, double& random) {

    double noise = sqrt(-2 * log(random));
    if(FIX_SCENARIO!=1 && !CPUDoPrint){
        random=QuickRandom::RandGeneration(random);
    }
    noise *= cos(2 * M_PI * random)* ModelParams::NOISE_GOAL_ANGLE;
    
    if(goal_mode == "goal") {
        AgentStepGoal(agent, 1, noise);
    }
    else if (goal_mode == "cur_vel") {
        PedStepCurVel(agent, 1, noise);
    }
    else if (goal_mode == "path") {
        AgentStepPath(agent, 1, noise);
    }

//     const COORD& goal = GetGoalPos(agent);
// 	if (goal.x == -1 && goal.y == -1) {  //stop intention
// 		return;
// 	}

// 	MyVector goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);

// 	if(goal_vec.GetLength() >= 0.5 ){

// 		double a = goal_vec.GetAngle();

// 		//double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
// 		double noise = sqrt(-2 * log(random));
// 		if(FIX_SCENARIO!=1 && !CPUDoPrint){
// 			random=QuickRandom::RandGeneration(random);
// 		}

// 		noise *= cos(2 * M_PI * random)* ModelParams::NOISE_GOAL_ANGLE;
// 		a += noise;

// 		//TODO noisy speed
// //		cout << "agent vel = " << agent.vel << endl;
// 		MyVector move(a, agent.speed/freq, 0);
// 		agent.pos.x += move.dw;
// 		agent.pos.y += move.dh;
// 	}

    // return;
}


double gaussian_prob(double x, double stddev) {
    double a = 1.0 / stddev / sqrt(2 * M_PI);
    double b = - x * x / 2.0 / (stddev * stddev);
    return a * exp(b);
}

COORD WorldModel::GetGoalPosFromPaths(int agent_id, int intention_id, int pos_along_path, 
    const COORD& agent_pos, AgentType type, bool agent_cross_dir){
    auto& path_candidates = PathCandidates(agent_id);

    if (intention_id < path_candidates.size()){
        auto& path = path_candidates[intention_id];
        COORD pursuit = path[path.forward(pos_along_path, 3.0)];
        return pursuit; 
    }
    else if (intention_id < GetNumIntentions(agent_id)-1) { 
        ERR("This code block should has been disabled");
        // not-stop intention not included in path list = agent cross intention
        if(type == AgentType::ped){
            int path_id = 0; // agent has only 1 path
            auto& path = path_candidates[path_id];
            COORD cross_dir= path.GetCrossDir(pos_along_path, agent_cross_dir); 
            return COORD(agent_pos.x + cross_dir.x/freq*3, agent_pos.y + cross_dir.y/freq*3);
        } else {
            cout << " intention_id";           
            for (auto & path: path_candidates){
                cerr << "candidate path:";
                for (auto& point : path)
                    cerr << point.x << " " << point.y << " ";
                cerr << endl;
            }
            ERR(string_sprintf(
                "Agent type %d should not have cross intention %d for num_intentions %d and num_paths %d\n", 
                type, intention_id, GetNumIntentions(agent_id), path_candidates.size()));
        }
    }
    else if (intention_id == GetNumIntentions(agent_id)-1){ // stop intention
        return COORD(agent_pos.x, agent_pos.y);
    }
    else{
        ERR(string_sprintf("Intention ID %d excesses # intentions %d for agent %d of type %d\n", 
            intention_id, GetNumIntentions(agent_id), agent_id, type));
    }
}

bool fetch_cross_dir(const Agent& agent) {
    if (agent.type() == AgentType::ped)
        return static_cast<const Pedestrian*>(&agent)->cross_dir;
    else
        return true;
}

std::vector<COORD> fetch_bounding_box(const Agent& agent) {
    if (agent.type() == AgentType::car){
        return static_cast<const Vehicle*>(&agent)->bb;
    }
    else{
        std::vector<COORD> bb;
        return bb;
    }
}

double fetch_heading_dir(const Agent& agent) {
    if (agent.type()==AgentType::car){
        auto* car = static_cast<const Vehicle*>(&agent);
        // if (car->vel.Length()>1.0){
        //     double diff = fabs(car->heading_dir - car->vel.GetAngle());
            // if (diff > 0.8 && diff < 2*M_PI - 0.8)
            //     ERR(string_sprintf("heading-veldir mismatch: %f %f", 
            //         car->heading_dir, car->vel.GetAngle()));
        // }
        return static_cast<const Vehicle*>(&agent)->heading_dir;
    }
    else
        return 0.0;
}

COORD WorldModel::GetGoalPos(const Agent& agent, int intention_id){
        
    if (goal_mode == "path") {
        bool cross_dir = fetch_cross_dir(agent);
        return GetGoalPosFromPaths(agent.id, intention_id, 0, 
            COORD(agent.w, agent.h), agent.type(), cross_dir); // assuming path is up-to-date
    }
    else if (goal_mode == "cur_vel")
        return COORD(agent.w + agent.vel.x/freq*3, agent.h + agent.vel.y/freq*3);
    else if (goal_mode == "goal")
        return goals[intention_id];
    else{
        rasie_unsupported_goal_mode(__FUNCTION__);
    }
}

COORD WorldModel::GetGoalPos(const AgentStruct& agent, int intention_id){

    if (intention_id ==-1)
        intention_id = agent.intention;

    if (goal_mode == "path") 
        return GetGoalPosFromPaths(agent.id, intention_id, agent.pos_along_path, 
            agent.pos, agent.type, agent.cross_dir);
    else if (goal_mode == "cur_vel")
        return COORD(agent.pos.x + agent.vel.x/freq*3, agent.pos.y + agent.vel.y/freq*3);
    else if (goal_mode == "goal")
        return goals[intention_id];
    else{
        rasie_unsupported_goal_mode(__FUNCTION__);
    }
}

COORD WorldModel::GetGoalPos(const AgentBelief& agent, int intention_id){

    int pos_along_path = 0; // assumed that AgentBelied has up-to-date paths
    if (goal_mode == "path") 
        return GetGoalPosFromPaths(agent.id, intention_id, pos_along_path, 
            agent.pos, agent.type, agent.cross_dir);
    else if (goal_mode == "cur_vel")
        return COORD(agent.pos.x + agent.vel.x/freq*3, agent.pos.y + agent.vel.y/freq*3);
    else if (goal_mode == "goal")
        return goals[intention_id];
    else{
        rasie_unsupported_goal_mode(__FUNCTION__);
    }
}

int WorldModel::GetNumIntentions(const AgentStruct& agent){
    if (goal_mode == "path")
        return NumPaths(agent.id) + 1; // number of paths + stop (agent has one more cross intention)
    else if (goal_mode == "cur_vel")
        return 2;
    else if (goal_mode == "goal")
        return goals.size();
}


int WorldModel::GetNumIntentions(const Agent& agent){
    if (goal_mode == "path")
        return NumPaths(agent.id) + 1; // number of paths + stop (agent has one more cross intention)
    else if (goal_mode == "cur_vel")
        return 2;
    else if (goal_mode == "goal")
        return goals.size();
}

int WorldModel::GetNumIntentions(int agent_id){
    if (goal_mode == "path")
        return NumPaths(agent_id) + 1; // number of paths + stop (agent has one more cross intention)
    else if (goal_mode == "cur_vel")
        return 2;
    else if (goal_mode == "goal")
        return goals.size();
}

void WorldModel::AgentStepGoal(AgentStruct& agent, int step, double noise) {
    if (agent.type == AgentType::ped)
        PedStepGoal(agent, step, noise);
    else if (agent.type == AgentType::car)
        VehStepGoal(agent, step, noise);
    else
        ERR(string_sprintf(
            "unsupported agent mode %d", agent.type));
}

void WorldModel::PedStepGoal(AgentStruct& agent, int step, double noise) {
  const COORD& goal = goals[agent.intention];
	if (agent.intention == GetNumIntentions(agent.id)-1) {  //stop intention
		return;
	}

	COORD goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);
    if (noise!=0){
        double a = goal_vec.GetAngle(); 
        a += noise;
        MyVector move(a, step * agent.speed/freq, 0);
        agent.pos.x += move.dw;
        agent.pos.y += move.dh;    
    } else{
        goal_vec.AdjustLength(step * agent.speed / freq);
        agent.pos.x += goal_vec.x;
        agent.pos.y += goal_vec.y;
    }
}

void WorldModel::VehStepGoal(AgentStruct& agent, int step, double noise) {
  const COORD& goal = goals[agent.intention];
    if (agent.intention == GetNumIntentions(agent.id)-1) {  //stop intention
        return;
    }

    COORD old_pos = agent.pos;
    COORD goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);
    goal_vec.AdjustLength(3.0);// lookahead 3 meters

    COORD pursuit_point = goal_vec + agent.pos;
    // double steering = PurepursuitAngle(agent, pursuit_point);
    double steering = PControlAngle<AgentStruct>(agent, pursuit_point);
    
    steering += noise;
    BicycleModel(agent, steering, agent.speed);

    agent.vel = (agent.pos - old_pos)*freq;
}

void WorldModel::PedStepCurVel(AgentStruct& agent, int step, double noise) {
    if (noise!=0){
        double a = agent.vel.GetAngle(); 
        a += noise;
        MyVector move(a, step * agent.vel.Length()/freq, 0);
        agent.pos.x += move.dw;
        agent.pos.y += move.dh;
    }
    else{
        agent.pos.x += agent.vel.x * (float(step)/freq);
        agent.pos.y += agent.vel.y * (float(step)/freq);
    }
}

double WorldModel::PurepursuitAngle(const CarStruct& car, COORD& pursuit_point) const{
    COORD rear_pos;
    rear_pos.x = car.pos.x - ModelParams::CAR_REAR * cos(car.heading_dir); 
    rear_pos.y = car.pos.y - ModelParams::CAR_REAR * sin(car.heading_dir);
    
    double offset = (rear_pos-pursuit_point).Length();
    double target_angle = atan2(pursuit_point.y - rear_pos.y, pursuit_point.x - rear_pos.x);    
    double angular_offset = cap_angle(target_angle - car.heading_dir);

    COORD relative_point(
        offset * cos(angular_offset), offset * sin(angular_offset));
    if (relative_point.x == 0)
        return 0;
    else{
        double turning_radius = relative_point.Length() / (2 * abs(relative_point.y)); // Intersecting chords theorem.
        double steering_angle = atan2(ModelParams::CAR_WHEEL_DIST, turning_radius);
        if (relative_point.y < 0)
            steering_angle *= -1;
        
        return steering_angle;
    }
}

double WorldModel::PurepursuitAngle(const AgentStruct& agent, COORD& pursuit_point) const{
    COORD rear_pos;
    rear_pos.x = agent.pos.x - agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir); 
    rear_pos.y = agent.pos.y - agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);
    
    double offset = (rear_pos-pursuit_point).Length();
    double target_angle = atan2(pursuit_point.y - rear_pos.y, pursuit_point.x - rear_pos.x);    
    double angular_offset = cap_angle(target_angle - agent.heading_dir);

    COORD relative_point(
        offset * cos(angular_offset), offset * sin(angular_offset));
    if (relative_point.x == 0)
        return 0;
    else{
        double turning_radius = relative_point.Length() / (2 * abs(relative_point.y)); // Intersecting chords theorem.
        double steering_angle = atan2(agent.bb_extent_y * 2 * 0.8, turning_radius);
        if (relative_point.y < 0)
            steering_angle *= -1;
        
        return steering_angle;
    }
}
// std::vector<COORD> get_bounding_box_corners(AgentStruct& agent){
//     COORD forward_vec = COORD(cos(agent.heading_dir), sin(agent.heading_dir));
//     COORD sideward_vec = COORD(-sin(agent.heading_dir), cos(agent.heading_dir));
    
//     double side_len = agent.bb_extent_x;
//     double forward_len = agent.bb_extent_y;

//     if (agent.type == AgentType::ped){
//         side_len = 0.23;
//         forward_len = 0.23;
//     }

//     return std::vector<COORD>({
//         agent.pos + forward_vec * forward_len + sideward_vec * side_len,
//         agent.pos + forward_vec * forward_len - sideward_vec * side_len,
//         agent.pos - forward_vec * forward_len - sideward_vec * side_len,
//         agent.pos - forward_vec * forward_len + sideward_vec * side_len,
//     });
// }

void WorldModel::cal_bb_extents(AgentBelief& b, AgentStruct& agent){
    COORD forward_vec = COORD(cos(b.heading_dir), sin(b.heading_dir));
    COORD sideward_vec = COORD(-sin(b.heading_dir), cos(b.heading_dir));
    
    if (agent.type == AgentType::ped){
        agent.bb_extent_x = 0.3; agent.bb_extent_y = 0.3;        
    }
    else {
        agent.bb_extent_x = 0.0; agent.bb_extent_y = 0.0;
    }    

    for (auto& point: b.bb){
        agent.bb_extent_x = max((point - agent.pos).dot(sideward_vec), agent.bb_extent_x);
        agent.bb_extent_y = max((point - agent.pos).dot(forward_vec), agent.bb_extent_y);
    }
}

void WorldModel::cal_bb_extents(AgentStruct& agent, std::vector<COORD>& bb, double heading_dir){
    COORD forward_vec = COORD(cos(heading_dir), sin(heading_dir));
    COORD sideward_vec = COORD(-sin(heading_dir), cos(heading_dir));
    
    if (agent.type == AgentType::ped){
        agent.bb_extent_x = 0.3; agent.bb_extent_y = 0.3;        
    }
    else {
        agent.bb_extent_x = 0.0; agent.bb_extent_y = 0.0;
    }    

    for (auto& point: bb){
        agent.bb_extent_x = max((point - agent.pos).dot(sideward_vec), agent.bb_extent_x);
        agent.bb_extent_y = max((point - agent.pos).dot(forward_vec), agent.bb_extent_y);
    }
}

void WorldModel::AgentStepPath(AgentStruct& agent, int step, double noise, bool doPrint) {
    if (agent.type == AgentType::ped)
        PedStepPath(agent, step, noise, doPrint);
    else if (agent.type == AgentType::car)
        VehStepPath(agent, step, noise, doPrint);
    else
        ERR(string_sprintf(
            "unsupported agent mode %d", agent.type));
}

void WorldModel::PedStepPath(AgentStruct& agent, int step, double noise, bool doPrint) {
    auto& path_candidates = PathCandidates(agent.id);

    int intention;
    if(doPrint)
        intention = 0;
    else
        intention = agent.intention;

    int old_path_pos = agent.pos_along_path;

    if (intention < path_candidates.size()){
        auto& path = path_candidates[intention];
        
        agent.pos_along_path = path.forward(agent.pos_along_path, agent.speed * (float(step)/freq));
        COORD new_pos = path[agent.pos_along_path];
        
        if (noise!=0){
            COORD goal_vec = new_pos - agent.pos;
            double a = goal_vec.GetAngle(); 
            a += noise;
            MyVector move(a, step * agent.speed/freq, 0);
            agent.pos.x += move.dw;
            agent.pos.y += move.dh;
        }
        else{
            agent.pos = new_pos;
        }

        agent.vel = (new_pos - agent.pos) * freq;

        if(doPrint && agent.pos_along_path == old_path_pos)
            logv << "[PedStepPath]: agent " << agent.id << " no move: path length " 
                << path.size() << " speed " << agent.speed  << " forward distance " << agent.speed * (float(step)/freq) << endl;   
    }
}


void WorldModel::VehStepPath(AgentStruct& agent, int step, double noise, bool doPrint) {
    auto& path_candidates = PathCandidates(agent.id);

    int intention;
    if(doPrint)
        intention = 0;
    else
        intention = agent.intention;

    int old_path_pos = agent.pos_along_path;

    if (intention < path_candidates.size()){
        COORD old_pos = agent.pos;
        auto& path = path_candidates[intention];
        int pursuit_pos = path.forward(agent.pos_along_path, 3.0);
        COORD pursuit_point = path[pursuit_pos];

        // double steering = PurepursuitAngle(agent, pursuit_point);
        double steering = PControlAngle<AgentStruct>(agent, pursuit_point);
        steering += noise;
        BicycleModel(agent, steering, agent.speed);

        agent.pos_along_path = path.nearest(agent.pos);
        agent.vel = (agent.pos - old_pos)*freq;

        if(doPrint && agent.pos_along_path == old_path_pos)
            logv << "[PedStepPath]: agent " << agent.id << " no move: path length " 
                << path.size() << " speed " << agent.speed  << " forward distance " << agent.speed * (float(step)/freq) << endl;   
    }
    else{ // stop intention

    }
}

double WorldModel::agentMoveProb(COORD prev, const Agent& agent, int intention_id) {
	const double K = 0.001;
  COORD curr(agent.w, agent.h);
  const COORD& goal = GetGoalPos(agent, intention_id); 
	double move_dist = Norm(curr.x-prev.x, curr.y-prev.y),
		   goal_dist = Norm(goal.x-prev.x, goal.y-prev.y);
	double sensor_noise = 0.1;
    if(ModelParams::is_simulation) sensor_noise = 0.02;

    bool debug=false;
	// CHECK: beneficial to add back noise?
	if(debug) cout<<"intention id "<<intention_id<<endl;
	if (intention_id == GetNumIntentions(agent.id)-1) {  //stop intention 
		return (move_dist < sensor_noise) ? 0.4 : 0;
	} else {
		if (move_dist < sensor_noise) return 0;

		double cosa = DotProduct(curr.x-prev.x, curr.y-prev.y, goal.x-prev.x, goal.y-prev.y) / (move_dist * goal_dist);
		if(cosa >1) cosa = 1;
		else if(cosa < -1) cosa = -1;
		double angle = acos(cosa);
		double prob = gaussian_prob(angle, ModelParams::NOISE_GOAL_ANGLE) + K;
        assert(prob == prob);//nan detector
        return prob;
	}
}

void WorldModel::EnsureMeanDirExist(int agent_id){
    auto it = ped_mean_dirs.find(agent_id);
    if ( it == ped_mean_dirs.end()){ 
        // DEBUG(string_sprintf("Encountering new agent id %d, adding mean dir for it...", agent_id));
        vector<COORD> dirs(GetNumIntentions(agent_id));
        ped_mean_dirs[agent_id] = dirs;
    }
    else if (it->second.size() != GetNumIntentions(agent_id)){ // path list size has been updated
        ped_mean_dirs[agent_id].resize(GetNumIntentions(agent_id));
    }
}

double WorldModel::agentMoveProb(COORD prev, const Agent& agent, int intention_id, int ped_mode) {
	const double K = 0.001;
//	cout << __FUNCTION__ << "@" << __LINE__ << endl;
    COORD curr(agent.w, agent.h);
    int agent_id = agent.id;
	double move_dist = Norm(curr.x-prev.x, curr.y-prev.y);

	COORD goal;

	if(ped_mode == AGENT_ATT){
        EnsureMeanDirExist(agent_id);

		if(intention_id >= ped_mean_dirs[agent_id].size()){
			ERR(string_sprintf(
                "Encountering overflowed intention id %d at agent %d\n ", 
                intention_id, agent_id));
		}
		goal = ped_mean_dirs[agent_id][intention_id] + prev;
	}
	else{
		goal = GetGoalPos(agent, intention_id);
	}

	double goal_dist = Norm(goal.x-prev.x, goal.y-prev.y);

	double sensor_noise = 0.1;
    if(ModelParams::is_simulation) sensor_noise = 0.02;

    bool debug=false;
	// CHECK: beneficial to add back noise?
	if(debug) cout<<"intention id "<<intention_id<<endl;
	if (intention_id == GetNumIntentions(agent.id)-1) {  //stop intention
		logv <<"stop intention" << endl;

		return (move_dist < sensor_noise) ? 0.4 : 0;
	} else {

		double angle = 0;
		if (goal_dist > 1e-5 && move_dist > sensor_noise){
			double cosa = DotProduct(curr.x-prev.x, curr.y-prev.y, goal.x-prev.x, goal.y-prev.y) / (move_dist * goal_dist);
			if(cosa >1) cosa = 1;
			else if(cosa < -1) cosa = -1;
			angle = acos(cosa);
		}
		else
			logv <<"goal_dist=" << goal_dist << " move_dist " << move_dist << endl;
//		cout << __FUNCTION__ << "@" << __LINE__ << endl;

		double angle_prob = gaussian_prob(angle, ModelParams::NOISE_GOAL_ANGLE) + K;

		double vel_error=0;
		if(ped_mode == AGENT_ATT){
			double mean_vel = ped_mean_dirs[agent_id][intention_id].Length();
			vel_error = move_dist - mean_vel;

			logv <<"ATT angle error=" << angle << endl;
			logv <<"ATT vel_error=" << vel_error << endl;

		}
		else{
			vel_error = move_dist - ModelParams::PED_SPEED/freq;

			logv <<"DIS angle error=" << angle << endl;
			logv <<"DIS vel_error=" << vel_error << endl;

		}
//		cout << __FUNCTION__ << "@" << __LINE__ << endl;

		double vel_prob = gaussian_prob(vel_error, ModelParams::NOISE_PED_VEL/freq) + K;
		assert(angle_prob == angle_prob ); // nan detector
        assert(vel_prob == vel_prob ); // nan detector
        return angle_prob* vel_prob;
	}
}

void WorldModel::FixGPUVel(CarStruct &car)
{
	float tmp=car.vel/(ModelParams::AccSpeed/freq);
	car.vel=((int)(tmp+0.5))*(ModelParams::AccSpeed/freq);
}

void WorldModel::BicycleModel(CarStruct &car, double steering, double end_vel) {
    if(steering!=0){
        assert(tan(steering)!=0);
        double TurningRadius = ModelParams::CAR_WHEEL_DIST/tan(steering);
        assert(TurningRadius!=0);
        double beta= end_vel/freq/TurningRadius;

        COORD rear_pos;
        rear_pos.x = car.pos.x - ModelParams::CAR_REAR * cos(car.heading_dir); 
        rear_pos.y = car.pos.y - ModelParams::CAR_REAR * sin(car.heading_dir);
        // move and rotate
        rear_pos.x += TurningRadius*(sin(car.heading_dir+beta)-sin(car.heading_dir));
        rear_pos.y += TurningRadius*(cos(car.heading_dir)-cos(car.heading_dir+beta));
        car.heading_dir = cap_angle(car.heading_dir+beta);
        car.pos.x = rear_pos.x + ModelParams::CAR_REAR * cos(car.heading_dir);
        car.pos.y = rear_pos.y + ModelParams::CAR_REAR * sin(car.heading_dir);
    }
    else{
        car.pos.x += (end_vel/freq) * cos(car.heading_dir);
        car.pos.y += (end_vel/freq) * sin(car.heading_dir);
    } 
}

void WorldModel::BicycleModel(AgentStruct &agent, double steering, double end_vel) {
    if(steering!=0){
        assert(tan(steering)!=0);
        // assuming front-real length is 0.8 * total car length
        double TurningRadius = agent.bb_extent_y * 2 * 0.8 /tan(steering);
        assert(TurningRadius!=0);
        double beta= end_vel/freq/TurningRadius;

        COORD rear_pos;
        rear_pos.x = agent.pos.x - agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir); 
        rear_pos.y = agent.pos.y - agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);
        // move and rotate
        rear_pos.x += TurningRadius*(sin(agent.heading_dir+beta)-sin(agent.heading_dir));
        rear_pos.y += TurningRadius*(cos(agent.heading_dir)-cos(agent.heading_dir+beta));
        agent.heading_dir = cap_angle(agent.heading_dir+beta);
        agent.pos.x = rear_pos.x + agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir);
        agent.pos.y = rear_pos.y + agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);
    }
    else{
        agent.pos.x += (end_vel/freq) * cos(agent.heading_dir);
        agent.pos.y += (end_vel/freq) * sin(agent.heading_dir);
    }
}

void WorldModel::RobStep(CarStruct &car, double steering, Random& random) {
	BicycleModel(car, steering, car.vel);
}

void WorldModel::RobStep(CarStruct &car, double steering, double& random) {
    BicycleModel(car, steering, car.vel);	
}


void WorldModel::RobStep(CarStruct &car, double& random, double acc, double steering) {
    double end_vel = car.vel + acc / freq;
    end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);
    BicycleModel(car, steering, end_vel);
}

void WorldModel::RobStep(CarStruct &car, Random& random, double acc, double steering) {
    double end_vel = car.vel + acc / freq;
    end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);
    BicycleModel(car, steering, end_vel);
}


void WorldModel::RobVelStep(CarStruct &car, double acc, Random& random) {
    const double N = ModelParams::NOISE_ROBVEL;
    if (N>0) {
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
    if (N>0) {
    	if(FIX_SCENARIO!=1 && !CPUDoPrint)
    		random=QuickRandom::RandGeneration(random);
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
	car.pos.x+=(car.vel/freq) * cos(car.heading_dir);
	car.pos.y+=(car.vel/freq) * sin(car.heading_dir);
}

void WorldModel::RobStepCurAction(CarStruct &car, double acc, double steering) {
	bool fix_scenario = CPUDoPrint;
	CPUDoPrint = true;  // to block the random number generation

	double det_prob=1;

	RobStep(car, steering, det_prob);
	RobVelStep(car, acc, det_prob);

	CPUDoPrint = fix_scenario;
}

double WorldModel::ISRobVelStep(CarStruct &car, double acc, Random& random) {
    const double N = 4 * ModelParams::NOISE_ROBVEL;
    double weight = 1;
    if (N>0) {
        double prob = random.NextDouble();
        if (prob > N) {
            car.vel += acc / freq;
            weight = (1.0 - ModelParams::NOISE_ROBVEL)/(1.0 -N);
        }
        else weight = ModelParams::NOISE_ROBVEL / N;
    } else {
        car.vel += acc / freq;
    }

    car.vel = max(min(car.vel, ModelParams::VEL_MAX), 0.0);

    return weight;
}

void WorldModel::setPath(Path path) {
    this->path = path;
    ModelParams::GOAL_TRAVELLED = path.getlength();
}

void WorldModel::updatePedBelief(AgentBelief& b, const Agent& curr_agent) {
//    const double ALPHA = 0.8;

    const double ALPHA = 0.1;

	const double SMOOTHING=ModelParams::BELIEF_SMOOTHING;

    bool debug=false;

    b.bb = fetch_bounding_box(curr_agent);
    b.heading_dir = fetch_heading_dir(curr_agent);
    b.cross_dir = fetch_cross_dir(curr_agent);
    
    // cout  << "curr_agent.reset_intention = " << curr_agent.reset_intention << ", id =" << curr_agent.id << endl;
    if(/*true*/!curr_agent.reset_intention){

        b.goal_paths = PathCandidates(b.id);

        for(int i=0; i<GetNumIntentions(curr_agent); i++) {

            // Attentive mode
            logv << "TEST ATT" << endl;
            double prob = agentMoveProb(b.pos, curr_agent, i, AGENT_ATT);
            if(debug) cout << "attentive likelihood " << i << ": " << prob << endl;
            b.prob_modes_goals[AGENT_ATT][i] *=  prob;
            // Keep the belief noisy to avoid aggressive policies
            b.prob_modes_goals[AGENT_ATT][i] += SMOOTHING / GetNumIntentions(curr_agent)/2.0; // CHECK: decrease or increase noise

            // Detracted mode
            logv << "TEST DIS" << endl;
            prob = agentMoveProb(b.pos, curr_agent, i, AGENT_DIS);
            if(debug) cout << "Detracted likelihood " << i << ": " << prob << endl;
            b.prob_modes_goals[AGENT_DIS][i] *=  prob;
            // Important: Keep the belief noisy to avoid aggressive policies
            b.prob_modes_goals[AGENT_DIS][i] += SMOOTHING / GetNumIntentions(curr_agent)/2.0; // CHECK: decrease or increase noise
        }
        if(debug) {
            for(double w: b.prob_modes_goals[AGENT_ATT]) {
                cout << w << " ";
            }
            cout << endl;
            for(double w: b.prob_modes_goals[AGENT_DIS]) {
                cout << w << " ";
            }
            cout << endl;
        }
        // normalize
        double total_weight_att = accumulate(b.prob_modes_goals[AGENT_ATT].begin(), b.prob_modes_goals[AGENT_ATT].end(), double(0.0));
        double total_weight_dis = accumulate(b.prob_modes_goals[AGENT_DIS].begin(), b.prob_modes_goals[AGENT_DIS].end(), double(0.0));
        double total_weight = total_weight_att + total_weight_dis;

        if (total_weight <= 0.01)
            ERR(string_sprintf("total_weight too small %f", total_weight));

        if(debug) 
            cout << "[updatePedBelief] total_weight = " << total_weight << endl;
        for(double& w : b.prob_modes_goals[AGENT_ATT]) {
            w /= total_weight;
        }
        for(double& w : b.prob_modes_goals[AGENT_DIS]) {
            w /= total_weight;
        }
    }

    COORD cur_pos(curr_agent.w, curr_agent.h);
    double moved_dist = COORD::EuclideanDistance(b.pos, cur_pos);
    b.vel = b.vel*ALPHA + (cur_pos - b.pos)* (ModelParams::control_freq * (1-ALPHA));
    // b.speed = ALPHA * b.speed + (1-ALPHA) * moved_dist * ModelParams::control_freq;
    if (b.type == AgentType::car){
        b.speed = b.vel.Length();
    }
    else{
        b.speed = ModelParams::PED_SPEED;
    }
	b.pos = cur_pos;
}

double WorldModel::GetPrefSpeed(const Agent& agent){
    if (agent.type() == AgentType::car)
        return agent.vel.Length();
    else    
        return ModelParams::PED_SPEED;
}

AgentBelief WorldModel::initPedBelief(const Agent& agent) {
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

    // b.speed = ModelParams::PED_SPEED;
    //    cout << "AGENT_DIS + 1= "<< AGENT_DIS + 1 << endl;
    
    int num_types = NUM_AGENT_TYPES;
    for(int i =0 ; i < num_types; i++){
        // vector<double> temp_probs;
        // temp_probs.reserve(20);
        // for (int i =0; i<GetNumIntentions(agent); i++)
        //     temp_probs.push_back(1.0/GetNumIntentions(agent)/num_types);
        // b.prob_modes_goals.push_back(temp_probs);
        b.prob_modes_goals.push_back(vector<double>());
        b.prob_modes_goals.back().reserve(20);
        for (int i =0; i<GetNumIntentions(agent); i++)
            b.prob_modes_goals.back().push_back(1.0/GetNumIntentions(agent)/num_types);
    }

    return b;
}

void WorldStateTracker::cleanAgents() {
    cleanPed();
    cleanVeh();

//    DEBUG("After cleaning agents");
    model.print_path_map();
}

void WorldStateTracker::cleanPed() {
    vector<Pedestrian> ped_list_new;
    for(int i=0;i<ped_list.size();i++)
    {
        bool insert = AgentIsAlive(ped_list[i], ped_list_new);
        if (insert)
            ped_list_new.push_back(ped_list[i]);
        else{
            logi << "Cleaning ped " << ped_list[i].id << 
                " from ped_list, id_map_num_paths, id_map_paths" << endl;
            model.id_map_num_paths.erase(ped_list[i].id);
            model.id_map_paths.erase(ped_list[i].id);
        }
    }
    ped_list=ped_list_new;
}

void WorldStateTracker::cleanVeh() {
    vector<Vehicle> veh_list_new;
    for(int i=0;i<veh_list.size();i++)
    {
        bool insert = AgentIsAlive(veh_list[i], veh_list_new);
        if (insert)
            veh_list_new.push_back(veh_list[i]);
        else{
            logi << "Cleaning veh " << veh_list[i].id << 
                " from veh_list, id_map_num_paths, id_map_paths" << endl;
            model.id_map_num_paths.erase(veh_list[i].id);
            model.id_map_paths.erase(veh_list[i].id);
        }
    }
    veh_list=veh_list_new;
}

double get_timestamp(){
	return Globals::ElapsedTime(init_time);
}

void WorldStateTracker::tracPos(Agent& des, const Agent& src, bool doPrint){
    des.w=src.w;
    des.h=src.h;
}

void WorldStateTracker::tracBoundingBox(Vehicle& des, const Vehicle& src, bool doPrint){
    des.bb=src.bb;
    des.heading_dir = src.heading_dir;
}

void WorldStateTracker::tracCrossDirs(Pedestrian& des, const Pedestrian& src, bool doPrint){
    des.cross_dir=src.cross_dir;
}

void WorldStateTracker::updatePathPool(Agent& des){
    model.id_map_paths[des.id] = des.paths; 

    switch (des.type()){
        case AgentType::ped:
            model.id_map_num_paths[des.id] = des.paths.size(); break;
        case AgentType::car:
            model.id_map_num_paths[des.id] = des.paths.size(); break;
        default:
            cout << __FUNCTION__ <<": unsupported agent type " << des.type() << endl;
            raise(SIGABRT); break;
    }
}

void WorldStateTracker::tracIntention(Agent& des, const Agent& src, bool doPrint){

    if(des.paths.size()!=src.paths.size())
        des.reset_intention = true;
    else
        des.reset_intention = src.reset_intention;
    // des.paths = src.paths;

    des.paths.resize(0);

    for (const Path& path: src.paths){
        des.paths.push_back(path.interpolate(60.0)); // reset the resolution of the path to ModelParams:PATH_STEP
    }

    updatePathPool(des);
}

void WorldStateTracker::trackVel(Agent& des, const Agent& src, bool& no_move, bool doPrint){
    double duration =  src.ros_time_stamp - des.ros_time_stamp;

    if (duration < 0.001 / Globals::config.time_scale){
        no_move = false;

        DEBUG(string_sprintf("Update duration too short for agent %d: %f, (%f-%f)", 
            src.id, duration, src.ros_time_stamp, des.ros_time_stamp));
        return;
    }

    des.vel.x = (src.w - des.w) / duration;
    des.vel.y = (src.h - des.h) / duration;

    if (des.vel.Length()>1e-4){
        no_move = false;
    }
    else{
//        DEBUG(string_sprintf("Vel too small: (%f, %f)", des.vel.x, des.vel.y));
    }
}


void WorldStateTracker::updateVeh(const Vehicle& veh, bool doPrint){

    int i=0;
    bool no_move = true;
    for(;i<veh_list.size();i++) {
        if (veh_list[i].id==veh.id) {
            logv <<"[updateVel] updating agent " << veh.id << endl;

            trackVel(veh_list[i], veh, no_move, doPrint);
            tracPos(veh_list[i], veh, doPrint);
            tracIntention(veh_list[i], veh, doPrint);
            tracBoundingBox(veh_list[i], veh, doPrint);

            veh_list[i].time_stamp = veh.time_stamp;
            break;
        }

        if (abs(veh_list[i].w-veh.w)<=0.1 && abs(veh_list[i].h-veh.h)<=0.1)   //overlap
        {}
    }

    if (i==veh_list.size()) {
        logv << "[updateVel] updating new agent " << veh.id << endl;

        no_move = false;

        veh_list.push_back(veh);

        veh_list.back().vel.x = 0.01; // to avoid subsequent runtime error
        veh_list.back().vel.y = 0.01; // to avoid subsequent runtime error
        veh_list.back().time_stamp = veh.time_stamp;

        updatePathPool(veh_list.back());
    }

    if (no_move){
        cout << __FUNCTION__ << " no_move veh "<< veh.id <<
                " caught: vel " << veh_list[i].vel.x <<" "<< veh_list[i].vel.y << endl;
    }
}

void WorldStateTracker::updateVehState(const Vehicle& veh, bool doPrint){

    int i=0;
    bool no_move = true;
    for(;i<veh_list.size();i++) {
        if (veh_list[i].id==veh.id) {
            logv <<"[updateVel] updating agent " << veh.id << endl;

            trackVel(veh_list[i], veh, no_move, doPrint);
            tracPos(veh_list[i], veh, doPrint);
            tracBoundingBox(veh_list[i], veh, doPrint);
            veh_list[i].time_stamp = veh.time_stamp;
            break;
        }

        if (abs(veh_list[i].w-veh.w)<=0.1 && abs(veh_list[i].h-veh.h)<=0.1)   //overlap
        {}
    }

    if (i==veh_list.size()) {
        logv << "[updateVel] updating new agent " << veh.id << endl;

        no_move = false;

        veh_list.push_back(veh);
        veh_list.back().vel.x = 0.01; // to avoid subsequent runtime error
        veh_list.back().vel.y = 0.01; // to avoid subsequent runtime error
        veh_list.back().time_stamp = veh.time_stamp;
    }

    if (no_move){
//        cout << __FUNCTION__ << " no_move veh "<< veh.id <<
//                " caught: vel " << veh_list[i].vel.x <<" "<< veh_list[i].vel.y << endl;
    }
}

void WorldStateTracker::updateVehPaths(const Vehicle& veh, bool doPrint){

    int i=0;
    bool no_move = true;
    for(;i<veh_list.size();i++) {
        if (veh_list[i].id==veh.id) {
            logv <<"[updateVel] updating agent path " << veh.id << endl;
            tracIntention(veh_list[i], veh, doPrint);
            break;
        }
    }

}

void WorldStateTracker::updatePed(const Pedestrian& agent, bool doPrint){
    int i=0;

    bool no_move = true;
    for(;i<ped_list.size();i++) {
        if (ped_list[i].id==agent.id) {
            logv <<"[updatePed] updating agent " << agent.id << endl;
    
            trackVel(ped_list[i], agent, no_move, doPrint);
            tracPos(ped_list[i], agent, doPrint);
            tracIntention(ped_list[i], agent, doPrint);
            tracCrossDirs(ped_list[i], agent, doPrint);
            ped_list[i].time_stamp = agent.time_stamp;

            break;
        }
        if (abs(ped_list[i].w-agent.w)<=0.1 && abs(ped_list[i].h-agent.h)<=0.1)   //overlap
        {}
    }

    if (i==ped_list.size()) {
    	no_move = false;
        //not found, new agent
   	    logv << "[updatePed] updating new agent " << agent.id << endl;

        ped_list.push_back(agent);

        ped_list.back().vel.x = 0.01; // to avoid subsequent runtime error
        ped_list.back().vel.y = 0.01; // to avoid subsequent runtime error
        ped_list.back().time_stamp = agent.time_stamp;

        updatePathPool(ped_list.back());
    }

    if (no_move){
		cout << __FUNCTION__ << " no_move agent "<< agent.id <<
				" caught: vel " << ped_list[i].vel.x <<" "<< ped_list[i].vel.y << endl;
    }
}

void WorldStateTracker::updatePedState(const Pedestrian& agent, bool doPrint){
    int i=0;

    bool no_move = true;
    for(;i<ped_list.size();i++) {
        if (ped_list[i].id==agent.id) {
            logv <<"[updatePed] updating agent " << agent.id << endl;

            trackVel(ped_list[i], agent, no_move, doPrint);
            tracPos(ped_list[i], agent, doPrint);
            ped_list[i].time_stamp = agent.time_stamp;
            break;
        }
        if (abs(ped_list[i].w-agent.w)<=0.1 && abs(ped_list[i].h-agent.h)<=0.1)   //overlap
        {}
    }

    if (i==ped_list.size()) {
    	no_move = false;
        //not found, new agent
   	    logv << "[updatePed] updating new agent " << agent.id << endl;

        ped_list.push_back(agent);
        ped_list.back().vel.x = 0.01; // to avoid subsequent runtime error
        ped_list.back().vel.y = 0.01; // to avoid subsequent runtime error
        ped_list.back().time_stamp = agent.time_stamp;
    }

    if (no_move){
		cout << __FUNCTION__ << " no_move agent "<< agent.id <<
				" caught: vel " << ped_list[i].vel.x <<" "<< ped_list[i].vel.y << endl;
    }
}

void WorldStateTracker::updatePedPaths(const Pedestrian& agent, bool doPrint){
    for(int i=0;i<ped_list.size();i++) {
        if (ped_list[i].id==agent.id) {
            logv <<"[updatePed] updating agent paths" << agent.id << endl;

            tracIntention(ped_list[i], agent, doPrint);
            tracCrossDirs(ped_list[i], agent, doPrint);
            break;
        }
    }
}

void WorldStateTracker::removeAgents() {
	ped_list.resize(0);
    veh_list.resize(0);
    model.id_map_num_paths.clear();
    model.id_map_paths.clear();
}

void WorldStateTracker::updateCar(const CarStruct car) {
    carpos=car.pos;
    carvel=car.vel;
    car_heading_dir=car.heading_dir;
    ValidateCar(__FUNCTION__);
}

void WorldStateTracker::ValidateCar(const char* func){
    if (car_odom_heading == -10) // initial value
        return; 
    if (fabs(car_heading_dir - car_odom_heading)> 0.1 && fabs(car_heading_dir - car_odom_heading)< 2*M_PI-0.1){
        ERR(string_sprintf(
            "%s: car_heading in stateTracker different from odom: %f, %f",
            func, car_heading_dir, car_odom_heading));
    }
}

bool WorldStateTracker::emergency() {
    //TODO improve emergency stop to prevent the false detection of leg
    double mindist = numeric_limits<double>::infinity();
    for(auto& agent : ped_list) {
		COORD p(agent.w, agent.h);
        double d = COORD::EuclideanDistance(carpos, p);
        if (d < mindist) mindist = d;
    }
	cout << "emergency mindist = " << mindist << endl;
	return (mindist < 0.5);
}

void WorldStateTracker::updateVel(double vel) {
	carvel = vel;
}

vector<WorldStateTracker::AgentDistPair> WorldStateTracker::getSortedAgents(bool doPrint) {
    // sort agents
    vector<AgentDistPair> sorted_agents;

    if(doPrint) cout << "[getSortedAgents] state tracker agent_list size " 
        << ped_list.size() + veh_list.size() << endl;
    
    for(auto& p: ped_list) {
        COORD cp(p.w, p.h);
        float dist = COORD::EuclideanDistance(cp, carpos);

        COORD ped_dir = COORD(p.w, p.h) - carpos;
        COORD car_dir = COORD(cos(car_heading_dir), sin(car_heading_dir));
        double proj = (ped_dir.x*car_dir.x + ped_dir.y*car_dir.y)/ ped_dir.Length();
        if (proj > 0.6)
        	dist -= 2.0;
        if (proj < -0.7)
            dist += 2.0;

        sorted_agents.push_back(AgentDistPair(dist, &p));

		if(doPrint) cout << "[getSortedAgents] agent id:"<< p.id << endl;
    }

    for(auto& veh: veh_list) {
        COORD cp(veh.w, veh.h);
        float dist = COORD::EuclideanDistance(cp, carpos);

        COORD ped_dir = COORD(veh.w, veh.h) - carpos;
        COORD car_dir = COORD(cos(car_heading_dir), sin(car_heading_dir));
        double proj = (ped_dir.x*car_dir.x + ped_dir.y*car_dir.y)/ ped_dir.Length();
        if (proj > 0.6)
            dist -= 2.0;
        if (proj < -0.7)
            dist += 2.0;

        sorted_agents.push_back(AgentDistPair(dist, &veh));

        if(doPrint) cout << "[getSortedAgents] veh id:"<< veh.id << endl;
    }

    sort(sorted_agents.begin(), sorted_agents.end(),
            [](const AgentDistPair& a, const AgentDistPair& b) -> bool {
                return a.first < b.first;
            });

    return sorted_agents;
}

void WorldStateTracker::setPomdpCar(CarStruct& car){
    car.pos = carpos;
    car.vel = carvel;
    car.heading_dir = /*0*/car_heading_dir;
}

void WorldStateTracker::setPomdpPed(AgentStruct& agent, const Agent& src){
    agent.pos.x=src.w;
    agent.pos.y=src.h;
    agent.id = src.id;
    agent.type = src.type();
    agent.vel=src.vel;
    agent.speed = ModelParams::PED_SPEED;
    agent.intention = -1; // which path to take
    agent.pos_along_path = 0.0; // travel distance along the path
    agent.cross_dir = fetch_cross_dir(src);; // which path to take
    auto bb = fetch_bounding_box(src);
    agent.heading_dir = fetch_heading_dir(src);
    // cout << __FUNCTION__ << ": type " << agent.type << ", heading "<< agent.heading_dir << endl;

    model.cal_bb_extents(agent, bb, agent.heading_dir);
}

PomdpState WorldStateTracker::getPomdpState() {

    auto sorted_agents = getSortedAgents();

    // construct PomdpState
    PomdpState pomdpState;

    setPomdpCar(pomdpState.car);
    
    pomdpState.num = min((int)sorted_agents.size(), ModelParams::N_PED_IN); //current_state.num = num_of_peds_world;

    for(int i=0;i<pomdpState.num;i++) {
      const auto& agent = *(sorted_agents[i].second);
      setPomdpPed(pomdpState.agents[i], agent);
    }

	return pomdpState;
}

PomdpStateWorld WorldStateTracker::getPomdpWorldState() {
    // cout << __FUNCTION__  << endl;


    auto sorted_agents = getSortedAgents();

    // text(sorted_agents);

    PomdpStateWorld worldState;

    setPomdpCar(worldState.car);
    
    worldState.num = min((int)sorted_agents.size(), ModelParams::N_PED_WORLD); //current_state.num = num_of_peds_world;

    for(int i=0;i<worldState.num;i++) {
      const auto& agent = *(sorted_agents[i].second);
      setPomdpPed(worldState.agents[i], agent);
    }
    
    return worldState;
}

void AgentBelief::reset_belief(int new_size){
    reset = true;
    double accum_prob = 0;

    if (new_size != prob_modes_goals[0].size()){
        for (auto& prob_goals: prob_modes_goals){ 
            cout << "Resizing prob_goals from "<< prob_goals.size() << " to " << new_size << endl;
            prob_goals.resize(new_size);
            // prob_goals.clear();
            // cout << "Clear complete" << endl;

            // for (int i=0;i<new_size;i++)
                // prob_goals.push_back(0.0);
            cout << "Resize complete" << endl;
        }
    }

    for (auto& prob_goals: prob_modes_goals){
        std::fill(prob_goals.begin(), prob_goals.end(), 1.0);

        accum_prob += accumulate(prob_goals.begin(),prob_goals.end(),0);
    }

    assert(accum_prob != 0);

    // normalize distribution
    for (auto& prob_goals: prob_modes_goals){
        for (auto& prob : prob_goals)
            prob = prob / accum_prob;
    }
}

void WorldBeliefTracker::update() {

    // DEBUG("Update agent_beliefs");
    auto sorted_agents = stateTracker.getSortedAgents();

    if (logging::level()>=logging::VERBOSE){
    	logi << "belief update start" << endl;
    	stateTracker.text(sorted_agents);
    }

    map<int, const Agent*> newagents;
    for(WorldStateTracker::AgentDistPair& dp: sorted_agents) {
        auto p = dp.second;
        newagents[p->id] = p;
    }

    // DEBUG("remove disappeared agent_beliefs");
    vector<int> agents_to_remove;

    for(const auto& p: agent_beliefs) {
        if (newagents.find(p.first) == newagents.end()) {
            logi << "["<<__FUNCTION__<< "]" << " removing agent "<< p.first << endl;
            agents_to_remove.push_back(p.first);
            // agent_beliefs.erase(p.first);
        }
    }

    for(const auto& i: agents_to_remove) {
        agent_beliefs.erase(i);
    }

    // DEBUG("reset agent belief if paths are updated");

    for(auto& dp: sorted_agents) {
        auto& agent = *dp.second;
        if (agent_beliefs.find(agent.id) != agent_beliefs.end()){
            if (agent.reset_intention) {   
                agent_beliefs[agent.id].reset_belief(model.GetNumIntentions(agent.id));
                logi << "["<<__FUNCTION__<< "]" << " belief reset: agent "<< agent.id << endl;
            }
            else{
                agent_beliefs[agent.id].reset = false;
            }
        }
    }    

    if (logging::level()>=logging::VERBOSE){
		text(agent_beliefs);
	}
    //

    // DEBUG("Run ORCA for all possible hidden variable combinations");
    model.PrepareAttentiveAgentMeanDirs(agent_beliefs, car);

    model.PrintMeanDirs(agent_beliefs, newagents);

    // update car
    car.pos = stateTracker.carpos;
    car.vel = stateTracker.carvel;
	car.heading_dir = /*0*/stateTracker.car_heading_dir;

//    DEBUG("Before belief update");
    stateTracker.model.print_path_map();

    // DEBUG("update existing agent_beliefs");

    for(auto& kv : agent_beliefs) {
    	if (newagents.find(kv.first) == newagents.end()){
            ERR(string_sprintf("updating non existing id %d in new agent list", kv.first))
    	}
        model.updatePedBelief(kv.second, *newagents[kv.first]);
    }

//    DEBUG("add new agent_beliefs");
    // 
    for(const auto& kv: newagents) {
		auto& p = *kv.second;
        if (agent_beliefs.find(p.id) == agent_beliefs.end()) {
        	logi << "Init belief entry for new agent " << p.id << endl;
            agent_beliefs[p.id] = model.initPedBelief(p);
        }
    }

    // DEBUG("maintain sorted peds");
    // 
	sorted_beliefs.clear();
	for(const auto& dp: sorted_agents) {
		auto& p = *dp.second;
		sorted_beliefs.push_back(&agent_beliefs[p.id]);
	}

	cur_time_stamp = Globals::ElapsedTime();

	if (logging::level()>=logging::VERBOSE){
		logi << "belief update end" << endl;
	}

    return;
}

int AgentBelief::sample_goal() const {
    double r = /*double(rand()) / RAND_MAX*/ Random::RANDOM.NextDouble();
    // int i = 0;
    // r -= prob_goals[i];
    // while(r > 0) {
    //     i++;
    //     r -= prob_goals[i];
    // }

    for (auto prob_goals: prob_modes_goals){
        int i = 0;
        for (int prob: prob_goals){
            r -= prob; 
            if (r <= 0) 
                return i;
            i++;      
        }
    }

    return prob_modes_goals[0].size() - 1; // stop intention as default
}

void AgentBelief::sample_goal_mode(int& goal, int& mode, bool use_att_mode) const {
//	logv << "[AgentBelief::sample_goal_mode] " << endl;

    double r = Random::RANDOM.NextDouble();

    if (use_att_mode){
        // sample full modes
        bool done = false;

        double total_prob = 0;
        for (auto& goal_probs: prob_modes_goals){
            // total_prob += std::accumulate(goal_probs.begin(), goal_probs.end(), 0);
            for (auto p: goal_probs)
                total_prob += p;
        }

        assert(total_prob!=0);

        // cout << prob_modes_goals << endl;
        // cout << __FUNCTION__ << " total_prob=" << total_prob << endl; 

        r = r * total_prob;
        
        for (int ped_type = 0 ; ped_type < prob_modes_goals.size() ; ped_type++){
            auto& goal_probs = prob_modes_goals[ped_type];
            for (int ped_goal = 0 ; ped_goal < goal_probs.size() ; ped_goal++){
                r -= prob_modes_goals[ped_type][ped_goal];
                if(r <= 0.001) {
                    goal = ped_goal;
                    mode = ped_type;
                    done = true;
                    break;
                }
            }
            if (done) break;
        }

        if(r > 0) {
            logv << "[WARNING]: [AgentBelief::sample_goal_mode] execess probability " << r << endl;
            goal = 0;
            mode = 0;
        } 

        if(goal > 10){
            cout << "the rest r = " << r << endl;
        }
    } else { // only sample for attentive mode
        int ped_type = AGENT_DIS;
        auto& goal_probs = prob_modes_goals[ped_type];
        double total_prob = 0; //accumulate(goal_probs.begin(), goal_probs.end(), 0);
        for (auto p: goal_probs)
            total_prob += p;

        assert(total_prob!=0);
        r = r * total_prob;
        for (int ped_goal = 0 ; ped_goal < goal_probs.size() ; ped_goal++){
            r -= prob_modes_goals[ped_type][ped_goal];
            if(r <= 0) {
                goal = ped_goal;
                mode = ped_type;
                break;
            }
        }
    }  
}

int AgentBelief::maxlikely_intention() const {
    double ml = 0;
    int mode = AGENT_DIS;
    auto& prob_goals = prob_modes_goals[mode];
    int mi = prob_goals.size()-1; // stop intention
    for(int i=0; i<prob_goals.size(); i++) {
        if (prob_goals[i] > ml && prob_goals[i] > 0.5) {
            ml = prob_goals[i];
            mi = i;
        }
    }
    return mi;
}

void WorldBeliefTracker::printBelief() const {
    //return;
	logi << sorted_beliefs.size() << " entries in sorted_beliefs:" << endl;
	int num = 0;
    for(int i=0; i < sorted_beliefs.size() && i < ModelParams::N_PED_IN; i++) {
		auto& p = *sorted_beliefs[i];
		if (COORD::EuclideanDistance(p.pos, car.pos) <= ModelParams::LASER_RANGE) {
            cout << "agent belief " << p.id << ": ";

            for (auto& prob_goals : p.prob_modes_goals){
                for (int g = 0; g < prob_goals.size(); g ++)
                    cout << " " << prob_goals[g];
                cout << endl;
            }
		}
    }
}

PomdpState WorldBeliefTracker::text() const{
	if (logging::level()>=logging::VERBOSE){
		for(int i=0; i < sorted_beliefs.size() && i < min(20,ModelParams::N_PED_IN); i++) {
			auto& p = *sorted_beliefs[i];
			cout << "[WorldBeliefTracker::text] " << this << "->p:" << &p << endl;
			cout << " sorted agent " << p.id << endl;

			cout << "-prob_modes_goals: " << p.prob_modes_goals << endl;
			// for (int i=0;i<p.prob_modes_goals.size();i++){
			// 	for (int j=0;j<p.prob_modes_goals[i].size();j++){
			// 		cout << p.prob_modes_goals[i][j] << " ";
			// 	}
			// 	cout<< endl;
			// }
		}
	}
}

void WorldBeliefTracker::text(const std::map<int, AgentBelief>& agent_bs) const{
    if (logging::level()>=logging::VERBOSE){
        cout << "=> Agent beliefs: " << endl;
        for (auto itr = agent_bs.begin(); itr != agent_bs.end(); ++itr) {
            int id = itr->first;
            auto& b = itr->second;
            fprintf(stderr, "==> id / type / pos / vel / heading / reset / cross : %d / %d / (%f %f) / (%f %f) / %f / %d / %d \n",  
                b.id, b.type, b.pos.x, b.pos.y, b.vel.x, b.vel.y, b.heading_dir, b.reset, b.cross_dir);  
        }
    }
}

void WorldBeliefTracker::ValidateCar(const char* func){
    if (stateTracker.car_odom_heading == -10) // initial value
        return;
    if (fabs(car.heading_dir - stateTracker.car_odom_heading)> 0.1){
        ERR(string_sprintf(
            "%s: car_heading in stateTracker different from odom: %f, %f",
            func, car.heading_dir, stateTracker.car_odom_heading));
    }
}

PomdpState WorldBeliefTracker::sample(bool predict, bool use_att_mode) {
    ValidateCar(__FUNCTION__);

    PomdpState s;
    s.car = car;
	s.num = 0;

    for(int i=0; i < sorted_beliefs.size() && i < ModelParams::N_PED_IN; i++) {
		auto& p = *sorted_beliefs[i];

        if (p.type >= AgentType::num_values){
            ERR(string_sprintf("non-initialized type in state: %d", p.type));
        }   

		if (COORD::EuclideanDistance(p.pos, car.pos) < ModelParams::LASER_RANGE) {
			s.agents[s.num].pos = p.pos;

            assert(p.prob_modes_goals.size() == 2);
            for (auto& prob_goals: p.prob_modes_goals){
                if(prob_goals.size()!=model.GetNumIntentions(p.id)){
                    DEBUG(string_sprintf("prob_goals.size()!=model.GetNumIntentions(p.id): %d, %d, id %d",
                        prob_goals.size(), model.GetNumIntentions(p.id), p.id));
                    p.reset_belief(model.GetNumIntentions(p.id));
                }
            }
			p.sample_goal_mode(s.agents[s.num].intention, s.agents[s.num].mode
                , use_att_mode);
            model.ValidateIntention(p.id, s.agents[s.num].intention, __FUNCTION__, __LINE__);
			
            s.agents[s.num].id = p.id;
            s.agents[s.num].vel = p.vel;
            s.agents[s.num].speed = p.speed;
            s.agents[s.num].pos_along_path = 0; // assuming that paths are up to date here
            s.agents[s.num].cross_dir = p.cross_dir;
            s.agents[s.num].type = p.type;
            s.agents[s.num].heading_dir = p.heading_dir;
            model.cal_bb_extents(p, s.agents[s.num]);
			s.num ++;
		}
    }

    s.time_stamp  = cur_time_stamp;

//	logv << "[WorldBeliefTracker::sample] done" << endl;
    if (predict){
    	PomdpState predicted_s = predictPedsCurVel(&s, cur_acc, cur_steering);
    	return predicted_s;
    }

    return s;
}

vector<PomdpState> WorldBeliefTracker::sample(int num, bool predict, bool use_att_mode) {

	if(DESPOT::Debug_mode)
		Random::RANDOM.seed(0);

    vector<PomdpState> particles;
	logv << "[WorldBeliefTracker::sample] Sampling" << endl;

    for(int i=0; i<num; i++) {
        particles.push_back(sample(predict, use_att_mode));
    }

    cout << "Num agents for planning: " << particles[0].num << endl;

    return particles;
}

vector<AgentStruct> WorldBeliefTracker::predictAgents() {
    vector<AgentStruct> prediction;

    for(const auto ptr: sorted_beliefs) {
        const auto& p = *ptr;
        double dist = COORD::EuclideanDistance(p.pos, car.pos);
        int step = (p.speed + car.vel>1e-5)?int(dist / (p.speed + car.vel) * ModelParams::control_freq):100000;

        int intention_id = p.maxlikely_intention();
        AgentStruct agent_pred(p.pos, intention_id, p.id);
		agent_pred.vel = p.vel;
        agent_pred.speed =  p.speed;
        agent_pred.type = p.type;

        for(int i=0; i<4; i++) {
            AgentStruct agent = agent_pred;

            if(model.goal_mode == "goal"){
                model.AgentStepGoal(agent, step + i);
            }
            else if (model.goal_mode == "cur_vel"){
                model.PedStepCurVel(agent, step + i);
            }
            else if (model.goal_mode == "path"){
                model.AgentStepPath(agent, step + i);
            }
            
            prediction.push_back(agent);
        }
    }

    return prediction;
}

PomdpState WorldBeliefTracker::predictPedsCurVel(PomdpState* ped_state, double acc, double steering) {

//	cout << __FUNCTION__ << "applying action " << acc <<"/"<< steering << endl;

	PomdpState predicted_state = *ped_state;

//	if (acc >0){
//		cout << "source agent list addr "<< ped_state->agents << endl;
//		cout << "des agent list addr "<< predicted_state.agents << endl;
//		cout << predicted_state.num << " agents active" << endl;
//		raise(SIGABRT);
//	}

//	model.RobStepCurVel(predicted_state.car);
	model.RobStepCurAction(predicted_state.car, acc, steering);

	for(int i =0 ; i< predicted_state.num ; i++) {
		auto & p = predicted_state.agents[i];

//		if (acc >0 ){
//			if (ped_vel.Length()<1e-3){
//				cout << "agent " << p.id << " vel = " << ped_vel.Length() << endl;
//				cout.flush();
//				raise(SIGABRT);
//			}
//			else{
//				cout << "agent " << p.id << " vel = " << ped_vel.x <<" "<< ped_vel.y << endl;
//			}
//		}

		model.PedStepCurVel(p/*, ped_vel*/);
    }

    predicted_state.time_stamp = ped_state->time_stamp + 1.0 / ModelParams::control_freq;

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

    return predicted_state;
}

double GenerateGaussian(double rNum)
{
    if(FIX_SCENARIO!=1 && !CPUDoPrint)
        rNum=QuickRandom::RandGeneration(rNum);
    double result = sqrt(-2 * log(rNum));
    if(FIX_SCENARIO!=1 && !CPUDoPrint)
        rNum=QuickRandom::RandGeneration(rNum);

    result*= cos(2 * M_PI * rNum);
    return result;
}


void WorldBeliefTracker::PrintState(const State& s, ostream& out) const {
	const PomdpState & state=static_cast<const PomdpState&> (s);
    const COORD& carpos = state.car.pos;

	out << "Rob Pos: " << carpos.x<< " " <<carpos.y << endl;
	out << "Rob heading direction: " << state.car.heading_dir << endl;
	for(int i = 0; i < state.num; i ++) {
		out << "Ped Pos: " << state.agents[i].pos.x << " " << state.agents[i].pos.y << endl;
		out << "Goal: " << state.agents[i].intention << endl;
		out << "id: " << state.agents[i].id << endl;
	}
	out << "Vel: " << state.car.vel << endl;
	out<<  "num  " << state.num << endl;
	double min_dist = COORD::EuclideanDistance(carpos, state.agents[0].pos);
	out << "MinDist: " << min_dist << endl;
}

void WorldModel::ValidateIntention(int agent_id, int intention_id, const char* msg, int line){
    if (intention_id >= GetNumIntentions(agent_id)){
        int num_path = NumPaths(agent_id);
        int num_intention = GetNumIntentions(agent_id);
        ERR(string_sprintf("%s@%d: Intention ID excess # intentions: %d, # intention = %d, # path = %d", 
            msg, line, intention_id, num_intention, num_path));
    }
}


void WorldModel::PorcaSimulateAgents(AgentStruct agents[], int num_agents, CarStruct& car){

    // DEBUG("start_rvo_sim");
    int threadID=GetThreadID();

    // Construct a new set of agents every time
    traffic_agent_sim_[threadID]->clearAllAgents();

    //adding pedestrians
    for(int i=0; i<num_agents; i++){
        if(agents[i].mode==AGENT_ATT){
            traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(agents[i].pos.x, agents[i].pos.y));
            int intention_id = agents[i].intention;
            ValidateIntention(agents[i].id, intention_id, __FUNCTION__, __LINE__);
            if (intention_id == GetNumIntentions(agents[i])-1) { /// stop intention
                traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
            } else{
                auto goal_pos = GetGoalPos(agents[i], intention_id);
                RVO::Vector2 goal(goal_pos.x, goal_pos.y);
                if ( absSq(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) < 
                    traffic_agent_sim_[threadID]->getAgentRadius(i) * traffic_agent_sim_[threadID]->getAgentRadius(i) ) {
                    // Agent is within one radius of its goal, set preferred velocity to zero
                    traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
                } else {
                    // Agent is far away from its goal, set preferred velocity as unit vector towards i's goal.
                    double pref_speed = 0.0;
                    if (agents[i].type == AgentType::car)
                        pref_speed = 5.0;
                    else if (agents[i].type == AgentType::ped)
                        pref_speed = ModelParams::PED_SPEED;
                    traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, 
                        normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(i))
                        */*agents[i].speed*/pref_speed);
                }
            }
        }
        else {// distracted agents should also exist in the tree
            traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(agents[i].pos.x, agents[i].pos.y));
            COORD pref_vel = GetGoalPos(agents[i]) - agents[i].pos;
            pref_vel.AdjustLength(agents[i].speed);
            traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(pref_vel.x, pref_vel.y));
        }
    }
    // adding car as a "special" pedestrian
    add_car_agent(num_agents, car);

    traffic_agent_sim_[threadID]->doStep();

    // DEBUG("End rvo sim");
}

COORD WorldModel::GetPorcaVel(AgentStruct& agent, int i){
    // DEBUG("End rvo vel");
    int threadID=GetThreadID();
    assert(agent.mode==AGENT_ATT);

    COORD new_pos;
    new_pos.x=traffic_agent_sim_[threadID]->getAgentPosition(i).x();// + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(agent).x() - agents[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
    new_pos.y=traffic_agent_sim_[threadID]->getAgentPosition(i).y();// + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(agent).y() - agents[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

    // DEBUG("End rvo vel");
    return (new_pos - agent.pos) * freq; 
}

void WorldModel::AgentApplyPorcaVel(AgentStruct& agent, COORD& rvo_vel) {
    // DEBUG("Start rvo apply vel");
    COORD old_pos = agent.pos;
    double rvo_speed = rvo_vel.Length();
    if (agent.type == AgentType::car){
        rvo_vel.AdjustLength(3.0);
        COORD pursuit_point = agent.pos + rvo_vel;
        double steering = PControlAngle<AgentStruct>(agent, pursuit_point);
        // steering += noise;
        BicycleModel(agent, steering, rvo_speed);
    }
    else if (agent.type == AgentType::ped) {
        agent.pos = agent.pos + rvo_vel * (1.0/freq);
    }

    if (agent.intention != GetNumIntentions(agent.id) - 1) {
        auto& path = PathCandidates(agent.id)[agent.intention];
        agent.pos_along_path = path.nearest(agent.pos);
    }
    agent.vel = (agent.pos - old_pos)*freq;

    // DEBUG("End rvo apply vel");
}

void WorldModel::PorcaAgentStep(PomdpStateWorld& state, Random& random){
    PorcaAgentStep(state.agents, random, state.num, state.car);
}

void WorldModel::PorcaAgentStep(AgentStruct agents[], Random& random, int num_agents, CarStruct car){
    PorcaSimulateAgents(agents, num_agents, car);

    for (int i = 0; i < num_agents; ++i){
        auto& agent = agents[i];
        if(agent.mode == AGENT_ATT){
            COORD rvo_vel = GetPorcaVel(agent, i);
            AgentApplyPorcaVel(agent,rvo_vel);
            if(use_noise_in_rvo){
                agent.pos.x+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
                agent.pos.y+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
            }
        }
    }
}

void WorldModel::PorcaAgentStep(AgentStruct agents[], double& random, int num_agents, CarStruct car){
    PorcaSimulateAgents(agents, num_agents, car);

    for (int i = 0; i < num_agents; ++i){
        auto& agent = agents[i];
        if(agent.mode == AGENT_ATT){
            COORD rvo_vel = GetPorcaVel(agent, i);
            AgentApplyPorcaVel(agent, rvo_vel);
            if(use_noise_in_rvo){
                double rNum=GenerateGaussian(random);
                agent.pos.x+= rNum * ModelParams::NOISE_PED_POS / freq;
                rNum=GenerateGaussian(rNum);
                agent.pos.y+= rNum * ModelParams::NOISE_PED_POS / freq;
            }
        }
    }
}

COORD WorldModel::DistractedPedMeanDir(AgentStruct& agent, int intention_id) {
	COORD dir(0,0);
	const COORD& goal = GetGoalPos(agent, intention_id);
	if (intention_id == GetNumIntentions(agent.id)-1) {  //stop intention
		return dir;
	}

	MyVector goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);
//	goal_vec.AdjustLength(agent.vel / freq);

	dir.x = goal_vec.dw;
	dir.y = goal_vec.dh;

	return dir;
}

COORD WorldModel::AttentivePedMeanDir(int agent_id, int intention_id){
	return ped_mean_dirs[agent_id][intention_id];
}

#include "Vector2.h"

void WorldModel::add_car_agent(int id_in_sim, CarStruct& car){
    int threadID=GetThreadID();

	double car_x, car_y, car_yaw;

	car_yaw = car.heading_dir;

	if(ModelParams::car_model == "pomdp_car"){
		/// for pomdp car
		car_x = car.pos.x - ModelParams::CAR_LENGTH/2.0f*cos(car_yaw);
		car_y = car.pos.y - ModelParams::CAR_LENGTH/2.0f*sin(car_yaw);
		double car_radius = sqrt(pow(ModelParams::CAR_WIDTH/2.0f, 2) + pow(ModelParams::CAR_LENGTH/2.0f,2)) + CAR_EXPAND_SIZE;
		traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw)));
	} else if(ModelParams::car_model == "audi_r8"){
		/// for audi r8
		double car_radius = 1.15f;
		double car_radius_large = 1.6f;

		traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x, car.pos.y), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the id_in_sim-th pedestrian is the car. set its prefered velocity
		traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim,-1);

		traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x + 0.56 * 2.33 * cos(car_yaw), car.pos.y + 1.4* sin(car_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+1, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the id_in_sim-th pedestrian is the car. set its prefered velocity
		traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+1,-2);

		traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x + 0.56*3.66 * cos(car_yaw), car.pos.y + 2.8* sin(car_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+2, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the id_in_sim-th pedestrian is the car. set its prefered velocity
		traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+2,-3);

		traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x + 0.56*5 * cos(car_yaw), car.pos.y + 2.8* sin(car_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius_large, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+3, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the id_in_sim-th pedestrian is the car. set its prefered velocity
		traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+3,-4);
	}
    else if(ModelParams::car_model == "summit"){

        //TODO: this should use the car dimension from topic
        double car_radius = 1.15f;
        double car_radius_large = 1.6f;
        double shift = 1.0;

        traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x, car.pos.y), 4.0f, 2, 1.0f, 2.0f, 
            car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
        traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the id_in_sim-th pedestrian is the car. set its prefered velocity
        traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim,-1);

        traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x + shift * cos(car_yaw), car.pos.y + shift * sin(car_yaw)), 4.0f, 2, 1.0f, 2.0f, 
            car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
        traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+1, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the id_in_sim-th pedestrian is the car. set its prefered velocity
        traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+1,-2);

        traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x - shift * cos(car_yaw), car.pos.y - shift * sin(car_yaw)), 4.0f, 2, 1.0f, 2.0f, 
            car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
        traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+2, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the id_in_sim-th pedestrian is the car. set its prefered velocity
        traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+2,-3);
    }
}

double get_heading_dir(COORD vel){

}

void WorldModel::add_veh_agent(AgentBelief& veh){

    if(veh.type != AgentType::car)
        return;

    int threadID=GetThreadID();

    double veh_x, veh_y, veh_yaw;

    veh_yaw = veh.heading_dir;

    auto& veh_bb = veh.bb;

    // TODO: replace following to use the bonding box data.

    double car_radius = 1.15f;
    double car_radius_large = 1.6f;

    // DEBUG("agent circle");
    traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(veh.pos.x, veh.pos.y), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    // DEBUG("agent pref");
    // auto pref_vel = RVO::Vector2(veh.vel.Length() * cos(veh_yaw), veh.vel.Length() * sin(veh_yaw));
    // DEBUG("set pref");
    // traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim, pref_vel 
    //     ); // the id_in_sim-th pedestrian is the veh. set its prefered velocity
    // DEBUG("agent id");
    // traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim,-1);

    // DEBUG("next");
    // traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(veh.pos.x + 0.56 * 2.33 * cos(veh_yaw), veh.pos.y + 1.4* sin(veh_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    // traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+1, 
    //     RVO::Vector2(veh.vel.Length() * cos(veh_yaw), veh.vel.Length() * sin(veh_yaw))); // the id_in_sim-th pedestrian is the veh. set its prefered velocity
    // traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+1,-2);

    // traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(veh.pos.x + 0.56*3.66 * cos(veh_yaw), veh.pos.y + 2.8* sin(veh_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    // traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+2, 
    //     RVO::Vector2(veh.vel.Length() * cos(veh_yaw), veh.vel.Length() * sin(veh_yaw))); // the id_in_sim-th pedestrian is the veh. set its prefered velocity
    // traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+2,-3);

    // traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(veh.pos.x + 0.56*5 * cos(veh_yaw), veh.pos.y + 2.8* sin(veh_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius_large, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    // traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+3, 
    //     RVO::Vector2(veh.vel.Length() * cos(veh_yaw), veh.vel.Length() * sin(veh_yaw))); // the id_in_sim-th pedestrian is the veh. set its prefered velocity
    // traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+3,-4);
}

void WorldModel::PrepareAttentiveAgentMeanDirs(std::map<int, AgentBelief> agents, CarStruct& car){
	int num_agents = agents.size();

	logi << "[PrepareAttentiveAgentMeanDirs] num_agents in belief tracker: " << num_agents << endl;

    if (num_agents == 0)
        return;

    int threadID=GetThreadID();

    // Construct a new set of agents every time
	traffic_agent_sim_[threadID]->clearAllAgents();

    // DEBUG("adding agents to gamma");
	//
    std::vector<int> agent_ids;
    agent_ids.resize(num_agents);

    int i = 0;
    for (auto it = agents.begin(); it != agents.end(); it++) {
        auto & agent = it->second;
        switch (agent.type) {
            case AgentType::ped:
                // DEBUG("add ped");
                traffic_agent_sim_[threadID]->addAgent(
                    RVO::Vector2(agent.pos.x, agent.pos.y)); break;
            case AgentType::car:
                // DEBUG("add car");
                add_veh_agent(agent); break;
            default: 
                ERR("unsupported agent type");
                break;
        }

        agent_ids[i] = agent.id;
        // TODO: need to assign bounding boxes to vehicles
        
        logv << "agent " << i << " id "<< agent_ids[i] << endl;
        i++;
    }

    // DEBUG("adding car as a "special" agent");
	// 
	add_car_agent(num_agents, car);

    // DEBUG("Set the preferred velocity for each agent");
    // 
	traffic_agent_sim_[threadID]->doPreStep();// build kd tree and find neighbors for agents

    // DEBUG("Ensure the mean_dir container entry exist for all agents");
    for (int i = 0; i < num_agents ; i++){
        EnsureMeanDirExist(agent_ids[i]);
    }

    // DEBUG("predict directions for all agents and all intentions");
	for (size_t i = 0; i < num_agents; ++i) {

        int id = agent_ids[i];

		// For each ego agent
	    for(int intention_id=0; intention_id < GetNumIntentions(id); intention_id++) {

	    	RVO::Vector2 ori_pos(traffic_agent_sim_[threadID]->getAgentPosition(i).x(),  
                traffic_agent_sim_[threadID]->getAgentPosition(i).y());

            // DEBUG("Set preferred velocity for the ego agent according to intention_id");
			// Leave other pedestrians to have default preferred velocity
            ValidateIntention(id, intention_id, __FUNCTION__, __LINE__);
			if (intention_id == GetNumIntentions(id)-1) { /// stop intention
				traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
			} else{
                auto goal_pos = GetGoalPos(agents[id], intention_id);
				RVO::Vector2 goal(goal_pos.x, goal_pos.y);
				if ( absSq(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) < 
                    traffic_agent_sim_[threadID]->getAgentRadius(i) * traffic_agent_sim_[threadID]->getAgentRadius(i) ) {
					// Agent is within one radius of its goal, set preferred velocity to zero
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
				} else {
					// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(i))*ModelParams::PED_SPEED);
				}
			}

            // DEBUG("step gamma");
			traffic_agent_sim_[threadID]->doStepForPed(i); //TODO: this should be replace by GAMMA functions

			COORD dir;
			dir.x = traffic_agent_sim_[threadID]->getAgentPosition(i).x() - agents[id].pos.x;
			dir.y = traffic_agent_sim_[threadID]->getAgentPosition(i).y() - agents[id].pos.y;

			logv << "[PrepareAttentiveAgentMeanDirs] ped_mean_dirs len=" << ped_mean_dirs.size()
					<< " intention_list len=" << ped_mean_dirs[id].size() << "\n";

			logv << "[PrepareAttentiveAgentMeanDirs] i=" << i << " intention_id=" << intention_id << "\n";

            // DEBUG(string_sprintf("set mean, i=%d, intention_id=%d, ped_mean_dirs.size()=%d \n",
            //     i, intention_id, ped_mean_dirs.size()));

			ped_mean_dirs[id][intention_id]=dir;

            // DEBUG("reset agent state");
	    	// 
			traffic_agent_sim_[threadID]->setAgentPosition(i, ori_pos);
	    }
	}
}

void WorldModel::PrintMeanDirs(std::map<int, AgentBelief> old_agents, 
    map<int, const Agent*>& curr_agents){
	if(logging::level()>=logging::VERBOSE){
		int num_agents = old_agents.size();

        int count = 0;
        for (std::map<int, const Agent*>::iterator it=curr_agents.begin(); it!=curr_agents.end(); ++it) {

            if (count == 6) break;

            int agent_id = it->first;

            if ( old_agents.find(agent_id) == old_agents.end() )
                continue;

            cout <<"agent  "<< agent_id << endl;
            auto& cur_agent = *it->second;
            // For each ego agent
            auto& old_agent = old_agents[agent_id];
    					
    		cout << "prev pos: "<< old_agent.pos.x << "," << old_agent.pos.y << endl;
    		cout << "cur pos: "<< cur_agent.w << "," << cur_agent.h << endl;

    		COORD dir = COORD(cur_agent.w, cur_agent.h) - old_agent.pos;

    		cout << "dir: " << endl;
    		for(int intention_id=0; intention_id<GetNumIntentions(cur_agent.id); intention_id++) {
    			cout <<intention_id<<"," << dir.x << "," << dir.y << " ";
    		}
    		cout<< endl;

    		cout << "Meandir: " << endl;
    	    for(int intention_id=0; intention_id<GetNumIntentions(cur_agent.id); intention_id++) {
    	    	cout <<intention_id<<","<<ped_mean_dirs[agent_id][intention_id].x << ","<< ped_mean_dirs[agent_id][intention_id].y << " ";
    	    }
    	    cout<< endl;

    		cout << "Att probs: " << endl;
    	    for(int intention_id=0; intention_id<GetNumIntentions(cur_agent.id); intention_id++) {
    	    	logv <<"agent, goal, ped_mean_dirs.size() =" << cur_agent.id << " "
    	    					<< intention_id <<" "<< ped_mean_dirs.size() << endl;
    			double prob = agentMoveProb(old_agent.pos, cur_agent, intention_id, AGENT_ATT);

    			cout <<"prob: "<< intention_id<<","<<prob << endl;
    		}
    	    cout<< endl;

            count ++;
        }
	}
}

void WorldModel::AddObstacle(std::vector<RVO::Vector2> obs){
	for (auto& point:obs){
		obstacles.push_back(COORD(point.x(), point.y()));
	}
	// to seperate different obstacles
	obstacles.push_back(COORD(Globals::NEG_INFTY, Globals::NEG_INFTY));
}

bool WorldModel::CheckCarWithObstacles(const CarStruct& car, int flag){

    return false; // only for the summit project.


	COORD obs_first_point(Globals::NEG_INFTY, Globals::NEG_INFTY);
	COORD obs_last_point;

	for (auto& point: obstacles){
		if (obs_first_point.x == Globals::NEG_INFTY){ // first point of an obstacle
			obs_first_point = point;
		}
		else if (point.x == Globals::NEG_INFTY){ // stop point of an obstacle
			if(CheckCarWithObsLine(car, obs_last_point, obs_first_point, flag))
				return true;
			obs_first_point = COORD(Globals::NEG_INFTY, Globals::NEG_INFTY);
		}
		else{ // normal obstacle points
			if(CheckCarWithObsLine(car, obs_last_point, point, flag))
				return true;
		}
		obs_last_point = point;
	}

	return false;
}


bool WorldModel::CheckCarWithVehicle(const CarStruct& car, const AgentStruct& veh, int flag) {
    // if (!inFront(veh.pos, car, infront_angle_deg))
    //     return false;

    // double side_margin, front_margin, back_margin;

    // // flag = 1; // debugging
    // if(flag == 1){ // real collision
    //     side_margin = ModelParams::CAR_WIDTH / 2.0;
    //     front_margin = ModelParams::CAR_FRONT;
    //     back_margin = ModelParams::CAR_FRONT;   
    // }
    // else if (flag == 0) { // in search 
    //     side_margin = ModelParams::CAR_WIDTH / 2.0 + CAR_SIDE_MARGIN;
    //     front_margin = ModelParams::CAR_FRONT + CAR_FRONT_MARGIN;
    //     back_margin = ModelParams::CAR_FRONT + CAR_SIDE_MARGIN;
    // }

    // double ref_to_front_side = sqrt(front_margin*front_margin + side_margin*side_margin);
    // double ref_to_back_side = sqrt(back_margin*back_margin + side_margin*side_margin);
    // double ref_front_side_angle = atan(side_margin/front_margin);      
    // double ref_back_side_angle = atan(side_margin/back_margin);
 
    // std::vector<COORD> car_rect = ::ComputeRect(car.pos, car.heading_dir, ref_to_front_side, ref_to_back_side,
    //     ref_front_side_angle, ref_back_side_angle);

    // side_margin = veh.bb_extent_x;
    // front_margin = veh.bb_extent_y;
    // back_margin = veh.bb_extent_y;

    // ref_to_front_side = sqrt(front_margin*front_margin + side_margin*side_margin);
    // ref_to_back_side = sqrt(back_margin*back_margin + side_margin*side_margin);
    // ref_front_side_angle = atan(side_margin/front_margin);      
    // ref_back_side_angle = atan(side_margin/back_margin);
 
    // std::vector<COORD> veh_rect = ::ComputeRect(veh.pos, veh.heading_dir, ref_to_front_side, ref_to_back_side,
    //     ref_front_side_angle, ref_back_side_angle);

    // return ::InCollision(car_rect, veh_rect);
    
    // bool result = false;
    
    COORD tan_dir(-sin(veh.heading_dir), cos(veh.heading_dir));
    COORD along_dir(cos(veh.heading_dir), sin(veh.heading_dir));

    COORD test;

    bool result = false;
    test = veh.pos + tan_dir * veh.bb_extent_x + along_dir * veh.bb_extent_y;
    if(::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y, car.heading_dir, ModelParams::CAR_WIDTH/2.0, ModelParams::CAR_FRONT, flag)){
        // return true;
        result = true;
    }
    test = veh.pos - tan_dir * veh.bb_extent_x + along_dir * veh.bb_extent_y;
    if(::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y, car.heading_dir, ModelParams::CAR_WIDTH/2.0, ModelParams::CAR_FRONT, flag)){
        // return true;
        result = true;
    }
    test = veh.pos + tan_dir * veh.bb_extent_x - along_dir * veh.bb_extent_y;
    if(::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y, car.heading_dir, ModelParams::CAR_WIDTH/2.0, ModelParams::CAR_FRONT, flag)){
        // return true;
        result = true;
    }
    test = veh.pos - tan_dir * veh.bb_extent_x - along_dir * veh.bb_extent_y;
    if(::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y, car.heading_dir, ModelParams::CAR_WIDTH/2.0, ModelParams::CAR_FRONT, flag)){
        // return true;
        result = true;
    }

    tan_dir = COORD(-sin(car.heading_dir), cos(car.heading_dir));
    along_dir = COORD(cos(car.heading_dir), sin(car.heading_dir));

    test = car.pos + tan_dir * (ModelParams::CAR_WIDTH/2.0) + along_dir * ModelParams::CAR_FRONT;
    if(::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y, veh.heading_dir, veh.bb_extent_x, veh.bb_extent_y, flag)){
        // return true;
        result = true;
    }
    test = car.pos - tan_dir * (ModelParams::CAR_WIDTH/2.0) + along_dir * ModelParams::CAR_FRONT;
    if(::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y, veh.heading_dir, veh.bb_extent_x, veh.bb_extent_y, flag)){
        // return true;
        result = true;
    }
    test = car.pos + tan_dir * (ModelParams::CAR_WIDTH/2.0) - along_dir * ModelParams::CAR_FRONT;
    if(::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y, veh.heading_dir, veh.bb_extent_x, veh.bb_extent_y, flag)){
        // return true;
        result = true;
    }
    test = car.pos - tan_dir * (ModelParams::CAR_WIDTH/2.0) - along_dir * ModelParams::CAR_FRONT;
    if(::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y, veh.heading_dir, veh.bb_extent_x, veh.bb_extent_y, flag)){
        // return true;
        result = true;
    }

    // return false;

    if (flag == 1 && result){
        cout << "collision point";
        cout << " "<< veh.pos.x << " " << veh.pos.y;
        test = veh.pos + tan_dir * veh.bb_extent_x + along_dir * veh.bb_extent_y;
        cout << " "<< test.x << " " << test.y;
        test = veh.pos - tan_dir * veh.bb_extent_x + along_dir * veh.bb_extent_y;
        cout << " "<< test.x << " " << test.y;
        test = veh.pos + tan_dir * veh.bb_extent_x - along_dir * veh.bb_extent_y;
        cout << " "<< test.x << " " << test.y;
        test = veh.pos - tan_dir * veh.bb_extent_x - along_dir * veh.bb_extent_y;
        cout << " "<< test.x << " " << test.y;

        cout << " "<< car.pos.x << " " << car.pos.y;
        test = car.pos + tan_dir * (ModelParams::CAR_WIDTH/2.0) + along_dir * ModelParams::CAR_FRONT;
        cout << " "<< test.x << " " << test.y;
        test = car.pos - tan_dir * (ModelParams::CAR_WIDTH/2.0) + along_dir * ModelParams::CAR_FRONT;
        cout << " "<< test.x << " " << test.y;
        test = car.pos + tan_dir * (ModelParams::CAR_WIDTH/2.0) - along_dir * ModelParams::CAR_FRONT;
        cout << " "<< test.x << " " << test.y;
        test = car.pos - tan_dir * (ModelParams::CAR_WIDTH/2.0) - along_dir * ModelParams::CAR_FRONT;
        cout << " "<< test.x << " " << test.y << endl;
    }

    return result;


    // result = result || CheckCarWithObsLine(car, 
    //     veh.pos + tan_dir * veh.bb_extent_x, veh.pos + along_dir * veh.bb_extent_y, flag);
    // result = result || CheckCarWithObsLine(car, 
    //     veh.pos + tan_dir * veh.bb_extent_x, veh.pos - along_dir * veh.bb_extent_y, flag);
    // result = result || CheckCarWithObsLine(car, 
    //     veh.pos - tan_dir * veh.bb_extent_x, veh.pos - along_dir * veh.bb_extent_y, flag);
    // result = result || CheckCarWithObsLine(car, 
    //     veh.pos - tan_dir * veh.bb_extent_x, veh.pos + along_dir * veh.bb_extent_y, flag);
    // return result;
}

bool WorldModel::CheckCarWithObsLine(const CarStruct& car, COORD start_point, COORD end_point, int flag){
	const double step = ModelParams::OBS_LINE_STEP;

	double d = COORD::EuclideanDistance(start_point, end_point);
	double dx = (end_point.x-start_point.x) / d;
	double dy = (end_point.y-start_point.y) / d;
	double sx = start_point.x;
	double sy = start_point.y;

	double t=0, ti=0;

	while(t < ti+d) {

		double u = t - ti;
		double nx = sx + dx*u;
		double ny = sy + dy*u;

		if (flag == 0){
			if(::inCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir, false))
				return true;
		}
		else if (flag == 1){
			if(::inRealCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir, false))
				return true;
		}
		//if(::inCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir))
		//	return true;

		if (t == ti + d - 0.01)
			break;
		else
			t = min( t + step, ti + d - 0.01);
	}

	return false;
}


void WorldStateTracker::text(const vector<WorldStateTracker::AgentDistPair>& sorted_agents) const{
    
    if (logging::level()>=logging::VERBOSE){
        cout << "=> Sorted_agents:" << endl;

        for (auto& dist_agent_pair: sorted_agents){
            double dist = dist_agent_pair.first;
            auto& agent = *dist_agent_pair.second;

            fprintf(stderr, "==> id / type / pos / vel / reset: %d / %d / (%f %f) / (%f %f) / %d \n", 
                agent.id, agent.type(), agent.w, agent.h, agent.vel.x, agent.vel.y, agent.reset_intention);
        }
    }
}

void WorldStateTracker::text(const vector<Pedestrian>& tracked_peds) const{
    if (logging::level()>=logging::VERBOSE){
        cout << "=> ped_list:" << endl;
        for (auto& agent: tracked_peds) {
            fprintf(stderr, "==> id / type / pos / vel / cross / reset: %d / %d / (%f %f) / (%f %f) / %d / %d \n", 
                agent.id, agent.type(), agent.w, agent.h, agent.vel.x, agent.vel.y, agent.cross_dir, agent.reset_intention);   
        }
    }
}

void WorldStateTracker::text(const vector<Vehicle>& tracked_vehs) const{
    if (logging::level()>=logging::VERBOSE){
        cout << "=> veh_list:" << endl;
        for (auto& agent: tracked_vehs) {
            fprintf(stderr, "==> id / type / pos / vel / heading_dir / reset: %d / %d / (%f %f) / (%f %f) / %f / %d \n", 
                agent.id, agent.type(), agent.w, agent.h, agent.vel.x, agent.vel.y, agent.heading_dir, agent.reset_intention);   
        }
    }
}

void WorldStateTracker::check_world_status() {
	int alive_count = 0;
	for (auto& walker: ped_list){
		if (AgentIsUp2Date(walker)){
			alive_count += 1;
		}
	}

	for (auto& vehicle: veh_list){
		if (AgentIsUp2Date(vehicle)){
			alive_count += 1;
		}
	}

	if (alive_count == 0)
		DEBUG("No agent alive in the current scene.");
}
