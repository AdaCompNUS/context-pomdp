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
	logd << __FUNCTION__ << "] Get state info"<< endl;
	const PomdpState *state=static_cast<const PomdpState*>(_state);
	double mindist = numeric_limits<double>::infinity();
	const COORD& carpos = state->car.pos;
	double carvel = state->car.vel;

	logd << __FUNCTION__ <<"] Calculate steering"<< endl;

	double steering = GetSteerToPath(static_cast<const PomdpState*>(state)[0]);

	logd << __FUNCTION__ << "] Calculate acceleration"<< endl;

	double acceleration;
	// Closest pedestrian in front
	for (int i=0; i<state->num; i++) {
		const AgentStruct & p = state->agents[i];
		if(!inFront(p.pos, state->car)) continue;

		double d = Globals::POS_INFTY;

        if (p.type == AgentType::ped)
            d = COORD::EuclideanDistance(carpos, p.pos);
        else if (p.type == AgentType::car){
            d = COORD::EuclideanDistance(carpos, p.pos) - 2.0; 
        }
        
		if (d >= 0 && d < mindist)
			mindist = d;
	}

	// TODO set as a param
	logd << __FUNCTION__ <<"] Calculate min dist"<< endl;
	if (mindist < /*2*/3.5) {
		acceleration= (carvel <= 0.01) ? 0 : -ModelParams::AccSpeed;
	}
	else if (mindist < /*4*/5) {
		if (carvel > 1.0+1e-4) acceleration= -ModelParams::AccSpeed;
		else if (carvel < 0.5-1e-4) acceleration= ModelParams::AccSpeed;
		else acceleration= 0.0;
	}
	else
		acceleration= carvel >= ModelParams::VEL_MAX-1e-4 ? 0 : ModelParams::AccSpeed;

	logd << __FUNCTION__ <<"] Calculate action ID"<< endl;
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


bool WorldModel::inFront(const COORD ped_pos, const CarStruct car) const {
    if(ModelParams::IN_FRONT_ANGLE_DEG >= 180.0) {
        // inFront check is disabled
        return true;
    }
	double d0 = COORD::EuclideanDistance(car.pos, ped_pos);
	//if(d0<=0) return true;
	if(d0 <= /*0.7*/3.5) return true;
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
bool inCollision(double Px, double Py, double Cx, double Cy, double Ctheta);
bool inRealCollision(double Px, double Py, double Cx, double Cy, double Ctheta);

bool WorldModel::inCollision(const PomdpState& state) {
	const COORD& car_pos = state.car.pos;

    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;
        
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

bool WorldModel::inRealCollision(const PomdpStateWorld& state, int &id) {
    id=-1;
    const COORD& car_pos = state.car.pos;

    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;

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
            logd << "[WorldModel::inRealCollision] car_pos: ("<< car_pos.x <<","<< car_pos.y<<"), heading: ("
                << std::cos(state.car.heading_dir) <<","<< std::sin(state.car.heading_dir)<<"), agent_pos: ("
                << agent.pos.x <<","<< agent.pos.y<<")\n";
            return true;
        }
    }
    
    if(CheckCarWithObstacles(state.car,1))
    	return true;

    return false;
}

bool WorldModel::inRealCollision(const PomdpStateWorld& state) {
    const COORD& car_pos = state.car.pos;

    int id = -1;
    int i = 0;
    for(auto& agent: state.agents) {
        if (i >= state.num) 
            break;
        i++;
        
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
            logd << "[WorldModel::inRealCollision] car_pos: ("<< car_pos.x <<","<< car_pos.y<<"), heading: ("
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
    }

    if(CheckCarWithObstacles(state.car,0))
    	return true;

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
            logd << "[WorldModel::inRealCollision] car_pos: ("<< car_pos.x <<","<< car_pos.y<<"), heading: ("
                << std::cos(state.car.heading_dir) <<","<< std::sin(state.car.heading_dir)<<"), agent_pos: ("
                << agent.pos.x <<","<< agent.pos.y<<")\n";
            return true;
        }
    }

    if(CheckCarWithObstacles(state.car,0))
    	return true;

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

	if (goalpos.x == -1 && goalpos.y == -1)
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

    double r = CAR_LENGTH / tan(ModelParams::MaxSteerAngle);
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

    double r = CAR_LENGTH / tan(ModelParams::MaxSteerAngle);
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
    const COORD& goal = GetGoalPos(agent);
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return;
	}

	MyVector goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);
    double a = goal_vec.GetAngle();
	double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
    a += noise;

	//TODO noisy speed
    MyVector move(a, agent.speed/freq, 0);
    agent.pos.x += move.dw;
    agent.pos.y += move.dh;
    return;
}

void WorldModel::PedStep(AgentStruct &agent, double& random) {

    const COORD& goal = GetGoalPos(agent);
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return;
	}

	MyVector goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);

	if(goal_vec.GetLength() >= 0.5 ){

		double a = goal_vec.GetAngle();

		//double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
		double noise = sqrt(-2 * log(random));
		if(FIX_SCENARIO!=1 && !CPUDoPrint){
			random=QuickRandom::RandGeneration(random);
		}

		noise *= cos(2 * M_PI * random)* ModelParams::NOISE_GOAL_ANGLE;
		a += noise;

		//TODO noisy speed
//		cout << "agent vel = " << agent.vel << endl;
		MyVector move(a, agent.speed/freq, 0);
		agent.pos.x += move.dw;
		agent.pos.y += move.dh;
	}

    return;
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
        // not-stop intention not included in path list = agent cross intention
        if(type == AgentType::ped){
            int path_id = 0; // agent has only 1 path
            auto& path = path_candidates[path_id];
            COORD cross_dir= path.GetCrossDir(pos_along_path, agent_cross_dir); 
            return COORD(agent_pos.x + cross_dir.x/freq*3, agent_pos.y + cross_dir.y/freq*3);
        } else {
            cout << "Agent type " << type << " should not have cross intention" << endl;
            raise(SIGABRT);
        }
    }
    else if (intention_id == GetNumIntentions(agent_id)-1){ // stop intention
        return COORD(agent_pos.x, agent_pos.y);
    }
    else{
        cout << "Intention ID larger than # intentions for agent " << agent_id << endl;
        raise(SIGABRT);
    }
}

bool fetch_cross_dir(const Agent& agent) {
    if (agent.type()== AgentType::ped)
        return static_cast<const Pedestrian*>(&agent)->cross_dir;
    else
        return true;
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

    // return NumPaths(agent.id) + 1; // number of paths + stop (agent has one more cross intention)

    return 2; // current vel model
}


int WorldModel::GetNumIntentions(const Agent& agent){

    // return NumPaths(agent.id) + 1; // number of paths + stop (agent has one more cross intention)

    return 2; // current vel model
}

int WorldModel::GetNumIntentions(int agent_id){
    // return NumPaths(agent_id) + 1; // number of paths + stop (agent has one more cross intention)

    return 2; // current vel model
}

void WorldModel::PedStepGoal(AgentStruct& agent, int step) {
  const COORD& goal = goals[agent.intention];
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return;
	}

	MyVector goal_vec(goal.x - agent.pos.x, goal.y - agent.pos.y);
    goal_vec.AdjustLength(step * agent.speed / freq);
    agent.pos.x += goal_vec.dw;
    agent.pos.y += goal_vec.dh;
}

void WorldModel::PedStepCurVel(AgentStruct& agent) {
  agent.pos.x += agent.vel.x * (1.0/freq);
  agent.pos.y += agent.vel.y * (1.0/freq);
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
    
    agent.bb_extent_x = 0.0; agent.bb_extent_y = 0.0;

    for (auto& point: b.bb){
        agent.bb_extent_x = max((point - agent.pos).dot(sideward_vec), agent.bb_extent_x);
        agent.bb_extent_y = max((point - agent.pos).dot(forward_vec), agent.bb_extent_y);
    }
}

void WorldModel::PedStepPath(AgentStruct& agent) {
    auto& path_candidates = PathCandidates(agent.id);

    if (agent.intention < path_candidates.size()){
        auto& path = path_candidates[agent.intention];

        agent.pos_along_path = path.forward(agent.pos_along_path, agent.speed * (1.0/freq));
        COORD new_pos = path[agent.pos_along_path];
        agent.vel = (new_pos - agent.pos) * freq;
        agent.pos = new_pos;   
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
	if (goal.x == -1 && goal.y == -1) {  //stop intention 
		return (move_dist < sensor_noise) ? 0.4 : 0;
	} else {
		if (move_dist < sensor_noise) return 0;

		double cosa = DotProduct(curr.x-prev.x, curr.y-prev.y, goal.x-prev.x, goal.y-prev.y) / (move_dist * goal_dist);
		if(cosa >1) cosa = 1;
		else if(cosa < -1) cosa = -1;
		double angle = acos(cosa);
		return gaussian_prob(angle, ModelParams::NOISE_GOAL_ANGLE) + K;
	}
}

void WorldModel::EnsureMeanDirExist(int agent_id){

    auto it = ped_mean_dirs.find(agent_id);
    if ( it == ped_mean_dirs.end()){ 
        // if(agent_id >= ped_mean_dirs.size()){
        cout << "Encountering new agent id in EnsureMeanDirExist" << agent_id;
        // while (ped_mean_dirs.size() < agent_id + 1){
        cout << "adding init mean dir for agent " << agent_id << endl;
        vector<COORD> dirs(GetNumIntentions(agent_id));
        ped_mean_dirs[agent_id] = dirs;
        // }
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
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		logd <<"stop intention" << endl;

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
			logd <<"goal_dist=" << goal_dist << " move_dist " << move_dist << endl;
//		cout << __FUNCTION__ << "@" << __LINE__ << endl;

		double angle_prob = gaussian_prob(angle, ModelParams::NOISE_GOAL_ANGLE) + K;

		double vel_error=0;
		if(ped_mode == AGENT_ATT){
			double mean_vel = ped_mean_dirs[agent_id][intention_id].Length();
			vel_error = move_dist - mean_vel;

			logd <<"ATT angle error=" << angle << endl;
			logd <<"ATT vel_error=" << vel_error << endl;

		}
		else{
			vel_error = move_dist - ModelParams::PED_SPEED/freq;

			logd <<"DIS angle error=" << angle << endl;
			logd <<"DIS vel_error=" << vel_error << endl;

		}
//		cout << __FUNCTION__ << "@" << __LINE__ << endl;

		double vel_prob = gaussian_prob(vel_error, ModelParams::NOISE_PED_VEL/freq) + K;
		return angle_prob* vel_prob;
	}
}

void WorldModel::FixGPUVel(CarStruct &car)
{
	float tmp=car.vel/(ModelParams::AccSpeed/freq);
	car.vel=((int)(tmp+0.5))*(ModelParams::AccSpeed/freq);
}


void WorldModel::RobStep(CarStruct &car, double steering, Random& random) {
	if(steering!=0){
        assert(tan(steering)>0);
		double TurningRadius = CAR_LENGTH/tan(steering);
        assert(TurningRadius>0);
		double beta= car.vel/freq/TurningRadius;
		car.pos.x=car.pos.x+TurningRadius*(sin(car.heading_dir+beta)-sin(car.heading_dir));
		car.pos.y=car.pos.y+TurningRadius*(cos(car.heading_dir)-cos(car.heading_dir+beta));
		car.heading_dir=cap_angle(car.heading_dir+beta);
	}
	else{
		car.pos.x+=(car.vel/freq) * cos(car.heading_dir);
		car.pos.y+=(car.vel/freq) * sin(car.heading_dir);
	}
}


void WorldModel::RobStep(CarStruct &car, double steering, double& random) {
	if(steering!=0){
        //assert(tan(steering)>0);
		double TurningRadius = CAR_LENGTH/tan(steering);
        //assert(TurningRadius>0);
		double beta= car.vel/freq/TurningRadius;
		car.pos.x=car.pos.x+TurningRadius*(sin(car.heading_dir+beta)-sin(car.heading_dir));
		car.pos.y=car.pos.y+TurningRadius*(cos(car.heading_dir)-cos(car.heading_dir+beta));
		car.heading_dir=cap_angle(car.heading_dir+beta);
	}
	else{
		car.pos.x+=(car.vel/freq) * cos(car.heading_dir);
		car.pos.y+=(car.vel/freq) * sin(car.heading_dir);
	}
}


void WorldModel::RobStep(CarStruct &car, double& random, double acc, double steering) {
    double end_vel = car.vel + acc / freq;
    end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);

    if(steering!=0){
		assert(tan(steering)>0);
		double TurningRadius = CAR_LENGTH/tan(steering);
		assert(TurningRadius>0);
		double beta= end_vel/freq/TurningRadius;
		car.pos.x=car.pos.x+TurningRadius*(sin(car.heading_dir+beta)-sin(car.heading_dir));
		car.pos.y=car.pos.y+TurningRadius*(cos(car.heading_dir)-cos(car.heading_dir+beta));
		car.heading_dir=cap_angle(car.heading_dir+beta);
	}
	else{
		car.pos.x+=(end_vel/freq) * cos(car.heading_dir);
		car.pos.y+=(end_vel/freq) * sin(car.heading_dir);
	}
}

void WorldModel::RobStep(CarStruct &car, Random& random, double acc, double steering) {
    double end_vel = car.vel + acc / freq;
    end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);

    if(steering!=0){
		assert(tan(steering)>0);
		double TurningRadius = CAR_LENGTH/tan(steering);
		assert(TurningRadius>0);
		double beta= end_vel/freq/TurningRadius;
		car.pos.x=car.pos.x+TurningRadius*(sin(car.heading_dir+beta)-sin(car.heading_dir));
		car.pos.y=car.pos.y+TurningRadius*(cos(car.heading_dir)-cos(car.heading_dir+beta));
		car.heading_dir=cap_angle(car.heading_dir+beta);
	}
	else{
		car.pos.x+=(end_vel/freq) * cos(car.heading_dir);
		car.pos.y+=(end_vel/freq) * sin(car.heading_dir);
	}
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

    if(!curr_agent.reset_intention){

        b.goal_paths = PathCandidates(b.id);

        if (curr_agent.type()==AgentType::car){
            b.bb = static_cast<const Vehicle*>(&curr_agent)->bb;
            b.heading_dir = static_cast<const Vehicle*>(&curr_agent)->heading_dir;
        } else if (curr_agent.type() == AgentType::ped){
            b.cross_dir = static_cast<const Pedestrian*>(&curr_agent)->cross_dir;    
        }

        for(int i=0; i<GetNumIntentions(curr_agent); i++) {

            // Attentive mode
            logd << "TEST ATT" << endl;
            double prob = agentMoveProb(b.pos, curr_agent, i, AGENT_ATT);
            if(debug) cout << "attentive likelihood " << i << ": " << prob << endl;
            b.prob_modes_goals[AGENT_ATT][i] *=  prob;
            // Keep the belief noisy to avoid aggressive policies
            b.prob_modes_goals[AGENT_ATT][i] += SMOOTHING / GetNumIntentions(curr_agent)/2.0; // CHECK: decrease or increase noise

            // Detracted mode
            logd << "TEST DIS" << endl;
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
    b.speed = ALPHA * b.speed + (1-ALPHA) * moved_dist * ModelParams::control_freq;
	b.pos = cur_pos;
}

AgentBelief WorldModel::initPedBelief(const Agent& agent) {
    AgentBelief b;
    b.id = agent.id;
    b.pos = COORD(agent.w, agent.h);
    b.speed = ModelParams::PED_SPEED;
    b.vel = agent.vel;
    // b.speed = ModelParams::PED_SPEED;
//    cout << "AGENT_DIS + 1= "<< AGENT_DIS + 1 << endl;
    int num_types = AGENT_DIS + 1;
    for(int i =0 ; i < num_types; i++){
    	b.prob_modes_goals.push_back(vector<double>(GetNumIntentions(agent), 1.0/GetNumIntentions(agent)/num_types));
    }

    return b;
}


void WorldStateTracker::cleanAgents(){
    cleanPed();
    cleanVeh();
}

void WorldStateTracker::cleanPed() {
    vector<Pedestrian> ped_list_new;
    for(int i=0;i<ped_list.size();i++)
    {
        bool insert = AgentIsAlive(ped_list[i], ped_list_new);
        if (insert)
            ped_list_new.push_back(ped_list[i]);
        else{
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
            model.id_map_num_paths.erase(ped_list[i].id);
            model.id_map_paths.erase(ped_list[i].id);
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

void WorldStateTracker::tracIntention(Agent& des, const Agent& src, bool doPrint){
    des.reset_intention = src.reset_intention;
    des.paths = src.paths;
    model.id_map_paths[des.id] = des.paths; 

    switch (des.type()){
        case AgentType::ped:
            model.id_map_num_paths[des.id] = des.paths.size() + 1; break;
        case AgentType::car:
            model.id_map_num_paths[des.id] = des.paths.size(); break;
        default:
            cout << __FUNCTION__ <<": unsupported agent type " << des.type() << endl;
            raise(SIGABRT); break;
    }
}

void WorldStateTracker::trackVel(Agent& des, const Agent& src, bool& no_move, bool doPrint){
    double duration =  get_timestamp() - des.last_update;

    if (duration < 0.1 / Globals::config.time_scale){
        no_move = false;
        return;
    }

    des.vel.x = (src.w - des.w) / duration;
    des.vel.y = (src.h - des.h) / duration;

    if (Globals::config.use_prior) 
        if (des.vel.Length()>3.0){
            ERR(string_sprintf("WARNING: Unusual src for ped %d, speed: %f\n", src.id, des.vel.Length()));
        }

   if (des.vel.Length()>1e-4){
        no_move = false;
    }

}


void WorldStateTracker::updateVeh(const Vehicle& veh, bool doPrint){
    // double w,h;
    // COORD vel;
    // int id;  
    // double last_update;

    // bool reset_intention;
    // std::vector<std::vector<COORD>> paths;  

    // std::vector<COORD> bb;
    
    int i=0;
    bool no_move = true;
    for(;i<veh_list.size();i++) {
        if (veh_list[i].id==veh.id) {
            trackVel(veh_list[i], veh, no_move, doPrint);
            tracPos(veh_list[i], veh, doPrint);
            tracIntention(veh_list[i], veh, doPrint);
            tracBoundingBox(veh_list[i], veh, doPrint);

            veh_list[i].last_update = get_timestamp();

            break;
        }

        if (abs(veh_list[i].w-veh.w)<=0.1 && abs(veh_list[i].h-veh.h)<=0.1)   //overlap
        {}        
    }

    if (i==veh_list.size()) {
        no_move = false;

        veh_list.push_back(veh);

        veh_list.back().vel.x = 0;
        veh_list.back().vel.y = 0;
        veh_list.back().last_update = get_timestamp();
    }

    if (no_move){
        cout << __FUNCTION__ << " no_move veh "<< veh.id <<
                " caught: vel " << veh_list[i].vel.x <<" "<< veh_list[i].vel.y << endl;
    }
}

void WorldStateTracker::updatePed(const Pedestrian& agent, bool doPrint){
    int i=0;

    bool no_move = true;
    for(;i<ped_list.size();i++) {
        if (ped_list[i].id==agent.id) {

            trackVel(ped_list[i], agent, no_move, doPrint);
            tracPos(ped_list[i], agent, doPrint);
            tracIntention(ped_list[i], agent, doPrint);
            tracCrossDirs(ped_list[i], agent, doPrint);
            ped_list[i].last_update = get_timestamp();

            break;
        }
        if (abs(ped_list[i].w-agent.w)<=0.1 && abs(ped_list[i].h-agent.h)<=0.1)   //overlap
        {
        	//if (doPrint) cout <<"[updatePed] overlapping agent skipped" << agent.id << endl;
            //return;
        }
    }

    if (i==ped_list.size()) {
    	no_move = false;
        //not found, new agent
//    	cout <<"[updatePed] updating new agent" << agent.id << endl;

        ped_list.push_back(agent);

        ped_list.back().vel.x = 0;
        ped_list.back().vel.y = 0;
        ped_list.back().last_update = get_timestamp();
//        cout <<"[updatePed] new agent added" << agent.id << endl;
    }

    if (no_move){
		cout << __FUNCTION__ << " no_move agent "<< agent.id <<
				" caught: vel " << ped_list[i].vel.x <<" "<< ped_list[i].vel.y << endl;
//		raise(SIGABRT);
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
        if (proj < 0.7)
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
        if (proj < 0.7)
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
    agent.cross_dir = -1; // which path to take
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

    auto sorted_agents = getSortedAgents();

    PomdpStateWorld worldState;

    setPomdpCar(worldState.car);
    
    worldState.num = min((int)sorted_agents.size(), ModelParams::N_PED_WORLD); //current_state.num = num_of_peds_world;

    for(int i=0;i<worldState.num;i++) {
      const auto& agent = *(sorted_agents[i].second);
      setPomdpPed(worldState.agents[i], agent);
    }
    
    return worldState;
}

void AgentBelief::reset_belief(){
    reset = true;
    double accum_prob = 0;

    for (auto& prob_goals: prob_modes_goals){
        std::fill(prob_goals.begin(), prob_goals.end(), 1.0);

        accum_prob += accumulate(prob_goals.begin(),prob_goals.end(),0);
    }

    // normalize distribution
    for (auto& prob_goals: prob_modes_goals){
        for (auto& prob : prob_goals)
            prob = prob / accum_prob;
    }
}

void WorldBeliefTracker::update() {

	// Update agent_beliefs
    DEBUG("sorted_agents");
    auto sorted_agents = stateTracker.getSortedAgents();
    map<int, const Agent*> newagents;
    for(WorldStateTracker::AgentDistPair& dp: sorted_agents) {
        auto p = dp.second;
        // AgentStruct agent(COORD(p.w, p.h), -1, p.id);
        // agent.vel = p.vel;
        // agent.speed = ModelParams::PED_SPEED;
        // agent.type = p.type();
        // agent.cross_dir = p.cross_dir;
        newagents[p->id] = p;
    }

    // remove disappeared agent_beliefs
    DEBUG("remove");
    vector<int> agents_to_remove;

    for(const auto& p: agent_beliefs) {
        if (newagents.find(p.first) == newagents.end()) {
            agents_to_remove.push_back(p.first);
            logi << "["<<__FUNCTION__<< "]" << " removing agent "<< p.first << endl;
        }
    }

    DEBUG("reset");

    for(auto& dp: sorted_agents) {
        auto& agent = *dp.second;
        if (agent.reset_intention) {
            agent_beliefs[agent.id].reset_belief();
            logi << "["<<__FUNCTION__<< "]" << " belief reset: agent "<< agent.id << endl;
        }
        else{
            agent_beliefs[agent.id].reset = false;
        }
    }    

    for(const auto& i: agents_to_remove) {
        agent_beliefs.erase(i);
    }

    // Run ORCA for all possible hidden variable combinations

    DEBUG("meandir");
    model.PrepareAttentiveAgentMeanDirs(agent_beliefs, car);

    model.PrintMeanDirs(agent_beliefs, newagents);

    // update car
    car.pos = stateTracker.carpos;
    car.vel = stateTracker.carvel;
	car.heading_dir = /*0*/stateTracker.car_heading_dir;


    DEBUG("update ped belief");
    // update existing agent_beliefs
    for(auto& kv : agent_beliefs) {

    	if (newagents.find(kv.first) == newagents.end()){
            ERR(string_sprintf("updating non existing id %d in new agent list", kv.first))
    	}
        model.updatePedBelief(kv.second, *newagents[kv.first]);
    }

    DEBUG("new ped");
    // add new agent_beliefs
    for(const auto& kv: newagents) {
		auto& p = *kv.second;
        if (agent_beliefs.find(p.id) == agent_beliefs.end()) {
            agent_beliefs[p.id] = model.initPedBelief(p);
        }
    }

    DEBUG("sorted");

	sorted_beliefs.clear();
	for(const auto& dp: sorted_agents) {
		auto& p = *dp.second;
		sorted_beliefs.push_back(&agent_beliefs[p.id]);
	}

	cur_time_stamp = SolverPrior::get_timestamp();

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

void AgentBelief::sample_goal_mode(int& goal, int& mode) const {
//	logd << "[AgentBelief::sample_goal_mode] " << endl;

    double r = Random::RANDOM.NextDouble();

    bool done = false;
    for (int ped_type = 0 ; ped_type < prob_modes_goals.size() ; ped_type++){
    	auto& goal_probs = prob_modes_goals[ped_type];
    	for (int ped_goal = 0 ; ped_goal < goal_probs.size() ; ped_goal++){
    	    r -= prob_modes_goals[ped_type][ped_goal];
    		if(r <= 0) {
    		    goal = ped_goal;
    		    mode = ped_type;
    		    done = true;
    		    break;
    		}
    		else{
			}
    	}
    	if (done)
    		break;
    }

    if(r > 0) {
    	logd << "[WARNING]: [AgentBelief::sample_goal_mode] execess probability " << r << endl;
    	goal = 0;
    	mode = 0;
    }
    else{
//    	logd << "[AgentBelief::sample_goal_mode] sampled values " << goal << " " << mode << endl;
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
	if (logging::level()>=4){
		for(int i=0; i < sorted_beliefs.size() && i < min(6,ModelParams::N_PED_IN); i++) {
			auto& p = *sorted_beliefs[i];
			cout << "[WorldBeliefTracker::text] " << this << "->p:" << &p << endl;
			cout << " sorted agent " << i << endl;

			cout << "-prob_modes_goals: " << p.prob_modes_goals << endl;
			for (int i=0;i<p.prob_modes_goals.size();i++){
				for (int j=0;j<p.prob_modes_goals[i].size();j++){
					cout << p.prob_modes_goals[i][j] << " ";
				}
				cout<< endl;
			}
		}
	}
}
PomdpState WorldBeliefTracker::sample(bool predict) {
    PomdpState s;
    s.car = car;

	s.num = 0;
    for(int i=0; i < sorted_beliefs.size() && i < ModelParams::N_PED_IN; i++) {
		auto& p = *sorted_beliefs[i];

		if (COORD::EuclideanDistance(p.pos, car.pos) < ModelParams::LASER_RANGE) {
			s.agents[s.num].pos = p.pos;
//			s.agents[s.num].goal = p.sample_goal();
			p.sample_goal_mode(s.agents[s.num].intention, s.agents[s.num].mode);
			s.agents[s.num].id = p.id;
            s.agents[s.num].vel = p.vel;
            s.agents[s.num].speed = p.speed;
            s.agents[s.num].pos_along_path = 0; // assuming that paths are up to date here
            s.agents[s.num].cross_dir = p.cross_dir;
            s.agents[s.num].type = p.type;
            model.cal_bb_extents(p, s.agents[s.num]);
			s.num ++;
		}
    }

    s.time_stamp  = cur_time_stamp;

//	logd << "[WorldBeliefTracker::sample] done" << endl;
    if (predict){
    	PomdpState predicted_s = predictPedsCurVel(&s, cur_acc, cur_steering);
    	return predicted_s;
    }

    return s;
}

vector<PomdpState> WorldBeliefTracker::sample(int num, bool predict) {

	if(DESPOT::Debug_mode)
		Random::RANDOM.seed(0);

    vector<PomdpState> particles;
	logd << "[WorldBeliefTracker::sample] Sampling" << endl;

    for(int i=0; i<num; i++) {
        particles.push_back(sample(predict));
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
                model.PedStepGoal(agent, step + i);
            }
            else if (model.goal_mode == "cur_vel"){
                model.PedStepCurVel(agent);
            }
            else if (model.goal_mode == "path"){
                model.PedStepPath(agent);
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


void WorldModel::RVO2AgentStep(PomdpStateWorld& state, Random& random){

	AgentStruct* agents = state.agents;
	CarStruct car = state.car;
	int num_agents = state.num;
    int threadID=GetThreadID();

    // Construct a new set of agents every time
	traffic_agent_sim_[threadID]->clearAllAgents();

	//adding pedestrians
	int num_att_pes = 0;
	for(int i=0; i<num_agents; i++){
		if(state.agents[i].mode==AGENT_ATT){
			traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(agents[i].pos.x, agents[i].pos.y));
			num_att_pes++;
		}
	}

	// adding car as a "special" pedestrian
	add_car_agent(num_agents, car);

	// Set the preferred velocity for each agent.
	int agent = 0;
	for (size_t i = 0; i < num_agents; ++i) {
		if(state.agents[i].mode==AGENT_ATT){
		   int intention_id = agents[i].intention;
			if (intention_id >= GetNumIntentions(agents[i])-1) { /// stop intention
				traffic_agent_sim_[threadID]->setAgentPrefVelocity(agent, RVO::Vector2(0.0f, 0.0f));
			} else{

                auto goal_pos = GetGoalPos(agents[i], intention_id);
				RVO::Vector2 goal(goal_pos.x, goal_pos.y);
				if ( absSq(goal - traffic_agent_sim_[threadID]->getAgentPosition(agent)) < traffic_agent_sim_[threadID]->getAgentRadius(agent) * traffic_agent_sim_[threadID]->getAgentRadius(agent) ) {
					// Agent is within one radius of its goal, set preferred velocity to zero
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(agent, RVO::Vector2(0.0f, 0.0f));
				} else {
					// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(agent, normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(agent))*ModelParams::PED_SPEED);
				}
			}
			agent++;
		}
	}

	// Update positions of only all agents in the RVO simulator
	traffic_agent_sim_[threadID]->doStep();
    agent = 0;

    // Update positions of only attentive agents
    for(int i=0; i<num_agents; i++){
    	if(state.agents[i].mode==AGENT_ATT){
    		agents[i].pos.x=traffic_agent_sim_[threadID]->getAgentPosition(agent).x();// + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(agent).x() - agents[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
    		agents[i].pos.y=traffic_agent_sim_[threadID]->getAgentPosition(agent).y();// + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(agent).y() - agents[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

    		if(use_noise_in_rvo){
    			agents[i].pos.x+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
    			agents[i].pos.y+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
    		}
    		agent++;
    	}
    }
}

void WorldModel::RVO2AgentStep(AgentStruct agents[], Random& random, int num_agents, CarStruct car){
    int threadID=GetThreadID();
    traffic_agent_sim_[threadID]->clearAllAgents();

    //adding pedestrians
    for(int i=0; i<num_agents; i++){
        traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(agents[i].pos.x, agents[i].pos.y));
    }

    // adding car as a "special" pedestrian
    add_car_agent(num_agents, car);

    // Set the preferred velocity for each agent.
    for (size_t i = 0; i < num_agents; ++i) {
        int intention_id = agents[i].intention;
        if (intention_id >= GetNumIntentions(agents[i])-1) { /// stop intention
            traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
        } else{
            auto goal_pos = GetGoalPos(agents[i], intention_id);
            RVO::Vector2 goal(goal_pos.x, goal_pos.y);
            if ( absSq(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) < traffic_agent_sim_[threadID]->getAgentRadius(i) * traffic_agent_sim_[threadID]->getAgentRadius(i) ) {
                // Agent is within one radius of its goal, set preferred velocity to zero
                traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
            } else {
                // Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
                traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(i))*ModelParams::PED_SPEED);
            }
        }
        
    }

    traffic_agent_sim_[threadID]->doStep();

    for(int i=0; i<num_agents; i++){
        agents[i].pos.x=traffic_agent_sim_[threadID]->getAgentPosition(i).x();// + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(i).x() - agents[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        agents[i].pos.y=traffic_agent_sim_[threadID]->getAgentPosition(i).y();// + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(i).y() - agents[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        if(use_noise_in_rvo){
			agents[i].pos.x+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
			agents[i].pos.y+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
		}
    }

}


void WorldModel::RVO2AgentStep(AgentStruct agents[], double& random, int num_agents, CarStruct car){
    int threadID=GetThreadID();
    traffic_agent_sim_[threadID]->clearAllAgents();

    //adding pedestrians
    for(int i=0; i<num_agents; i++){
        traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(agents[i].pos.x, agents[i].pos.y));
    }

    // adding car as a "special" pedestrian
	add_car_agent(num_agents, car);

    // Set the preferred velocity for each agent.
    for (size_t i = 0; i < num_agents; ++i) {

    	if(agents[i].mode==AGENT_ATT){
			int intention_id = agents[i].intention;
			if (intention_id >= GetNumIntentions(agents[i])-1) { /// stop intention
				traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
			} else{
                auto goal_pos = GetGoalPos(agents[i], intention_id);
                RVO::Vector2 goal(goal_pos.x, goal_pos.y);
				if ( absSq(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) < traffic_agent_sim_[threadID]->getAgentRadius(i) * traffic_agent_sim_[threadID]->getAgentRadius(i) ) {
					// Agent is within one radius of its goal, set preferred velocity to zero
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
				} else {
					// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(i))*ModelParams::PED_SPEED);
				}
			}
    	}
    }

    traffic_agent_sim_[threadID]->doStep();

    for(int i=0; i<num_agents; i++){

    	if(agents[i].mode==AGENT_ATT){
			agents[i].pos.x=traffic_agent_sim_[threadID]->getAgentPosition(i).x();// + rNum * (traffic_agent_sim_[threadID]->getAgentPosition(i).x() - agents[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
			agents[i].pos.y=traffic_agent_sim_[threadID]->getAgentPosition(i).y();// + rNum * (traffic_agent_sim_[threadID]->getAgentPosition(i).y() - agents[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

			if(use_noise_in_rvo){
				double rNum=GenerateGaussian(random);
				agents[i].pos.x+= rNum * ModelParams::NOISE_PED_POS / freq;
				rNum=GenerateGaussian(rNum);
				agents[i].pos.y+= rNum * ModelParams::NOISE_PED_POS / freq;
			}
    	}
    }
}

COORD WorldModel::DistractedPedMeanDir(AgentStruct& agent, int intention_id) {
	COORD dir(0,0);
	const COORD& goal = GetGoalPos(agent, intention_id);
	if (goal.x == -1 && goal.y == -1) {  //stop intention
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
		car_x = car.pos.x - CAR_LENGTH/2.0f*cos(car_yaw);
		car_y = car.pos.y - CAR_LENGTH/2.0f*sin(car_yaw);
		double car_radius = sqrt(pow(CAR_WIDTH/2.0f, 2) + pow(CAR_LENGTH/2.0f,2)) + CAR_EXPAND_SIZE;
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
}

double get_heading_dir(COORD vel){

}

void WorldModel::add_veh_agent(int id_in_sim, AgentBelief& veh){

    if(veh.type != AgentType::car)
        return;

    int threadID=GetThreadID();

    double veh_x, veh_y, veh_yaw;

    veh_yaw = veh.heading_dir;

    auto& veh_bb = veh.bb;

    // TODO: replace following to use the bonding box data.

    double car_radius = 1.15f;
    double car_radius_large = 1.6f;

    DEBUG("agent circle");
    traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(veh.pos.x, veh.pos.y), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    DEBUG("agent pref");
    traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim, 
        RVO::Vector2(veh.vel.Length() * cos(veh_yaw), veh.vel.Length() * sin(veh_yaw))); // the id_in_sim-th pedestrian is the veh. set its prefered velocity
    DEBUG("agent id");
    traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim,-1);

    DEBUG("next");
    traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(veh.pos.x + 0.56 * 2.33 * cos(veh_yaw), veh.pos.y + 1.4* sin(veh_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+1, 
        RVO::Vector2(veh.vel.Length() * cos(veh_yaw), veh.vel.Length() * sin(veh_yaw))); // the id_in_sim-th pedestrian is the veh. set its prefered velocity
    traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+1,-2);

    traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(veh.pos.x + 0.56*3.66 * cos(veh_yaw), veh.pos.y + 2.8* sin(veh_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+2, 
        RVO::Vector2(veh.vel.Length() * cos(veh_yaw), veh.vel.Length() * sin(veh_yaw))); // the id_in_sim-th pedestrian is the veh. set its prefered velocity
    traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+2,-3);

    traffic_agent_sim_[threadID]->addAgent(RVO::Vector2(veh.pos.x + 0.56*5 * cos(veh_yaw), veh.pos.y + 2.8* sin(veh_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius_large, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim+3, 
        RVO::Vector2(veh.vel.Length() * cos(veh_yaw), veh.vel.Length() * sin(veh_yaw))); // the id_in_sim-th pedestrian is the veh. set its prefered velocity
    traffic_agent_sim_[threadID]->setAgentPedID(id_in_sim+3,-4);

}

void WorldModel::PrepareAttentiveAgentMeanDirs(std::map<int, AgentBelief> agents, CarStruct& car){
	int num_agents = agents.size();

	logd << "num_agents in belief tracker: " << num_agents << endl;

    int threadID=GetThreadID();

    // Construct a new set of agents every time
	traffic_agent_sim_[threadID]->clearAllAgents();

    DEBUG("add agents");
	//adding pedestrians
    std::vector<int> agent_ids;
    agent_ids.resize(num_agents);
	for(int i=0; i<num_agents; i++){

        switch (agents[i].type) {
            case AgentType::ped:
                DEBUG("add ped");
        		traffic_agent_sim_[threadID]->addAgent(
                    RVO::Vector2(agents[i].pos.x, agents[i].pos.y)); break;
            case AgentType::car:
                DEBUG("add car");
                add_veh_agent(num_agents, agents[i]); break;
            default: 
                raise(SIGABRT); break;
        }
        // TODO: need to assign bounding boxes to vehicles
        agent_ids[i] = agents[i].id;
	}

    DEBUG("add_car");
	// adding car as a "special" pedestrian
	add_car_agent(num_agents, car);

	// Set the preferred velocity for each agent.

    DEBUG("prestep");
	traffic_agent_sim_[threadID]->doPreStep();// build kd tree and find neighbors for agents

    DEBUG("exist");
    for (int i = 0; i < num_agents ; i++)
        EnsureMeanDirExist(agent_ids[i]);

    DEBUG("pref");
	for (size_t i = 0; i < num_agents; ++i) {
		// For each ego agent
	    for(int intention_id=0; intention_id < GetNumIntentions(agent_ids[i]); intention_id++) {

	    	RVO::Vector2 ori_pos(traffic_agent_sim_[threadID]->getAgentPosition(i).x(),  
                traffic_agent_sim_[threadID]->getAgentPosition(i).y());

			// Set preferred velocity for the ego agent according to intention_id
			// Leave other pedestrians to have default preferred velocity

			if (intention_id >= GetNumIntentions(agent_ids[i])-1) { /// stop intention
				traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
			} else{
                auto goal_pos = GetGoalPos(agents[i], intention_id);
				RVO::Vector2 goal(goal_pos.x, goal_pos.y);
				if ( absSq(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) < traffic_agent_sim_[threadID]->getAgentRadius(i) * traffic_agent_sim_[threadID]->getAgentRadius(i) ) {
					// Agent is within one radius of its goal, set preferred velocity to zero
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
				} else {
					// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(i, normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(i))*ModelParams::PED_SPEED);
				}
			}

			traffic_agent_sim_[threadID]->doStepForPed(i); //TODO: this should be replace by GAMMA functions

			COORD dir;
			dir.x = traffic_agent_sim_[threadID]->getAgentPosition(i).x() - agents[i].pos.x;
			dir.y = traffic_agent_sim_[threadID]->getAgentPosition(i).y() - agents[i].pos.y;

			logd << "[PrepareAttentiveAgentMeanDirs] ped_mean_dirs len=" << ped_mean_dirs.size()
					<< " intention_list len=" << ped_mean_dirs[i].size() << "\n";

			logd << "[PrepareAttentiveAgentMeanDirs] i=" << i << " intention_id=" << intention_id << "\n";

			ped_mean_dirs[i][intention_id]=dir;

	    	// reset agent state
			traffic_agent_sim_[threadID]->setAgentPosition(i, ori_pos);
	    }
	}
}

void WorldModel::PrintMeanDirs(std::map<int, AgentBelief> old_agents, 
    map<int, const Agent*>& curr_agents){
	if(logging::level()>=3){
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
    	    	logd <<"agent, goal, ped_mean_dirs.size() =" << cur_agent.id << " "
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
    bool result = false;
    result = result | CheckCarWithObsLine(car, 
        veh.pos + COORD(veh.bb_extent_x, veh.bb_extent_y), veh.pos + COORD(-veh.bb_extent_x, veh.bb_extent_y), flag);
    result = result | CheckCarWithObsLine(car, 
        veh.pos + COORD(-veh.bb_extent_x, veh.bb_extent_y), veh.pos + COORD(-veh.bb_extent_x, -veh.bb_extent_y), flag);
    result = result | CheckCarWithObsLine(car, 
        veh.pos + COORD(-veh.bb_extent_x, -veh.bb_extent_y), veh.pos + COORD(veh.bb_extent_x, -veh.bb_extent_y), flag);
    result = result | CheckCarWithObsLine(car, 
        veh.pos + COORD(veh.bb_extent_x, -veh.bb_extent_y), veh.pos + COORD(veh.bb_extent_x, veh.bb_extent_y), flag);
    return result;
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

		if (flag ==0){
			if(::inCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir))
				return true;
		}
		else{
			if(::inRealCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir))
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
/*
bool WorldModel::pedMoveMeanDir(const PomdpStateWorld& state){

    logd << "[WorldModel::"<<__FUNCTION__<<"] Calculate mean dir for "<< state.num<< " agents and "<<goals.size()<<" goals"<< endl;

	AttentivePedMeanStep(state.agents, state.num, state.car);


	for(int agent_id=0;agent_id< state.num; agent_id++)
		for(int intention_id=0;intention_id< goals.size(); intention_id++){
			AgentStruct updated_ped=state.agents[agent_id];
			// assume that the pedestrian takes the goal
			updated_ped.goal=intention_id;
			PedMotionDirDeterministic(updated_ped);
			// record the movement direction of the pedestrian taking the goal
			ped_mean_dirs[agent_id][intention_id]=updated_ped.pos-state.agents[agent_id].pos;
			ped_mean_speeds[agent_id][intention_id]=COORD::EuclideanDistance(updated_ped.pos,
					state.agents[agent_id].pos);
		}

    logd << "[WorldModel::"<<__FUNCTION__<<"] Mean dirs :"<<endl;

	return true;
}
*/




