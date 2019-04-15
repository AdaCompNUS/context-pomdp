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

     for(int i=0;i<ModelParams::N_PED_WORLD; i++){
    	 vector<COORD> dirs(goals.size());
    	 ped_mean_dirs.push_back(dirs);
     }
}

void WorldModel::InitRVO(){
    if(!Globals::config.use_multi_thread_ )
        //NumThreads=Globals::config.NUM_THREADS;
        Globals::config.NUM_THREADS=1;

    int NumThreads=Globals::config.NUM_THREADS;

    ped_sim_.resize(NumThreads);
    for(int tid=0; tid<NumThreads;tid++)
    {
        ped_sim_[tid] = new RVO::RVOSimulator();
        
        // Specify global time step of the simulation.
        ped_sim_[tid]->setTimeStep(1.0f/ModelParams::control_freq);

        ped_sim_[tid]->setAgentDefaults(5.0f, 5, 1.5f, 1.5f, PED_SIZE, 2.5f);
    }
    
}

WorldModel::~WorldModel(){
    ped_sim_.clear();
    ped_sim_.resize(0);
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
//		auto& p = state->peds[i];
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
	auto& car_goal=this->car_goal;

	logd << __FUNCTION__ << "] "<<car_goal.x<<car_goal.y<<endl;

	logd << __FUNCTION__ <<"] Calculate steering"<< endl;
	double steering=CalculateGoalDir(state->car,car_goal);
	steering=steering*ModelParams::MaxSteerAngle;

	logd << __FUNCTION__ << "] Calculate acceleration"<< endl;
	double acceleration;
	// Closest pedestrian in front
	for (int i=0; i<state->num; i++) {
		const PedStruct & p = state->peds[i];
		if(!inFront(p.pos, state->car)) continue;
		double d = COORD::EuclideanDistance(carpos, p.pos);
		if (d >= 0 && d < mindist)
			mindist = d;
	}

	// TODO set as a param
	logd << __FUNCTION__ <<"] Calculate min dist"<< endl;
	if (mindist < /*2*/3.5) {
		acceleration= (carvel <= 0.01) ? 0 : 2;
	}
	else if (mindist < /*4*/5) {
		if (carvel > 1.0+1e-4) acceleration= 2;
		else if (carvel < 0.5-1e-4) acceleration= 1;
		else acceleration= 0;
	}
	else
		acceleration= carvel >= ModelParams::VEL_MAX-1e-4 ? 0 : 1;

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

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
            return true;
        }
    }

    if(CheckCarWithObstacles(state.car,0))
    	return true;

    return false;
}

bool WorldModel::inRealCollision(const PomdpStateWorld& state, int &id) {
    id=-1;
    const COORD& car_pos = state.car.pos;

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inRealCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
            id=state.peds[i].id;
            logd << "[WorldModel::inRealCollision] car_pos: ("<< car_pos.x <<","<< car_pos.y<<"), heading: ("
				<< std::cos(state.car.heading_dir) <<","<< std::sin(state.car.heading_dir)<<"), ped_pos: ("
				<< pedpos.x <<","<< pedpos.y<<")\n";
            return true;
        }
    }

    if(CheckCarWithObstacles(state.car,1))
    	return true;

    return false;
}

bool WorldModel::inRealCollision(const PomdpStateWorld& state) {
    const COORD& car_pos = state.car.pos;

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inRealCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
            return true;
        }
    }

    if(CheckCarWithObstacles(state.car,1))
    	return true;

    return false;
}

bool WorldModel::inCollision(const PomdpStateWorld& state) {
	const COORD& car_pos = state.car.pos;

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
            return true;
        }
    }

    if(CheckCarWithObstacles(state.car,0))
    	return true;

    return false;
}

bool WorldModel::inCollision(const PomdpState& state, int &id) {
	id=-1;
	const COORD& car_pos = state.car.pos;

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
        	id=state.peds[i].id;
            return true;
        }
    }

    if(CheckCarWithObstacles(state.car,0))
    	return true;

    return false;
}

bool WorldModel::inCollision(const PomdpStateWorld& state, int &id) {
    id=-1;
	const COORD& car_pos = state.car.pos;

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, state.car.heading_dir)) {
            id=state.peds[i].id;
            logd << "[WorldModel::inCollision] car_pos: ("<< car_pos.x <<","<< car_pos.y<<"), heading: ("
    				<< std::cos(state.car.heading_dir) <<","<< std::sin(state.car.heading_dir)<<"), ped_pos: ("
            		<< pedpos.x <<","<< pedpos.y<<")\n";
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
		const PedStruct& p = state.peds[i];
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


bool WorldModel::isMovingAway(const PomdpState& state, int ped) {
    const auto& carpos = state.car.pos;

	const auto& pedpos = state.peds[ped].pos;
	const auto& goalpos = goals[state.peds[ped].goal];

	if (goalpos.x == -1 && goalpos.y == -1)
		return false;

	return DotProduct(goalpos.x - pedpos.x, goalpos.y - pedpos.y,
			cos(state.car.heading_dir), sin(state.car.heading_dir)) > 0;
}

///get the min distance between car and the peds in its front
double WorldModel::getMinCarPedDist(const PomdpState& state) {
    double mindist = numeric_limits<double>::infinity();
    const auto& carpos = state.car.pos;

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.peds[i];
		if(!inFront(p.pos, state.car)) continue;
        double d = COORD::EuclideanDistance(carpos, p.pos);
        if (d >= 0 && d < mindist) mindist = d;
    }

	return mindist;
}

///get the min distance between car and the peds
double WorldModel::getMinCarPedDistAllDirs(const PomdpState& state) {
    double mindist = numeric_limits<double>::infinity();
    const auto& carpos = state.car.pos;

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.peds[i];
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

    if (chord_len >= d + ModelParams::GOAL_TOLERANCE){
    	printf("No steering path available: chord_len=%f, d=%f, r=%f, theta=%f\n", chord_len, d, r, theta);
    	return Globals::config.sim_len;  // can never reach goal with in the round
    }
    else if (chord_len <  d + ModelParams::GOAL_TOLERANCE && chord_len >= d) {
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

void WorldModel::PedStep(PedStruct &ped, Random& random) {
    const COORD& goal = goals[ped.goal];
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return;
	}

	MyVector goal_vec(goal.x - ped.pos.x, goal.y - ped.pos.y);
    double a = goal_vec.GetAngle();
	double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
    a += noise;

	//TODO noisy speed
    MyVector move(a, ped.vel/freq, 0);
    ped.pos.x += move.dw;
    ped.pos.y += move.dh;
    return;
}

void WorldModel::PedStep(PedStruct &ped, double& random) {

    const COORD& goal = goals[ped.goal];
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return;
	}

	MyVector goal_vec(goal.x - ped.pos.x, goal.y - ped.pos.y);

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
//		cout << "ped vel = " << ped.vel << endl;
		MyVector move(a, ped.vel/freq, 0);
		ped.pos.x += move.dw;
		ped.pos.y += move.dh;
	}

    return;
}


double gaussian_prob(double x, double stddev) {
    double a = 1.0 / stddev / sqrt(2 * M_PI);
    double b = - x * x / 2.0 / (stddev * stddev);
    return a * exp(b);
}


void WorldModel::PedStepDeterministic(PedStruct& ped, int step) {
    const COORD& goal = goals[ped.goal];
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return;
	}

	MyVector goal_vec(goal.x - ped.pos.x, goal.y - ped.pos.y);
    goal_vec.AdjustLength(step * ped.vel / freq);
    ped.pos.x += goal_vec.dw;
    ped.pos.y += goal_vec.dh;
}

void WorldModel::PedStepCurVel(PedStruct& ped, COORD vel) {

//	if (vel.Length()>=1e-3){
//		cout << "recaling vel with "<< (1.0/freq) << endl;
//	}
//	vel = vel * (1.0/freq);
    ped.pos.x += vel.x * (1.0/freq);
    ped.pos.y += vel.y * (1.0/freq);
}

double WorldModel::pedMoveProb(COORD prev, COORD curr, int goal_id) {
	const double K = 0.001;
    const COORD& goal = goals[goal_id];
	double move_dist = Norm(curr.x-prev.x, curr.y-prev.y),
		   goal_dist = Norm(goal.x-prev.x, goal.y-prev.y);
	double sensor_noise = 0.1;
    if(ModelParams::is_simulation) sensor_noise = 0.02;

    bool debug=false;
	// CHECK: beneficial to add back noise?
	if(debug) cout<<"goal id "<<goal_id<<endl;
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

double WorldModel::pedMoveProb(COORD prev, COORD curr, int ped_id, int goal_id, int ped_mode) {
	const double K = 0.001;
//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

	double move_dist = Norm(curr.x-prev.x, curr.y-prev.y);

	COORD goal;

	if(ped_mode == PED_ATT){
		if(ped_id >= ped_mean_dirs.size()){
			cout << "Encountering overflowed ped id " << ped_id;
			while (ped_mean_dirs.size() < ped_id + 1){
				cout << "adding init mean dir for ped " << ped_mean_dirs.size()<< endl;
				vector<COORD> dirs(goals.size());
				ped_mean_dirs.push_back(dirs);
			}
		}
		if(goal_id >= ped_mean_dirs[0].size()){
			cerr << "Encountering overflowed goal id " << goal_id;
			exit(-1);
		}
		goal = ped_mean_dirs[ped_id][goal_id] + prev;
	}
	else{
		goal = goals[goal_id];
	}
//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

	double goal_dist = Norm(goal.x-prev.x, goal.y-prev.y);

	double sensor_noise = 0.1;
    if(ModelParams::is_simulation) sensor_noise = 0.02;

    bool debug=false;
	// CHECK: beneficial to add back noise?
	if(debug) cout<<"goal id "<<goal_id<<endl;
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		logd <<"stop intention" << endl;

		return (move_dist < sensor_noise) ? 0.4 : 0;
	} else {
//		cout << __FUNCTION__ << "@" << __LINE__ << endl;

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
		if(ped_mode == PED_ATT){
			double mean_vel = ped_mean_dirs[ped_id][goal_id].Length();
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

void WorldModel::updatePedBelief(PedBelief& b, const PedStruct& curr_ped) {
//    const double ALPHA = 0.8;

    const double ALPHA = 0.1;

	const double SMOOTHING=ModelParams::BELIEF_SMOOTHING;

    bool debug=false;

    for(int i=0; i<goals.size(); i++) {

    	// Attentive mode
    	logd << "TEST ATT" << endl;
		double prob = pedMoveProb(b.pos, curr_ped.pos, curr_ped.id, i, PED_ATT);
		if(debug) cout << "attentive likelihood " << i << ": " << prob << endl;
        b.prob_modes_goals[PED_ATT][i] *=  prob;
		// Keep the belief noisy to avoid aggressive policies
		b.prob_modes_goals[PED_ATT][i] += SMOOTHING / goals.size()/2.0; // CHECK: decrease or increase noise

		// Detracted mode
    	logd << "TEST DIS" << endl;
		prob = pedMoveProb(b.pos, curr_ped.pos, curr_ped.id, i, PED_DIS);
		if(debug) cout << "Detracted likelihood " << i << ": " << prob << endl;
		b.prob_modes_goals[PED_DIS][i] *=  prob;
		// Important: Keep the belief noisy to avoid aggressive policies
		b.prob_modes_goals[PED_DIS][i] += SMOOTHING / goals.size()/2.0; // CHECK: decrease or increase noise
	}

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

	if(debug) {
        for(double w: b.prob_modes_goals[PED_ATT]) {
            cout << w << " ";
        }
        cout << endl;
        for(double w: b.prob_modes_goals[PED_DIS]) {
			cout << w << " ";
		}
		cout << endl;
    }

    // normalize
    double total_weight_att = accumulate(b.prob_modes_goals[PED_ATT].begin(), b.prob_modes_goals[PED_ATT].end(), double(0.0));
    double total_weight_dis = accumulate(b.prob_modes_goals[PED_DIS].begin(), b.prob_modes_goals[PED_DIS].end(), double(0.0));
    double total_weight = total_weight_att + total_weight_dis;
	if(debug) cout << "[updatePedBelief] total_weight = " << total_weight << endl;
    for(double& w : b.prob_modes_goals[PED_ATT]) {
        w /= total_weight;
    }
    for(double& w : b.prob_modes_goals[PED_DIS]) {
		w /= total_weight;
	}

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

    double moved_dist = COORD::EuclideanDistance(b.pos, curr_ped.pos);
    b.vel = ALPHA * b.vel + (1-ALPHA) * moved_dist * ModelParams::control_freq;
	b.pos = curr_ped.pos;

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

}

PedBelief WorldModel::initPedBelief(const PedStruct& ped) {
    PedBelief b;
    b.id = ped.id;
    b.pos = ped.pos;
    b.vel = ModelParams::PED_SPEED;
//    cout << "PED_DIS + 1= "<< PED_DIS + 1 << endl;
    int num_types = PED_DIS + 1;
    for(int i =0 ; i < num_types; i++){
    	b.prob_modes_goals.push_back(vector<double>(goals.size(), 1.0/goals.size()/num_types));
    }

    return b;
}

double timestamp() {
    static double starttime=get_time_second();
    return get_time_second()-starttime;
}

void WorldStateTracker::cleanPed() {
    vector<Pedestrian> ped_list_new;
    for(int i=0;i<ped_list.size();i++)
    {
        bool insert=true;
        double w1,h1;
        w1=ped_list[i].w;
        h1=ped_list[i].h;
        for(const auto& p: ped_list_new) {
            double w2,h2;
            w2=p.w;
            h2=p.h;
            if (abs(w1-w2)<=0.1&&abs(h1-h2)<=0.1) {
                insert=false;
                break;
            }
        }
        if (timestamp() - ped_list[i].last_update > 0.2) insert=false;
        if (insert)
            ped_list_new.push_back(ped_list[i]);
    }
    ped_list=ped_list_new;
}

double get_timestamp(){
	return Globals::ElapsedTime(init_time);
}

void WorldStateTracker::updatePed(const Pedestrian& ped, bool doPrint){
    int i=0;

//    raise(SIGABRT);

    bool no_move = true;
    for(;i<ped_list.size();i++) {
        if (ped_list[i].id==ped.id) {

//        	cout <<"[updatePed] updating existing ped" << ped.id << endl;
            //found the corresponding ped,update the pose
        	double duration =  get_timestamp()-ped_list[i].last_update;

        	if (duration < 0.1 / Globals::config.time_scale){
        		no_move = false;
//				cout << "ped update skipped: duration "<< duration << endl;
				return;
//        		raise(SIGABRT);
        	}

        	ped_list[i].vel.x = (ped.w - ped_list[i].w) / duration;
			ped_list[i].vel.y = (ped.h - ped_list[i].h) / duration;

			if (ped_list[i].vel.Length()>3.0){
				cerr << "WARNING: Unusual ped " << i << " speed: " << ped_list[i].vel.Length() << endl;
				raise(SIGABRT);
			}

			if (ped_list[i].vel.Length()>1e-4){
				no_move = false;
			}

        	ped_list[i].w=ped.w;
            ped_list[i].h=ped.h;

            ped_list[i].last_update = get_timestamp();
//            cout <<"[updatePed] existing ped updated" << ped.id << endl;
            break;
        }
        if (abs(ped_list[i].w-ped.w)<=0.1 && abs(ped_list[i].h-ped.h)<=0.1)   //overlap
        {
        	//if (doPrint) cout <<"[updatePed] overlapping ped skipped" << ped.id << endl;
            //return;
        }
    }

    if (i==ped_list.size()) {
    	no_move = false;
        //not found, new ped
//    	cout <<"[updatePed] updating new ped" << ped.id << endl;

        ped_list.push_back(ped);

        ped_list.back().vel.x = 0;
        ped_list.back().vel.y = 0;
        ped_list.back().last_update = get_timestamp();
//        cout <<"[updatePed] new ped added" << ped.id << endl;
    }

    if (no_move){
		cout << __FUNCTION__ << " no_move ped "<< ped.id <<
				" caught: vel " << ped_list[i].vel.x <<" "<< ped_list[i].vel.y << endl;
//		raise(SIGABRT);
    }
}

COORD WorldStateTracker::getPedVel( int ped_id ){
	int i=0;
	for(;i<ped_list.size();i++) {
		if (ped_list[i].id == ped_id){

			return ped_list[i].vel;
		}
	}
	cout << __FUNCTION__ << " no matching ped found for vel" << endl;
	raise(SIGABRT);
}

void WorldStateTracker::removePeds() {
	ped_list.resize(0);
}

void WorldStateTracker::updateCar(const CarStruct car) {
    carpos=car.pos;
    carvel=car.vel;
    car_heading_dir=car.heading_dir;
}

bool WorldStateTracker::emergency() {
    //TODO improve emergency stop to prevent the false detection of leg
    double mindist = numeric_limits<double>::infinity();
    for(auto& ped : ped_list) {
		COORD p(ped.w, ped.h);
        double d = COORD::EuclideanDistance(carpos, p);
        if (d < mindist) mindist = d;
    }
	cout << "emergency mindist = " << mindist << endl;
	return (mindist < 0.5);
}

void WorldStateTracker::updateVel(double vel) {
	carvel = vel;
}

vector<WorldStateTracker::PedDistPair> WorldStateTracker::getSortedPeds(bool doPrint) {
    // sort peds
    vector<PedDistPair> sorted_peds;

    if(doPrint) cout << "[getSortedPeds] state tracker ped_list size " << ped_list.size() << endl;
    for(const auto& p: ped_list) {
        COORD cp(p.w, p.h);
        float dist = COORD::EuclideanDistance(cp, carpos);

        COORD ped_dir = COORD(p.w, p.h) - carpos;
        COORD car_dir = COORD(cos(car_heading_dir), sin(car_heading_dir));
        double proj = (ped_dir.x*car_dir.x + ped_dir.y*car_dir.y)/ ped_dir.Length();
        if (proj > 0.6)
        	dist -= 2.0;
        if (proj < 0.7)
            dist += 2.0;
        sorted_peds.push_back(PedDistPair(dist, p));

		if(doPrint) cout << "[getSortedPeds] ped id:"<< p.id << endl;

    }
    sort(sorted_peds.begin(), sorted_peds.end(),
            [](const PedDistPair& a, const PedDistPair& b) -> bool {
                return a.first < b.first;
            });


    return sorted_peds;
}

PomdpState WorldStateTracker::getPomdpState() {
    auto sorted_peds = getSortedPeds();

    // construct PomdpState
    PomdpState pomdpState;
    pomdpState.car.pos = carpos;
    pomdpState.car.vel = carvel;
	pomdpState.car.heading_dir = /*0*/car_heading_dir;
    pomdpState.num = sorted_peds.size();

	if (pomdpState.num > ModelParams::N_PED_IN) {
		pomdpState.num = ModelParams::N_PED_IN;
	}

    for(int i=0;i<pomdpState.num;i++) {
        const auto& ped = sorted_peds[i].second;
        pomdpState.peds[i].pos.x=ped.w;
        pomdpState.peds[i].pos.y=ped.h;
		pomdpState.peds[i].id = ped.id;
		pomdpState.peds[i].goal = -1;
    }
	return pomdpState;
}

void WorldBeliefTracker::update() {

	// Update peds
    auto sorted_peds = stateTracker.getSortedPeds();
    map<int, PedStruct> newpeds;
    for(const auto& dp: sorted_peds) {
        auto& p = dp.second;
        PedStruct ped(COORD(p.w, p.h), -1, p.id);
        newpeds[p.id] = ped;
    }

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

    // remove disappeared peds
    vector<int> peds_to_remove;
    for(const auto& p: peds) {
        if (newpeds.find(p.first) == newpeds.end()) {
            peds_to_remove.push_back(p.first);
            logi << "["<<__FUNCTION__<< "]" << " removing ped "<< p.first << endl;
        }
    }
//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

    for(const auto& i: peds_to_remove) {
        peds.erase(i);
    }

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;


    // Run ORCA for all possible hidden variable combinations

    model.PrepareAttentivePedMeanDirs(peds, car);

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;


    model.PrintMeanDirs(peds, newpeds);

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;


    // update car
    car.pos = stateTracker.carpos;
    car.vel = stateTracker.carvel;
	car.heading_dir = /*0*/stateTracker.car_heading_dir;


    // update existing peds
    for(auto& kv : peds) {

    	if (newpeds.find(kv.first) == newpeds.end()){
    		cerr << "error !!" << " updating non existing id in new ped list"<< endl;
    		exit(-1);
    	}
        model.updatePedBelief(kv.second, newpeds[kv.first]);
    }

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

    // add new peds
    for(const auto& kv: newpeds) {
		auto& p = kv.second;
        if (peds.find(p.id) == peds.end()) {
            peds[p.id] = model.initPedBelief(p);
        }
    }

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

	sorted_beliefs.clear();
	for(const auto& dp: sorted_peds) {
		auto& p = dp.second;
		sorted_beliefs.push_back(peds[p.id]);
	}

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

	cur_time_stamp = SolverPrior::get_timestamp();

//	cout << __FUNCTION__ << "@" << __LINE__ << endl;

    return;
}

int PedBelief::sample_goal() const {
    double r = /*double(rand()) / RAND_MAX*/ Random::RANDOM.NextDouble();
    int i = 0;
    r -= prob_goals[i];
    while(r > 0) {
        i++;
        r -= prob_goals[i];
    }
    return i;
}

void PedBelief::sample_goal_mode(int& goal, int& mode) const {
//	logd << "[PedBelief::sample_goal_mode] " << endl;

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
    	logd << "[WARNING]: [PedBelief::sample_goal_mode] execess probability " << r << endl;
    	goal = 0;
    	mode = 0;
    }
    else{
//    	logd << "[PedBelief::sample_goal_mode] sampled values " << goal << " " << mode << endl;
    }
}

int PedBelief::maxlikely_goal() const {
    double ml = 0;
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
		auto& p = sorted_beliefs[i];
		if (COORD::EuclideanDistance(p.pos, car.pos) <= ModelParams::LASER_RANGE) {
            cout << "ped belief " << p.id << ": ";
            for (int g = 0; g < p.prob_goals.size(); g ++)
                cout << " " << p.prob_goals[g];
            cout << endl;
		}
    }
}

PomdpState WorldBeliefTracker::text() const{
	if (logging::level()>=4){
		for(int i=0; i < sorted_beliefs.size() && i < min(6,ModelParams::N_PED_IN); i++) {
			auto& p = sorted_beliefs[i];
			cout << "[WorldBeliefTracker::text] " << this << "->p:" << &p << endl;
			cout << " sorted ped " << i << endl;
			cout << "-prob_goals: " << p.prob_goals << endl;
			for (int i=0;i<p.prob_goals.size();i++){
				cout << p.prob_goals[i] << " ";
			}
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
		auto& p = sorted_beliefs[i];

		if (COORD::EuclideanDistance(p.pos, car.pos) < ModelParams::LASER_RANGE) {
			s.peds[s.num].pos = p.pos;
//			s.peds[s.num].goal = p.sample_goal();
			p.sample_goal_mode(s.peds[s.num].goal, s.peds[s.num].mode);
			s.peds[s.num].id = p.id;
            s.peds[s.num].vel = p.vel;
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

    cout << "Num peds for planning: " << particles[0].num << endl;

    return particles;
}

vector<PedStruct> WorldBeliefTracker::predictPeds() {
    vector<PedStruct> prediction;

    for(const auto& p: sorted_beliefs) {
        double dist = COORD::EuclideanDistance(p.pos, car.pos);
        int step = (p.vel + car.vel>1e-5)?int(dist / (p.vel + car.vel) * ModelParams::control_freq):100000;

        for(int j=0; j<1; j++) {
            int goal = p.maxlikely_goal();
            PedStruct ped0(p.pos, goal, p.id);
			ped0.vel = p.vel;

            for(int i=0; i<4; i++) {
                PedStruct ped = ped0;
                model.PedStepDeterministic(ped, step + i);
                prediction.push_back(ped);
            }
        }
    }
    return prediction;
}

PomdpState WorldBeliefTracker::predictPedsCurVel(PomdpState* ped_state, double acc, double steering) {

//	cout << __FUNCTION__ << "applying action " << acc <<"/"<< steering << endl;

	PomdpState predicted_state = *ped_state;

//	if (acc >0){
//		cout << "source ped list addr "<< ped_state->peds << endl;
//		cout << "des ped list addr "<< predicted_state.peds << endl;
//		cout << predicted_state.num << " peds active" << endl;
//		raise(SIGABRT);
//	}

//	model.RobStepCurVel(predicted_state.car);
	model.RobStepCurAction(predicted_state.car, acc, steering);

	for(int i =0 ; i< predicted_state.num ; i++) {
		auto & p = predicted_state.peds[i];
		COORD ped_vel = stateTracker.getPedVel(p.id);

//		if (acc >0 ){
//			if (ped_vel.Length()<1e-3){
//				cout << "ped " << p.id << " vel = " << ped_vel.Length() << endl;
//				cout.flush();
//				raise(SIGABRT);
//			}
//			else{
//				cout << "ped " << p.id << " vel = " << ped_vel.x <<" "<< ped_vel.y << endl;
//			}
//		}

		model.PedStepCurVel(p, ped_vel);
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
		out << "Ped Pos: " << state.peds[i].pos.x << " " << state.peds[i].pos.y << endl;
		out << "Goal: " << state.peds[i].goal << endl;
		out << "id: " << state.peds[i].id << endl;
	}
	out << "Vel: " << state.car.vel << endl;
	out<<  "num  " << state.num << endl;
	double min_dist = COORD::EuclideanDistance(carpos, state.peds[0].pos);
	out << "MinDist: " << min_dist << endl;
}


void WorldModel::RVO2PedStep(PomdpStateWorld& state, Random& random){

	PedStruct* peds = state.peds;
	CarStruct car = state.car;
	int num_ped = state.num;
    int threadID=GetThreadID();

    // Construct a new set of agents every time
	ped_sim_[threadID]->clearAllAgents();

	//adding pedestrians
	int num_att_pes = 0;
	for(int i=0; i<num_ped; i++){
		if(state.peds[i].mode==PED_ATT){
			ped_sim_[threadID]->addAgent(RVO::Vector2(peds[i].pos.x, peds[i].pos.y));
			num_att_pes++;
		}
	}

	// adding car as a "special" pedestrian
	add_car_agents(num_ped, car);

	// Set the preferred velocity for each agent.
	int agent = 0;
	for (size_t i = 0; i < num_ped; ++i) {
		if(state.peds[i].mode==PED_ATT){
		   int goal_id = peds[i].goal;
			if (goal_id >= goals.size()-1) { /// stop intention
				ped_sim_[threadID]->setAgentPrefVelocity(agent, RVO::Vector2(0.0f, 0.0f));
			} else{
				RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
				if ( absSq(goal - ped_sim_[threadID]->getAgentPosition(agent)) < ped_sim_[threadID]->getAgentRadius(agent) * ped_sim_[threadID]->getAgentRadius(agent) ) {
					// Agent is within one radius of its goal, set preferred velocity to zero
					ped_sim_[threadID]->setAgentPrefVelocity(agent, RVO::Vector2(0.0f, 0.0f));
				} else {
					// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
					ped_sim_[threadID]->setAgentPrefVelocity(agent, normalize(goal - ped_sim_[threadID]->getAgentPosition(agent))*ModelParams::PED_SPEED);
				}
			}
			agent++;
		}
	}

	// Update positions of only all peds in the RVO simulator
	ped_sim_[threadID]->doStep();
    agent = 0;

    // Update positions of only attentive peds
    for(int i=0; i<num_ped; i++){
    	if(state.peds[i].mode==PED_ATT){
    		peds[i].pos.x=ped_sim_[threadID]->getAgentPosition(agent).x();// + random.NextGaussian() * (ped_sim_[threadID]->getAgentPosition(agent).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
    		peds[i].pos.y=ped_sim_[threadID]->getAgentPosition(agent).y();// + random.NextGaussian() * (ped_sim_[threadID]->getAgentPosition(agent).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

    		if(use_noise_in_rvo){
    			peds[i].pos.x+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
    			peds[i].pos.y+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
    		}
    		agent++;
    	}
    }
}

void WorldModel::RVO2PedStep(PedStruct peds[], Random& random, int num_ped, CarStruct car){
    int threadID=GetThreadID();
    ped_sim_[threadID]->clearAllAgents();

    //adding pedestrians
    for(int i=0; i<num_ped; i++){
        ped_sim_[threadID]->addAgent(RVO::Vector2(peds[i].pos.x, peds[i].pos.y));
    }

    // adding car as a "special" pedestrian
    add_car_agents(num_ped, car);

    // Set the preferred velocity for each agent.
    for (size_t i = 0; i < num_ped; ++i) {
        int goal_id = peds[i].goal;
        if (goal_id >= goals.size()-1) { /// stop intention
            ped_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
        } else{
            RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
            if ( absSq(goal - ped_sim_[threadID]->getAgentPosition(i)) < ped_sim_[threadID]->getAgentRadius(i) * ped_sim_[threadID]->getAgentRadius(i) ) {
                // Agent is within one radius of its goal, set preferred velocity to zero
                ped_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
            } else {
                // Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
                ped_sim_[threadID]->setAgentPrefVelocity(i, normalize(goal - ped_sim_[threadID]->getAgentPosition(i))*ModelParams::PED_SPEED);
            }
        }
        
    }

    ped_sim_[threadID]->doStep();

    for(int i=0; i<num_ped; i++){
        peds[i].pos.x=ped_sim_[threadID]->getAgentPosition(i).x();// + random.NextGaussian() * (ped_sim_[threadID]->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        peds[i].pos.y=ped_sim_[threadID]->getAgentPosition(i).y();// + random.NextGaussian() * (ped_sim_[threadID]->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        if(use_noise_in_rvo){
			peds[i].pos.x+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
			peds[i].pos.y+= random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
		}
    }

}


void WorldModel::RVO2PedStep(PedStruct peds[], double& random, int num_ped, CarStruct car){
    int threadID=GetThreadID();
    ped_sim_[threadID]->clearAllAgents();

    //adding pedestrians
    for(int i=0; i<num_ped; i++){
        ped_sim_[threadID]->addAgent(RVO::Vector2(peds[i].pos.x, peds[i].pos.y));
    }

    // adding car as a "special" pedestrian
	add_car_agents(num_ped, car);

    // Set the preferred velocity for each agent.
    for (size_t i = 0; i < num_ped; ++i) {

    	if(peds[i].mode==PED_ATT){
			int goal_id = peds[i].goal;
			if (goal_id >= goals.size()-1) { /// stop intention
				ped_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
			} else{
				RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
				if ( absSq(goal - ped_sim_[threadID]->getAgentPosition(i)) < ped_sim_[threadID]->getAgentRadius(i) * ped_sim_[threadID]->getAgentRadius(i) ) {
					// Agent is within one radius of its goal, set preferred velocity to zero
					ped_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
				} else {
					// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
					ped_sim_[threadID]->setAgentPrefVelocity(i, normalize(goal - ped_sim_[threadID]->getAgentPosition(i))*ModelParams::PED_SPEED);
				}
			}
    	}
    }

    ped_sim_[threadID]->doStep();

    for(int i=0; i<num_ped; i++){

    	if(peds[i].mode==PED_ATT){
			peds[i].pos.x=ped_sim_[threadID]->getAgentPosition(i).x();// + rNum * (ped_sim_[threadID]->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
			peds[i].pos.y=ped_sim_[threadID]->getAgentPosition(i).y();// + rNum * (ped_sim_[threadID]->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

			if(use_noise_in_rvo){
				double rNum=GenerateGaussian(random);
				peds[i].pos.x+= rNum * ModelParams::NOISE_PED_POS / freq;
				rNum=GenerateGaussian(rNum);
				peds[i].pos.y+= rNum * ModelParams::NOISE_PED_POS / freq;
			}
    	}
    }
}

COORD WorldModel::DistractedPedMeanDir(COORD& ped, int goal_id) {
	COORD dir(0,0);
	const COORD& goal = goals[goal_id];
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return dir;
	}

	MyVector goal_vec(goal.x - ped.x, goal.y - ped.y);
//	goal_vec.AdjustLength(ped.vel / freq);

	dir.x = goal_vec.dw;
	dir.y = goal_vec.dh;

	return dir;
}

COORD WorldModel::AttentivePedMeanDir(int ped_id, int goal_id){
	return ped_mean_dirs[ped_id][goal_id];
}

#include "Vector2.h"

void WorldModel::add_car_agents(int num_ped, CarStruct& car){
    int threadID=GetThreadID();

	double car_x, car_y, car_yaw;

	car_yaw = car.heading_dir;

	if(ModelParams::car_model == "pomdp_car"){
		/// for pomdp car
		car_x = car.pos.x - CAR_LENGTH/2.0f*cos(car_yaw);
		car_y = car.pos.y - CAR_LENGTH/2.0f*sin(car_yaw);
		double car_radius = sqrt(pow(CAR_WIDTH/2.0f, 2) + pow(CAR_LENGTH/2.0f,2)) + CAR_EXPAND_SIZE;
		ped_sim_[threadID]->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		ped_sim_[threadID]->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw)));
	} else if(ModelParams::car_model == "audi_r8"){
		/// for audi r8
		double car_radius = 1.15f;
		double car_radius_large = 1.6f;

		ped_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x, car.pos.y), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		ped_sim_[threadID]->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the num_ped-th pedestrian is the car. set its prefered velocity
		ped_sim_[threadID]->setAgentPedID(num_ped,-1);

		ped_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x + 0.56 * 2.33 * cos(car_yaw), car.pos.y + 1.4* sin(car_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		ped_sim_[threadID]->setAgentPrefVelocity(num_ped+1, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the num_ped-th pedestrian is the car. set its prefered velocity
		ped_sim_[threadID]->setAgentPedID(num_ped+1,-2);

		ped_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x + 0.56*3.66 * cos(car_yaw), car.pos.y + 2.8* sin(car_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		ped_sim_[threadID]->setAgentPrefVelocity(num_ped+2, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the num_ped-th pedestrian is the car. set its prefered velocity
		ped_sim_[threadID]->setAgentPedID(num_ped+2,-3);

		ped_sim_[threadID]->addAgent(RVO::Vector2(car.pos.x + 0.56*5 * cos(car_yaw), car.pos.y + 2.8* sin(car_yaw)), 4.0f, 2, 1.0f, 2.0f, car_radius_large, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
		ped_sim_[threadID]->setAgentPrefVelocity(num_ped+3, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the num_ped-th pedestrian is the car. set its prefered velocity
		ped_sim_[threadID]->setAgentPedID(num_ped+3,-4);
	}
}

void WorldModel::PrepareAttentivePedMeanDirs(std::map<int, PedBelief> peds, CarStruct& car){
	int num_ped = peds.size();

	logd << "num_peds in belief tracker: " << num_ped << endl;

    int threadID=GetThreadID();

    // Construct a new set of agents every time
	ped_sim_[threadID]->clearAllAgents();

	//adding pedestrians
	for(int i=0; i<num_ped; i++){
		ped_sim_[threadID]->addAgent(RVO::Vector2(peds[i].pos.x, peds[i].pos.y));
	}

	// adding car as a "special" pedestrian
	add_car_agents(num_ped, car);

	// Set the preferred velocity for each agent.

	ped_sim_[threadID]->doPreStep();// build kd tree and find neighbors for peds


	if(num_ped > ped_mean_dirs.size()){
		cout << "Encountering overflowed peds list of size " << num_ped;
		while (ped_mean_dirs.size() < num_ped){
			cout << "adding init mean dir for ped " << ped_mean_dirs.size()<< endl;
			vector<COORD> dirs(goals.size());
			ped_mean_dirs.push_back(dirs);
		}
	}

	for (size_t i = 0; i < num_ped; ++i) {
		// For each ego ped
	    for(int goal_id=0; goal_id<goals.size(); goal_id++) {

	    	RVO::Vector2 ori_pos(ped_sim_[threadID]->getAgentPosition(i).x(),  ped_sim_[threadID]->getAgentPosition(i).y());

			// Set preferred velocity for the ego ped according to goal_id
			// Leave other pedestrians to have default preferred velocity

			if (goal_id >= goals.size()-1) { /// stop intention
				ped_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
			} else{
				RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
				if ( absSq(goal - ped_sim_[threadID]->getAgentPosition(i)) < ped_sim_[threadID]->getAgentRadius(i) * ped_sim_[threadID]->getAgentRadius(i) ) {
					// Agent is within one radius of its goal, set preferred velocity to zero
					ped_sim_[threadID]->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
				} else {
					// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
					ped_sim_[threadID]->setAgentPrefVelocity(i, normalize(goal - ped_sim_[threadID]->getAgentPosition(i))*ModelParams::PED_SPEED);
				}
			}

			ped_sim_[threadID]->doStepForPed(i);

			COORD dir;
			dir.x = ped_sim_[threadID]->getAgentPosition(i).x() - peds[i].pos.x;
			dir.y = ped_sim_[threadID]->getAgentPosition(i).y() - peds[i].pos.y;

			logd << "[PrepareAttentivePedMeanDirs] ped_mean_dirs len=" << ped_mean_dirs.size()
					<< " goal_list len=" << ped_mean_dirs[0].size() << "\n";

			logd << "[PrepareAttentivePedMeanDirs] i=" << i << " goal_id=" << goal_id << "\n";

			ped_mean_dirs[i][goal_id]=dir;

	    	// reset ped state
			ped_sim_[threadID]->setAgentPosition(i, ori_pos);
	    }
	}
}

void WorldModel::PrintMeanDirs(std::map<int, PedBelief> old_peds, map<int, PedStruct>& curr_peds){
	if(logging::level()>=4){
		int num_ped = old_peds.size();

		for (size_t i = 0; i < 6; ++i) {
			cout <<"ped "<< i << endl;
			// For each ego ped
			auto& old_ped = old_peds[i];
			auto& cur_ped = curr_peds[i];

			cout << "prev pos: "<< old_ped.pos.x << "," << old_ped.pos.y << endl;
			cout << "cur pos: "<< cur_ped.pos.x << "," << cur_ped.pos.y << endl;

			COORD dir = cur_ped.pos - old_ped.pos;

			cout << "dir: " << endl;
			for(int goal_id=0; goal_id<goals.size(); goal_id++) {
				cout <<goal_id<<"," << dir.x << "," << dir.y << " ";
			}
			cout<< endl;

			cout << "Meandir: " << endl;
		    for(int goal_id=0; goal_id<goals.size(); goal_id++) {
		    	cout <<goal_id<<","<<ped_mean_dirs[i][goal_id].x << ","<< ped_mean_dirs[i][goal_id].y << " ";
		    }
		    cout<< endl;

			cout << "Att probs: " << endl;
		    for(int goal_id=0; goal_id<goals.size(); goal_id++) {
		    	logd <<"ped, goal, ped_mean_dirs.size() =" << cur_ped.id << " "
		    					<< goal_id <<" "<< ped_mean_dirs.size() << endl;
				double prob = pedMoveProb(old_ped.pos, cur_ped.pos, cur_ped.id, goal_id, PED_ATT);

				cout <<"prob: "<< goal_id<<","<<prob << endl;
			}
		    cout<< endl;
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

bool WorldModel::CheckCarWithObsLine(const CarStruct& car, COORD& start_point, COORD& end_point, int flag){
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

    logd << "[WorldModel::"<<__FUNCTION__<<"] Calculate mean dir for "<< state.num<< " peds and "<<goals.size()<<" goals"<< endl;

	AttentivePedMeanStep(state.peds, state.num, state.car);


	for(int ped_id=0;ped_id< state.num; ped_id++)
		for(int goal_id=0;goal_id< goals.size(); goal_id++){
			PedStruct updated_ped=state.peds[ped_id];
			// assume that the pedestrian takes the goal
			updated_ped.goal=goal_id;
			PedMotionDirDeterministic(updated_ped);
			// record the movement direction of the pedestrian taking the goal
			ped_mean_dirs[ped_id][goal_id]=updated_ped.pos-state.peds[ped_id].pos;
			ped_mean_speeds[ped_id][goal_id]=COORD::EuclideanDistance(updated_ped.pos,
					state.peds[ped_id].pos);
		}

    logd << "[WorldModel::"<<__FUNCTION__<<"] Mean dirs :"<<endl;

	return true;
}
*/




