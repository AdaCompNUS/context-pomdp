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

	double move_dist = Norm(curr.x-prev.x, curr.y-prev.y);

	COORD goal;

	if(ped_mode == PED_ATT)
		goal = ped_mean_dirs[ped_id][goal_id];
	else{
		goal = goals[goal_id];
	}

	double goal_dist = Norm(goal.x-prev.x, goal.y-prev.y);

	double sensor_noise = 0.1;
    if(ModelParams::is_simulation) sensor_noise = 0.02;

    bool debug=false;
	// CHECK: beneficial to add back noise?
	if(debug) cout<<"goal id "<<goal_id<<endl;
	if (goal.x == -1 && goal.y == -1) {  //stop intention
		return (move_dist < sensor_noise) ? 0.4 : 0;
	} else {
		if (move_dist < sensor_noise) return 0;

		if(ped_mode == PED_ATT){
			// TODO: change to guassian around the true next position

		}
		else{
			double cosa = DotProduct(curr.x-prev.x, curr.y-prev.y, goal.x-prev.x, goal.y-prev.y) / (move_dist * goal_dist);
			if(cosa >1) cosa = 1;
			else if(cosa < -1) cosa = -1;
			double angle = acos(cosa);
			return gaussian_prob(angle, ModelParams::NOISE_GOAL_ANGLE) + K;
		}
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
    const double ALPHA = 0.8;
	const double SMOOTHING=ModelParams::BELIEF_SMOOTHING;

    bool debug=false;

    for(int i=0; i<goals.size(); i++) {

    	// Attentive mode
		double prob = pedMoveProb(b.pos, curr_ped.pos, curr_ped.id, i, PED_ATT);
		if(debug) cout << "attentive likelihood " << i << ": " << prob << endl;
        b.prob_modes_goals[PED_ATT][i] *=  prob;
		// Keep the belief noisy to avoid aggressive policies
		b.prob_modes_goals[PED_ATT][i] += SMOOTHING / goals.size()/2.0; // CHECK: decrease or increase noise

		// Detracted mode
		prob = pedMoveProb(b.pos, curr_ped.pos, curr_ped.id, i, PED_DIS);
		if(debug) cout << "Detracted likelihood " << i << ": " << prob << endl;
		b.prob_modes_goals[PED_DIS][i] *=  prob;
		// Important: Keep the belief noisy to avoid aggressive policies
		b.prob_modes_goals[PED_DIS][i] += SMOOTHING / goals.size()/2.0; // CHECK: decrease or increase noise
	}
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

    double moved_dist = COORD::EuclideanDistance(b.pos, curr_ped.pos);
    b.vel = ALPHA * b.vel + (1-ALPHA) * moved_dist * ModelParams::control_freq;
	b.pos = curr_ped.pos;
}

PedBelief WorldModel::initPedBelief(const PedStruct& ped) {
    PedBelief b;
    b.id = ped.id;
    b.pos = ped.pos;
    b.vel = ModelParams::PED_SPEED;
    for(int i =0 ; i < ModelParams::N_PED_IN; i++){
    	b.prob_modes_goals.push_back(vector<double>(goals.size(), 1.0/goals.size()));
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

void WorldStateTracker::updatePed(const Pedestrian& ped, bool doPrint){
    int i=0;
    for(;i<ped_list.size();i++) {
        if (ped_list[i].id==ped.id) {
            //found the corresponding ped,update the pose
            ped_list[i].w=ped.w;
            ped_list[i].h=ped.h;
            ped_list[i].last_update = timestamp();
            if (doPrint) cout <<"[updatePed] existing ped updated" << ped.id << endl;
            break;
        }
        if (abs(ped_list[i].w-ped.w)<=0.1 && abs(ped_list[i].h-ped.h)<=0.1)   //overlap
        {
        	//if (doPrint) cout <<"[updatePed] overlapping ped skipped" << ped.id << endl;
            //return;
        }
    }
    if (i==ped_list.size()) {
        //not found, new ped
        ped_list.push_back(ped);
        ped_list.back().last_update = timestamp();
        if (doPrint) cout <<"[updatePed] new ped added" << ped.id << endl;
    }
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

    // remove disappeared peds
    vector<int> peds_to_remove;
    for(const auto& p: peds) {
        if (newpeds.find(p.first) == newpeds.end()) {
            peds_to_remove.push_back(p.first);
            logi << "["<<__FUNCTION__<< "]" << " removing ped "<< p.first << endl;
        }
    }
    for(const auto& i: peds_to_remove) {
        peds.erase(i);
    }

    // Run ORCA for all possible hidden variable combinations

    model.PrepareAttentivePedMeanDirs(peds, car);

    // update car
    car.pos = stateTracker.carpos;
    car.vel = stateTracker.carvel;
	car.heading_dir = /*0*/stateTracker.car_heading_dir;

    // update existing peds
    for(auto& kv : peds) {
        model.updatePedBelief(kv.second, newpeds[kv.first]);
    }

    // add new peds
    for(const auto& kv: newpeds) {
		auto& p = kv.second;
        if (peds.find(p.id) == peds.end()) {
            peds[p.id] = model.initPedBelief(p);
        }
    }

	sorted_beliefs.clear();
	for(const auto& dp: sorted_peds) {
		auto& p = dp.second;
		sorted_beliefs.push_back(peds[p.id]);
	}


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

PomdpState WorldBeliefTracker::sample() {
    PomdpState s;
    s.car = car;

	s.num = 0;
    for(int i=0; i < sorted_beliefs.size() && i < ModelParams::N_PED_IN; i++) {
		auto& p = sorted_beliefs[i];

//		logd << "[WorldBeliefTracker::sample] " << this << "->p:" << &p << endl;
//		logd << "-prob_goals: " << p.prob_goals << endl;
//		for (int i=0;i<p.prob_goals.size();i++){
//			logd << p.prob_goals[i] << " ";
//		}
//		logd << "-prob_modes_goals: " << p.prob_modes_goals << endl;
//		for (int i=0;i<p.prob_modes_goals.size();i++){
//			logd << p.prob_modes_goals[i] << " ";
//		}

		if (COORD::EuclideanDistance(p.pos, car.pos) < ModelParams::LASER_RANGE) {
			s.peds[s.num].pos = p.pos;
//			s.peds[s.num].goal = p.sample_goal();
			p.sample_goal_mode(s.peds[s.num].goal, s.peds[s.num].mode);
			s.peds[s.num].id = p.id;
            s.peds[s.num].vel = p.vel;
			s.num ++;
		}
    }

//	logd << "[WorldBeliefTracker::sample] done" << endl;

    return s;
}

vector<PomdpState> WorldBeliefTracker::sample(int num) {

	if(DESPOT::Debug_mode)
		Random::RANDOM.seed(0);

    vector<PomdpState> particles;
	logd << "[WorldBeliefTracker::sample] Sampling" << endl;

    for(int i=0; i<num; i++) {
        particles.push_back(sample());
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
	double car_x, car_y, car_yaw;
	car_yaw = car.heading_dir;

	car_x = car.pos.x - CAR_LENGTH/2.0f*cos(car_yaw);
	car_y = car.pos.y - CAR_LENGTH/2.0f*sin(car_yaw);

	/// for pomdp car
	double car_radius = sqrt(pow(CAR_WIDTH/2.0f, 2) + pow(CAR_LENGTH/2.0f,2)) + CAR_EXPAND_SIZE;
	ped_sim_[threadID]->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
	ped_sim_[threadID]->setAgentPrefVelocity(num_att_pes, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw)));

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
    double car_x, car_y, car_yaw;
    car_yaw = car.heading_dir;

    car_x = car.pos.x - CAR_LENGTH/2.0f*cos(car_yaw);
    car_y = car.pos.y - CAR_LENGTH/2.0f*sin(car_yaw);

    /// for pomdp car
    double car_radius = sqrt(pow(CAR_WIDTH/2.0f, 2) + pow(CAR_LENGTH/2.0f,2)) + CAR_EXPAND_SIZE;
    ped_sim_[threadID]->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
    ped_sim_[threadID]->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw)));

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
    double car_x, car_y, car_yaw;
    car_yaw = car.heading_dir;

	car_x = car.pos.x - CAR_LENGTH/2.0f*cos(car_yaw);
	car_y = car.pos.y - CAR_LENGTH/2.0f*sin(car_yaw);

    /// for pomdp car
	double car_radius = sqrt(pow(CAR_WIDTH/2.0f, 2) + pow(CAR_LENGTH/2.0f,2)) + CAR_EXPAND_SIZE;
	ped_sim_[threadID]->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
	ped_sim_[threadID]->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw)));

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
        double rNum=GenerateGaussian(random);
        peds[i].pos.x=ped_sim_[threadID]->getAgentPosition(i).x();// + rNum * (ped_sim_[threadID]->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        rNum=GenerateGaussian(rNum);
        peds[i].pos.y=ped_sim_[threadID]->getAgentPosition(i).y();// + rNum * (ped_sim_[threadID]->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
      
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
	double car_x, car_y, car_yaw;
	car_yaw = car.heading_dir;

	car_x = car.pos.x - CAR_LENGTH/2.0f*cos(car_yaw);
	car_y = car.pos.y - CAR_LENGTH/2.0f*sin(car_yaw);

	/// for pomdp car
	double car_radius = sqrt(pow(CAR_WIDTH/2.0f, 2) + pow(CAR_LENGTH/2.0f,2)) + CAR_EXPAND_SIZE;
	ped_sim_[threadID]->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 2, 1.0f, 2.0f, car_radius, ModelParams::VEL_MAX, RVO::Vector2(), "vehicle");
	ped_sim_[threadID]->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw)));

	// Set the preferred velocity for each agent.

	ped_sim_[threadID]->doPreStep();// build kd tree and find neighbors for peds

	for (size_t i = 0; i < num_ped; ++i) {
		// For each ego ped
	    for(int goal_id=0; goal_id<goals.size(); goal_id++) {
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
	    }
	}
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




