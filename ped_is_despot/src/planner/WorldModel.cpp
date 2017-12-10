#include <limits>
#include <cmath>
#include <cstdlib>
#include "WorldModel.h"
#include "math_utils.h"
#include "coord.h"
#include <iostream>
#include <fstream>
using namespace std;

WorldModel::WorldModel(): freq(ModelParams::control_freq),
    in_front_angle_cos(cos(ModelParams::IN_FRONT_ANGLE_DEG / 180.0 * M_PI)) {
    /*goals = {
        COORD(54, 4),  //green, bus stop
        COORD(31, 4),  //red    upper bus stop
        COORD(5,  5),  //blue   bridge to campus
        COORD(44,49),  //sky blue    after create
        COORD(18,62),  //yellow     residence area
        COORD(66,17),  //brown     garage
		COORD(-1,-1)   //pink     stop intention
    };*/
	/*
    goals = {
        COORD(107, 167),
        COORD(121, 169),
        COORD(125,  143),
        COORD(109,109),
        COORD(122,114)
    };
	*/

    // goals = {
    //     /*COORD(-200, 7.5),
    //     COORD(10, 150),*/
    //     COORD(-2, 7.5),
    //     COORD(10, 17),
    //     COORD(-1,-1) 
    // };


    goals = { // Unity airport departure
        COORD(-197.80, -134.80), // phora
        COORD(-180.15, -137.54), // Nana?
        COORD(-169.33, -141.1), // gate 1,2,3 
        COORD(-174.8, -148.53), // Cafe2
        COORD(-201.55, -148.53), //Cafe1
        COORD(-216.57, -145), // Gate 4,5,6
        COORD(-1, -1) // stop
    };

    ped_sim_ = new RVO::RVOSimulator();
    
    // Specify global time step of the simulation.
    ped_sim_->setTimeStep(0.33f);    

    // Specify default parameters for agents that are subsequently added.
    //ped_sim_->setAgentDefaults(5.0f, 8, 10.0f, 5.0f, 0.5f, 2.0f);
    //ped_sim_->setAgentDefaults(1.5f, 1, 3.0f, 6.0f, 0.15f, 3.0f);
    //ped_sim_->setAgentDefaults(3.0f, 2, 2.0f, 2.0f, 0.25f, 3.0f);
    ped_sim_->setAgentDefaults(5.0f, 3, 2.0f, 2.0f, 0.25f, 3.0f);
}

bool WorldModel::isLocalGoal(const PomdpState& state) {
    return state.car.dist_travelled > ModelParams::GOAL_TRAVELLED || state.car.pos >= path.size()-1;
}

bool WorldModel::isLocalGoal(const PomdpStateWorld& state) {
    return state.car.dist_travelled > ModelParams::GOAL_TRAVELLED || state.car.pos >= path.size()-1;
}

bool WorldModel::isGlobalGoal(const CarStruct& car) {
    double d = COORD::EuclideanDistance(path[car.pos], path[path.size()-1]);
    return (d<ModelParams::GOAL_TOLERANCE);
}

int WorldModel::defaultPolicy(const vector<State*>& particles)  {
	const PomdpState *state=static_cast<const PomdpState*>(particles[0]);
    double mindist = numeric_limits<double>::infinity();
    auto& carpos = path[state->car.pos];
    double carvel = state->car.vel;
    // Closest pedestrian in front
    for (int i=0; i<state->num; i++) {
		auto& p = state->peds[i];
		if(!inFront(p.pos, state->car.pos)) continue;
        double d = COORD::EuclideanDistance(carpos, p.pos);
		// cout << d << " " << carpos.x << " " << carpos.y << " "<< p.pos.x << " " << p.pos.y << endl;
        if (d >= 0 && d < mindist) 
			mindist = d;
    }

    if(state->scenario_id==0 && ModelParams::CPUDoPrint){
       std::ofstream fout;fout.open("/home/yuanfu/rollout.txt", std::ios::app);
	   fout << "min_dist = " << mindist << endl;
       fout.flush();
       fout.close();
    }
    // TODO set as a param
    if (mindist < 2) {
		return (carvel <= 0.01) ? 0 : 2;
    }

/*    if (mindist < 4) {
		if (carvel > 1.0) return 2;	
		else if (carvel < 0.5) return 1;
		else return 0;
    }
    return carvel >= ModelParams::VEL_MAX ? 0 : 1;*/
    if (mindist < 4/*5*/) {
        if (carvel > 1.0+1e-4) return 2;
        else if (carvel < 0.5-1e-4) return 1;
        else return 0;
    }
    return carvel >= ModelParams::VEL_MAX-1e-4 ? 0 : 1;
}

bool WorldModel::inFront(COORD ped_pos, int car) const {
    if(ModelParams::IN_FRONT_ANGLE_DEG >= 180.0) {
        // inFront check is disabled
        return true;
    }
	const COORD& car_pos = path[car];
	const COORD& forward_pos = path[path.forward(car, 1.0)];
	double d0 = COORD::EuclideanDistance(car_pos, ped_pos);
	//if(d0<=0) return true;
	if(d0 <= 0.7) return true;
	double d1 = COORD::EuclideanDistance(car_pos, forward_pos);
	if(d1<=0) return false;
	double dot = DotProduct(forward_pos.x - car_pos.x, forward_pos.y - car_pos.y,
			ped_pos.x - car_pos.x, ped_pos.y - car_pos.y);
	double cosa = dot / (d0 * d1);
	assert(cosa <= 1.0 + 1E-8 && cosa >= -1.0 - 1E-8);
    return cosa > in_front_angle_cos;
	//double angle = acos(cosa);
	//return (fabs(angle) < M_PI / 180 * 60);
}

/**
 * H: center of the head of the car
 * N: a point right in front of the car
 * M: an arbitrary point
 *
 * Check whether M is in the safety zone
 */
bool inCollision(double Mx, double My, double Hx, double Hy, double Nx, double Ny);
bool inCollision(double ped_x, double ped_y, double car_x, double car_y);//for scooter collision check;

bool WorldModel::inCollision(const PomdpState& state) {
    const int car = state.car.pos;
	const COORD& car_pos = path[car];
	const COORD& forward_pos = path[path.forward(car, 1.0)];

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, forward_pos.x, forward_pos.y)) {
            return true;
        }
        /*if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y)) {
            return true;
        }*/
    }
    return false;
}

bool WorldModel::inCollision(const PomdpStateWorld& state) {
    const int car = state.car.pos;
    const COORD& car_pos = path[car];
    const COORD& forward_pos = path[path.forward(car, 1.0)];

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, forward_pos.x, forward_pos.y)) {
            return true;
        }
        /*if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y)) {
            return true;
        }*/
    }
    return false;
}

bool WorldModel::inCollision(const PomdpState& state, int &id) {
	id=-1;
    const int car = state.car.pos;
	const COORD& car_pos = path[car];
	const COORD& forward_pos = path[path.forward(car, 1.0)];

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, forward_pos.x, forward_pos.y)) {
        	id=state.peds[i].id;
            return true;
        }
        /*if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y)) {
            id=state.peds[i].id;
            return true;
        }*/
    }
    return false;
}

bool WorldModel::inCollision(const PomdpStateWorld& state, int &id) {
    id=-1;
    const int car = state.car.pos;
    const COORD& car_pos = path[car];
    const COORD& forward_pos = path[path.forward(car, 1.0)];

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;
        if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y, forward_pos.x, forward_pos.y)) {
            id=state.peds[i].id;
            return true;
        }
        /*if(::inCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y)) {
            id=state.peds[i].id;
            return true;
        }*/
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
    const auto& carpos = path[state.car.pos];

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.peds[i];
		bool front = inFront(p.pos, state.car.pos);
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
    const auto& carpos = path[state.car.pos];
	const auto& nextcarpos = path[path.forward(state.car.pos, 1.0)];

	const auto& pedpos = state.peds[ped].pos;
	const auto& goalpos = goals[state.peds[ped].goal];

	if (goalpos.x == -1 && goalpos.y == -1)
		return false;

	return DotProduct(goalpos.x - pedpos.x, goalpos.y - pedpos.y,
			nextcarpos.x - carpos.x, nextcarpos.y - carpos.y) > 0;
}

///get the min distance between car and the peds in its front
double WorldModel::getMinCarPedDist(const PomdpState& state) {
    double mindist = numeric_limits<double>::infinity();
    const auto& carpos = path[state.car.pos];

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.peds[i];
		if(!inFront(p.pos, state.car.pos)) continue;
        double d = COORD::EuclideanDistance(carpos, p.pos);
        if (d >= 0 && d < mindist) mindist = d;
    }

	return mindist;
}

///get the min distance between car and the peds
double WorldModel::getMinCarPedDistAllDirs(const PomdpState& state) {
    double mindist = numeric_limits<double>::infinity();
    const auto& carpos = path[state.car.pos];

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.peds[i];
        double d = COORD::EuclideanDistance(carpos, p.pos);
        if (d >= 0 && d < mindist) mindist = d;
    }

	return mindist;
}

int WorldModel::minStepToGoal(const PomdpState& state) {
    double d = ModelParams::GOAL_TRAVELLED - state.car.dist_travelled;
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

double gaussian_prob(double x, double stddev) {
    double a = 1.0 / stddev / sqrt(2 * M_PI);
    double b = - x * x / 2.0 / (stddev * stddev);
    return a * exp(b);
}

double WorldModel::ISPedStep(CarStruct &car, PedStruct &ped, Random& random) {
//gaussian + distance
    double max_is_angle = 7.0*M_PI/64.0;
    COORD carpos = path[car.pos];
    if(COORD::EuclideanDistance(ped.pos, carpos)>3.5){
        const COORD& goal = goals[ped.goal];
        if (goal.x == -1 && goal.y == -1) {  //stop intention
            return 1;
        }

        MyVector goal_vec(goal.x - ped.pos.x, goal.y - ped.pos.y);
        double a = goal_vec.GetAngle();
        double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
        a += noise;

        //TODO noisy speed
        MyVector move(a, ped.vel/freq, 0);
        ped.pos.x += move.dw;
        ped.pos.y += move.dh;
        return 1;
    } else{
        double weight = 1.0;
        const COORD& goal = goals[ped.goal];
        if (goal.x == -1 && goal.y == -1) {  //stop intention
            return weight;
        }

        MyVector goal_vec(goal.x - ped.pos.x, goal.y - ped.pos.y);
        double goal_angle = goal_vec.GetAngle();

        //compute the angle to robot
        MyVector rob_vec(path[car.pos].x - ped.pos.x, path[car.pos].y - ped.pos.y);
        double rob_angle = rob_vec.GetAngle();

        double final_mean; //final mean angle

        if(abs(goal_angle-rob_angle) <= M_PI){
            if(goal_angle > rob_angle) final_mean = goal_angle - min(max_is_angle, goal_angle-rob_angle);
            else  final_mean = goal_angle + min(max_is_angle, rob_angle-goal_angle);
        }
        else{
            if(goal_angle > rob_angle) final_mean = goal_angle + min(max_is_angle, rob_angle+2*M_PI-goal_angle);
            else  final_mean = goal_angle - min(max_is_angle, goal_angle+2*M_PI-rob_angle);
        }

        if(final_mean>M_PI) final_mean -= M_PI;
        else if(final_mean<-M_PI) final_mean += M_PI;

        //random.NextGaussian() returns a random number sampled from N(0,1)
        double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE; //change to the number sampled from N(0, ModelParams::NOISE_GOAL_ANGLE)
        double final_angle = final_mean + noise; //change to the number sampled from N(rob_angle, ModelParams::NOISE_GOAL_ANGLE)

        //TODO noisy speed
        MyVector move(final_angle, ped.vel/freq, 0);
        ped.pos.x += move.dw;
        ped.pos.y += move.dh;

        weight = gaussian_prob((final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE, 1) /
                 gaussian_prob((final_angle - final_mean) / ModelParams::NOISE_GOAL_ANGLE, 1) ;
        return weight;
    }
//gaussian change the mean to the angle near goal
/*    double weight = 1.0;
    const COORD& goal = goals[ped.goal];
    if (goal.x == -1 && goal.y == -1) {  //stop intention
        return weight;
    }

    MyVector goal_vec(goal.x - ped.pos.x, goal.y - ped.pos.y);
    double goal_angle = goal_vec.GetAngle();

    //compute the angle to robot
    MyVector rob_vec(path[car.pos].x - ped.pos.x, path[car.pos].y - ped.pos.y);
    double rob_angle = rob_vec.GetAngle();

    double final_mean; //final mean angle

    if(abs(goal_angle-rob_angle) <= M_PI){
        if(goal_angle > rob_angle) final_mean = goal_angle - min(M_PI/8.0, goal_angle-rob_angle);
        else  final_mean = goal_angle + min(M_PI/8.0, rob_angle-goal_angle);
    }
    else{
        if(goal_angle > rob_angle) final_mean = goal_angle + min(M_PI/8.0, rob_angle+2*M_PI-goal_angle);
        else  final_mean = goal_angle - min(M_PI/8.0, goal_angle+2*M_PI-rob_angle);
    }

    if(final_mean>M_PI) final_mean -= M_PI;
    else if(final_mean<-M_PI) final_mean += M_PI;

    //random.NextGaussian() returns a random number sampled from N(0,1)
    double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE; //change to the number sampled from N(0, ModelParams::NOISE_GOAL_ANGLE)
    double final_angle = final_mean + noise; //change to the number sampled from N(rob_angle, ModelParams::NOISE_GOAL_ANGLE)

    //TODO noisy speed
    MyVector move(final_angle, ped.vel/freq, 0);
    ped.pos.x += move.dw;
    ped.pos.y += move.dh;

    weight = gaussian_prob((final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE, 1) /
             gaussian_prob((final_angle - final_mean) / ModelParams::NOISE_GOAL_ANGLE, 1) ;
    return weight;*/

//gaussian change the mean to the angle towards rob
    // double weight = 1.0;
    // const COORD& goal = goals[ped.goal];
    // if (goal.x == -1 && goal.y == -1) {  //stop intention
    //     return weight;
    // }

    // MyVector goal_vec(goal.x - ped.pos.x, goal.y - ped.pos.y);
    // double goal_angle = goal_vec.GetAngle();

    // //compute the angle to robot
    // MyVector rob_vec(path[car.pos].x - ped.pos.x, path[car.pos].y - ped.pos.y);
    // double rob_angle = rob_vec.GetAngle();

    // //random.NextGaussian() returns a random number sampled from N(0,1)
    // double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE; //change to the number sampled from N(0, ModelParams::NOISE_GOAL_ANGLE)
    // double final_angle = rob_angle + noise; //change to the number sampled from N(rob_angle, ModelParams::NOISE_GOAL_ANGLE)

    // //TODO noisy speed
    // MyVector move(final_angle, ped.vel/freq, 0);
    // ped.pos.x += move.dw;
    // ped.pos.y += move.dh;

    // /*
    // let sigma = ModelParams::NOISE_GOAL_ANGLE,  x is the ped's angle (not the angle it turns, but its final angle),
    // q(x) denote the importance distribution,
    // qn(x) denote the standard normal distribution, 
    // p(x) denote the original distribution, 

    // q(x) = pn((x - rob_angle) / sigma)
    // p(x) = pn((x - goal_angle) / sigma)

    // here, final_angle is the final actual angle
    // */
    // weight = gaussian_prob((final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE, 1) /
    //          gaussian_prob((final_angle - rob_angle) / ModelParams::NOISE_GOAL_ANGLE, 1) ;
    // return weight;
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

        if((move_dist * goal_dist) < 1e-5) {
            std::cout<<"move dist: "<<move_dist<<"  goal_dist: "<<goal_dist<<std::endl;
        }
		double cosa = DotProduct(curr.x-prev.x, curr.y-prev.y, goal.x-prev.x, goal.y-prev.y) / (move_dist * goal_dist);
        if(cosa >1) cosa = 1;
        else if(cosa < -1) cosa = -1;
		double angle = acos(cosa);
		return gaussian_prob(angle, ModelParams::NOISE_GOAL_ANGLE) + K;
	}
} 

void WorldModel::RobStep(CarStruct &car, Random& random) {
    double dist = car.vel / freq;
    //double dist_l=max(0.0,dist-ModelParams::AccSpeed/freq);
    //double dist_r=min(ModelParams::VEL_MAX,dist+ModelParams::AccSpeed/freq);
    //double sample_dist=random.NextDouble(dist_l,dist_r);
    //int nxt = path.forward(car.pos, sample_dist);
    int nxt = path.forward(car.pos, dist);
    car.pos = nxt;
    car.dist_travelled += dist;
}

void WorldModel::RobStep(CarStruct &car, Random& random, double acc) {
    double end_vel = car.vel + acc / freq;
    end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);
    double dist = (car.vel + end_vel)/2.0 / freq;
    //double dist_l=max(0.0,dist-ModelParams::AccSpeed/freq);
    //double dist_r=min(ModelParams::VEL_MAX,dist+ModelParams::AccSpeed/freq);
    //double sample_dist=random.NextDouble(dist_l,dist_r);
    //int nxt = path.forward(car.pos, sample_dist);
    int nxt = path.forward(car.pos, dist);
    car.pos = nxt;
    car.dist_travelled += dist;
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
	/*
	for(int i=0; i<this->path.size(); i++) {
		const auto& p = this->path[i];
		cout << p.x << " " << p.y << endl;
	}*/
}

void WorldModel::updatePedBelief(PedBelief& b, const PedStruct& curr_ped) {
    const double ALPHA = 0.8;
	const double SMOOTHING=ModelParams::BELIEF_SMOOTHING;

    bool debug=false;

/*    cout<<"before update: "<<curr_ped.id<<endl;
    for(double w: b.prob_goals) {
        cout << w << " ";
    }
    cout << endl;*/
	
    for(int i=0; i<goals.size(); i++) {
		double prob = pedMoveProb(b.pos, curr_ped.pos, i);
		if(debug) cout << "likelihood " << i << ": " << prob << endl;
        b.prob_goals[i] *=  prob;

		// Important: Keep the belief noisy to avoid aggressive policies
		b.prob_goals[i] += SMOOTHING / goals.size(); // CHECK: decrease or increase noise
	}
	if(debug) {
        for(double w: b.prob_goals) {
            cout << w << " ";
        }
        cout << endl;
    }

    // normalize
    double total_weight = accumulate(b.prob_goals.begin(), b.prob_goals.end(), double(0.0));
	if(debug) cout << "total_weight = " << total_weight << endl;
    for(double& w : b.prob_goals) {
        w /= total_weight;
    }

/*    cout<<"after update: "<<curr_ped.id<<endl;
	for(double w: b.prob_goals) {
		cout << w << " ";
	}
	cout << endl;*/

    double moved_dist = COORD::EuclideanDistance(b.pos, curr_ped.pos);
    b.vel = ALPHA * b.vel + (1-ALPHA) * moved_dist * ModelParams::control_freq;
	b.pos = curr_ped.pos;
}

PedBelief WorldModel::initPedBelief(const PedStruct& ped) {
    PedBelief b = {ped.id, ped.pos, ModelParams::PED_SPEED, vector<double>(goals.size(), 1.0/goals.size())};
    return b;
}

double timestamp() {
    //return ((double) clock()) / CLOCKS_PER_SEC;
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

void WorldStateTracker::updatePed(const Pedestrian& ped){
    int i=0;
    for(;i<ped_list.size();i++) {
        if (ped_list[i].id==ped.id) {
            //found the corresponding ped,update the pose
            //double dw = ped.w - ped_list[i].w;
            //double dh = ped.h - ped_list[i].h;
            //double dist = sqrt(dw*dw + dh*dh);
            ped_list[i].w=ped.w;
            ped_list[i].h=ped.h;
            //ped_list[i].vel = ped_list[i].vel * ALPHA + dist * ModelParams::control_freq * (1-ALPHA);
            ped_list[i].last_update = timestamp();
            break;
        }
        if (abs(ped_list[i].w-ped.w)<=0.1 && abs(ped_list[i].h-ped.h)<=0.1)   //overlap
            return;
    }
    if (i==ped_list.size()) {
        //not found, new ped
        ped_list.push_back(ped);
        ped_list.back().last_update = timestamp();
        
    }
}

void WorldStateTracker::updateCar(const COORD& car) {
    carpos=car;
}

bool WorldStateTracker::emergency() {
    //TODO improve emergency stop to prevent the false detection of leg
    double mindist = numeric_limits<double>::infinity();
    for(auto& ped : ped_list) {
		COORD p(ped.w, ped.h);
        double d = COORD::EuclideanDistance(carpos, p);
        if (d < mindist) mindist = d;
    }
	//cout << "    emergency mindist = " << mindist << endl;
	return (mindist < 0.5);
}

void WorldStateTracker::updateVel(double vel) {
	/*
	if (vel>ModelParams::VEL_MAX)
		carvel=ModelParams::VEL_MAX;
	else carvel = vel;
	*/
	carvel = vel;
}

vector<WorldStateTracker::PedDistPair> WorldStateTracker::getSortedPeds() {
    // cout << "before sorting:" << endl;
    // sort peds
    vector<PedDistPair> sorted_peds;
    for(const auto& p: ped_list) {
        COORD cp(p.w, p.h);
        float dist = COORD::EuclideanDistance(cp, carpos);
        sorted_peds.push_back(PedDistPair(dist, p));
        // cout << " " << dist;
    }
    // cout << endl;
    sort(sorted_peds.begin(), sorted_peds.end(),
            [](const PedDistPair& a, const PedDistPair& b) -> bool {
                return a.first < b.first;
            });

/*    cout<<"debug: MMMMMMMMMMMMMMMM"<<endl;
    cout << "after sorting:" << endl;
    for(const auto& p : sorted_peds) {
        cout << " " << p.first;
    }
    cout << endl;*/

    return sorted_peds;
}

PomdpState WorldStateTracker::getPomdpState() {
    auto sorted_peds = getSortedPeds();

    // construct PomdpState
    PomdpState pomdpState;
    pomdpState.car.pos = model.path.nearest(carpos);
    pomdpState.car.vel = carvel;
	pomdpState.car.dist_travelled = 0;
    pomdpState.num = sorted_peds.size();

	if (pomdpState.num > ModelParams::N_PED_IN) {
		pomdpState.num = ModelParams::N_PED_IN;
	}

    //cout<<"pedestrian time stamps"<<endl;
    for(int i=0;i<pomdpState.num;i++) {
        const auto& ped = sorted_peds[i].second;
        pomdpState.peds[i].pos.x=ped.w;
        pomdpState.peds[i].pos.y=ped.h;
		pomdpState.peds[i].id = ped.id;
        //pomdpState.peds[i].vel = ped.vel;
		pomdpState.peds[i].goal = -1;
    //    cout<<"ped "<<i<<" "<<ped.last_update<<endl;
    }
	return pomdpState;
}

void WorldBeliefTracker::update() {
    // update car
    car.pos = model.path.nearest(stateTracker.carpos);

/*    std::ofstream fout; fout.open("/home/yuanfu/updatedplan", std::ios::trunc);
    fout<< "car coord: " << stateTracker.carpos.x <<" "<<stateTracker.carpos.y <<endl;
    for(int i=0;i<model.path.size();i++){
        fout<< "path node "<<i<<": "<< model.path[i].x <<" "<<model.path[i].y <<endl;
    }
    fout<<"End path"<<endl;
    fout.close();*/
    car.vel = stateTracker.carvel;
	car.dist_travelled = 0;

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
        }
    }
    for(const auto& i: peds_to_remove) {
        peds.erase(i);
    }

/*    cout<<"debug: MMMMMMMMMMMMMMMM"<<endl;
    cout << "after removing:" << endl;
    for(int i=0; i<peds.size();i++) {
        cout << " " << peds.at(i).id;
    }
    cout << endl;*/

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

/*    cout<<"debug: MMMMMMMMMMMMMMMM"<<endl;
    cout << "after removing:" << endl;
    for(int i=0; i<peds.size();i++) {
        cout << " " << peds.at(i).id;
    }
    cout << endl;*/

    return;
}

int PedBelief::sample_goal() const {
    double r = double(rand()) / RAND_MAX;
    int i = 0;
    r -= prob_goals[i];
    while(r > 0) {
        i++;
        r -= prob_goals[i];
    }
    return i;
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
    return;
	int num = 0;
    for(int i=0; i < sorted_beliefs.size() && i < ModelParams::N_PED_IN; i++) {
		auto& p = sorted_beliefs[i];
		if (COORD::EuclideanDistance(p.pos, model.path[car.pos]) < 10) {
            cout << "ped belief " << num << ": ";
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
		if (COORD::EuclideanDistance(p.pos, model.path[car.pos]) < 10) {
			s.peds[s.num].pos = p.pos;
			s.peds[s.num].goal = p.sample_goal();
			s.peds[s.num].id = p.id;
            s.peds[s.num].vel = p.vel;
            //s.peds[s.num].vel = (p.vel < 0.1? 0.1:p.vel);
            //s.peds[s.num].vel = 1.0;
			s.num ++;
		}
    }
    //cout<<"print state from planner "<<endl;
    //PrintState(s,cout);
    return s;
}

vector<PomdpState> WorldBeliefTracker::sample(int num) {
    vector<PomdpState> particles;
    for(int i=0; i<num; i++) {
        particles.push_back(sample());
    }

    //cout << "Num peds for planning: " << particles[0].num << endl;

    return particles;
}

vector<PedStruct> WorldBeliefTracker::predictPeds() {
    vector<PedStruct> prediction;

    for(const auto& p: sorted_beliefs) {
        double dist = COORD::EuclideanDistance(p.pos, model.path[car.pos]);
        //cout<<"predicPeds: "<<p.vel<<" "<<car.vel<<endl;
/*        double relative_speed = ((p.vel + car.vel < 0.2) ? 0.2 : (p.vel + car.vel));
        int step = int(dist / relative_speed * ModelParams::control_freq);*/

        int step = (p.vel + car.vel>1e-5)?int(dist / (p.vel + car.vel) * ModelParams::control_freq):100000;


        //for(int j=0; j<10; j++) {
            //int goal = p.sample_goal();
        for(int j=0; j<1; j++) {
            int goal = p.maxlikely_goal();
            PedStruct ped0(p.pos, goal, p.id);
			ped0.vel = p.vel;
            //for(int i=0; i<6; i++) {
            for(int i=0; i<ModelParams::N_PED_IN; i++) {
                PedStruct ped = ped0;
                model.PedStepDeterministic(ped, step+i*2);
                prediction.push_back(ped);
            }
        }
    }
    return prediction;
}

void WorldBeliefTracker::PrintState(const State& s, ostream& out) const {
	const PomdpState & state=static_cast<const PomdpState&> (s);
    COORD& carpos = model.path[state.car.pos];

	out << "Rob Pos: " << carpos.x<< " " <<carpos.y << endl;
	out << "Rob travelled: " << state.car.dist_travelled << endl;
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

void WorldModel::RVO2PedStep(PedStruct peds[], Random& random, int num_ped){

    ped_sim_->clearAllAgents();

    for(int i=0; i<num_ped; i++){
        ped_sim_->addAgent(RVO::Vector2(peds[i].pos.x, peds[i].pos.y));
    }

    // Set the preferred velocity for each agent.
    for (size_t i = 0; i < ped_sim_->getNumAgents(); ++i) {
        int goal_id = peds[i].goal;
        if (goal_id >= goals.size()-1) { /// stop intention
            ped_sim_->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
        } else{
            RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
            if ( absSq(goal - ped_sim_->getAgentPosition(i)) < ped_sim_->getAgentRadius(i) * ped_sim_->getAgentRadius(i) ) {
                // Agent is within one radius of its goal, set preferred velocity to zero
                ped_sim_->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
            } else {
                // Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
                ped_sim_->setAgentPrefVelocity(i, normalize(goal - ped_sim_->getAgentPosition(i)));
            }
        }
        
    }

    ped_sim_->doStep();

    for(int i=0; i<num_ped; i++){
        peds[i].pos.x=ped_sim_->getAgentPosition(i).x() + random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        peds[i].pos.y=ped_sim_->getAgentPosition(i).y() + random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
    }
}

void WorldModel::RVO2PedStep(PedStruct peds[], Random& random, int num_ped, CarStruct car){

    ped_sim_->clearAllAgents();

    //adding pedestrians
    for(int i=0; i<num_ped; i++){
        ped_sim_->addAgent(RVO::Vector2(peds[i].pos.x, peds[i].pos.y));
    }

    // adding car as a "special" pedestrian
    double car_x, car_y, car_yaw;
    car_x = path[car.pos].x;
    car_y = path[car.pos].y;
    car_yaw = path.getYaw(car.pos);
  //  ped_sim_->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 1, 3.0f, 5.0f, 0.8f, 3.0f);
  //  ped_sim_->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the num_ped-th pedestrian is the car. set its prefered velocity
    ped_sim_->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 2, 1.0f, 2.0f, 1.0f, 3.0f, RVO::Vector2(), "vehicle");
    ped_sim_->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw)));

    // Set the preferred velocity for each agent.
    for (size_t i = 0; i < num_ped; ++i) {
        int goal_id = peds[i].goal;
        if (goal_id >= goals.size()-1) { /// stop intention
            ped_sim_->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
        } else{
            RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
            if ( absSq(goal - ped_sim_->getAgentPosition(i)) < ped_sim_->getAgentRadius(i) * ped_sim_->getAgentRadius(i) ) {
                // Agent is within one radius of its goal, set preferred velocity to zero
                ped_sim_->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
                //ped_sim_->setAgentPrefVelocity(i, normalize(goal - ped_sim_->getAgentPosition(i))*0.6);
            } else {
                // Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
                //ped_sim_->setAgentPrefVelocity(i, normalize(goal - ped_sim_->getAgentPosition(i)));
                ped_sim_->setAgentPrefVelocity(i, normalize(goal - ped_sim_->getAgentPosition(i))*1.2);
            }
        }
        
    }

    ped_sim_->doStep();

    for(int i=0; i<num_ped; i++){
        //peds[i].pos.x=ped_sim_->getAgentPosition(i).x() + random.NextGaussian() * (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        //peds[i].pos.y=ped_sim_->getAgentPosition(i).y() + random.NextGaussian() * (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
/*        float vel = freq*sqrt((ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)*(ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)
                +(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y)*(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y));
        if (vel < 0.2 && vel > 0){
            peds[i].pos.x=ped_sim_->getAgentPosition(i).x() + (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)*(1/vel); //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
            peds[i].pos.y=ped_sim_->getAgentPosition(i).y() + (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)*(1/vel);//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        }
        else{
            peds[i].pos.x=ped_sim_->getAgentPosition(i).x();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
            peds[i].pos.y=ped_sim_->getAgentPosition(i).y();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        } */

        peds[i].pos.x=ped_sim_->getAgentPosition(i).x();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
        peds[i].pos.y=ped_sim_->getAgentPosition(i).y();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
      
    }

/*    ped_sim_->setNotUpdated();

    //adding pedestrians
    for(int i=0; i<num_ped; i++){

        int goal_id = peds[i].goal;
        RVO::Vector2 pos = RVO::Vector2(peds[i].pos.x, peds[i].pos.y);
        RVO::Vector2 pref_vel;

        if (goal_id >= goals.size()-1) { /// stop intention
            pref_vel = RVO::Vector2(0.0f, 0.0f);
        }
        else{
            RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
            if ( absSq(goal - pos) < 0.0625 ) {
                pref_vel = normalize(goal - pos)*0.6;
            } else {
                pref_vel = normalize(goal - pos)*1.2;
            }
        }

        ped_sim_->updateAgent(peds[i].id, pos, pref_vel);
    }

    ped_sim_->deleteOldAgents();

    double car_x, car_y, car_yaw;
    car_x = path[car.pos].x;
    car_y = path[car.pos].y;
    car_yaw = path.getYaw(car.pos);

    //addAgent (const Vector2 &position, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed)
    ped_sim_->addAgent(RVO::Vector2(car_x, car_y), 3.0f, 2, 1.0f, 2.0f, 1.0f, 3.0f, RVO::Vector2(), "vehicle");
    ped_sim_->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the num_ped-th pedestrian is the car. set its prefered velocity 
    ped_sim_->setAgentPedID(num_ped,-1);
    
    ped_sim_->doStep();

    for(int i=0; i<num_ped; i++){
        peds[i].vel = freq*sqrt((ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)*(ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)
            +(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y)*(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y));
            peds[i].pos.x=ped_sim_->getAgentPosition(i).x();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
            peds[i].pos.y=ped_sim_->getAgentPosition(i).y();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

    }*/
}

/*void WorldModel::RVO2PedStep(PedStruct peds[], Random& random, int num_ped, CarStruct car){

    ped_sim_->clearAllAgents();

    //adding pedestrians
    for(int i=0; i<num_ped; i++){
        ped_sim_->addAgent(RVO::Vector2(peds[i].pos.x, peds[i].pos.y));
    }

    // adding car as a "special" pedestrian
    double car_x, car_y, car_yaw;
    car_x = path[car.pos].x;
    car_y = path[car.pos].y;
    car_yaw = path.getYaw(car.pos);
    ped_sim_->addAgent(RVO::Vector2(car_x, car_y), 5.0f, 8, 10.0f, 5.0f, 0.7f, 1.2f);
    ped_sim_->setAgentPrefVelocity(num_ped, RVO::Vector2(car.vel * cos(car_yaw), car.vel * sin(car_yaw))); // the num_ped-th pedestrian is the car. set its prefered velocity

    // Set the preferred velocity for each agent.
    for (size_t i = 0; i < num_ped; ++i) {
        int goal_id = peds[i].goal;
        if (goal_id >= goals.size()-1) { /// stop intention
            ped_sim_->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
        } else{
            RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
                if ( absSq(goal - ped_sim_->getAgentPosition(i)) < ped_sim_->getAgentRadius(i) * ped_sim_->getAgentRadius(i) ) {
                    // Agent is within one radius of its goal, set preferred velocity to zero
                    //ped_sim_->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
                    ped_sim_->setAgentPrefVelocity(i, normalize(goal - ped_sim_->getAgentPosition(i)));
                } else {
                    // Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
                    ped_sim_->setAgentPrefVelocity(i, normalize(goal - ped_sim_->getAgentPosition(i))*2);
                }
        }
        
    }

    ped_sim_->doStep();

    for(int i=0; i<num_ped; i++){
        peds[i].vel = freq*sqrt((ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)*(ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)
                +(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y)*(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y));
            if (peds[i].vel < 0.2 && peds[i].vel > 0){
                peds[i].pos.x=ped_sim_->getAgentPosition(i).x() + (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)*(1/peds[i].vel); //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
                peds[i].pos.y=ped_sim_->getAgentPosition(i).y() + (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)*(1/peds[i].vel);//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
                peds[i].vel = freq*sqrt((ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)*(ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)
                    +(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y)*(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y));
            }
            // if(peds[i].pos.x==ped_sim_->getAgentPosition(i).x()){
            //     std::cout<<"ped not changed!!!!!!"<<std::endl;
            // }
            // else{
            //     std::cout<<"changed$$$$$$"<<std::endl;
            // }
            //std::cout<<peds[i].id<<"  "<<peds[i].pos.x<<"  "<<ped_sim_->getAgentPosition(i).x();
            // if(peds[i].id == 0){
            //     std::cout<<peds[i].id<<"  "<<peds[i].pos.x<<"  "<<ped_sim_->getAgentPosition(i).x()<<"  "<<peds[i].vel<<std::endl;
            //     std::cout<<"  "<<peds[i].pos.y<<"  "<<ped_sim_->getAgentPosition(i).y()<<"  "<<peds[i].goal<<std::endl<<std::endl;

            // }
            else{
                peds[i].pos.x=ped_sim_->getAgentPosition(i).x();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
                peds[i].pos.y=ped_sim_->getAgentPosition(i).y();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
            }     
    }
}*/