#include "WorldModel.h"
#include"state.h"
#include"Path.h"
#include "despot/solver/despot.h"
#include "custom_particle_belief.h"
#include "pomdp_simulator.h"

#include "ped_pomdp.h"
#include <ped_is_despot/car_info.h>
#include <ped_is_despot/peds_info.h>
#include <ped_is_despot/ped_info.h>
#include <ped_is_despot/peds_believes.h>

#include <iostream>
#include <fstream>
using namespace std;
using namespace despot;
//#define LINE_CASE
#define CROSS_CASE



int n_sim = 1;

#ifdef LINE_CASE
	const double PED_X0 = 35;/// used to generate peds' locations, where x is in (PED_X0, PED_X1), and y is in (PED_Y0, PED_Y1)
	const double PED_Y0 = 35;
	const double PED_X1 = 42;
	const double PED_Y1 = 52;
	const int n_peds = 6; // should be smaller than ModelParams::N_PED_IN
#elif defined(CROSS_CASE)
	const double PED_X0 = 0;/// used to generate peds' locations, where x is in (PED_X0, PED_X1), and y is in (PED_Y0, PED_Y1)
	const double PED_Y0 = 0;
	const double PED_X1 = 20;
	const double PED_Y1 = 15;
	const int n_peds = 16;//6; // should be smaller than ModelParams::N_PED_IN
#endif


/*POMDPSimulator::POMDPSimulator(DSPOMDP* model, unsigned seed)/// set the path to be a straight line
: POMDPWorld(model, seed)
{
#ifdef LINE_CASE
	start.x=40;start.y=40;
	goal.x=40;goal.y=50;
#elif defined(CROSS_CASE)
	start.x=20;start.y=7.5;
	goal.x=0;goal.y=7.5;
#endif

	Path p;
	p.push_back(start);
	p.push_back(goal);
	path = p.interpolate();
	worldModel.setPath(path);
	num_of_peds_world=0;
	stateTracker=new WorldStateTracker(worldModel);

}*/


POMDPSimulator::POMDPSimulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed, std::string obstacle_file_name): 
	SimulatorBase(_nh, obstacle_file_name), 
	POMDPWorld(model, seed){

	cerr << "DEBUG: Creating POMDPSimulator" << endl;

		
	global_frame_id = ModelParams::rosns + "/map";
	ros::NodeHandle n("~");
    n.param<std::string>("goal_file_name", worldModel.goal_file_name_, "null");

	#ifdef LINE_CASE
	start.x=40;start.y=40;
	goal.x=40;goal.y=50;
#elif defined(CROSS_CASE)
	start.x=20;start.y=7.5;
	goal.x=0;goal.y=7.5;
#endif

	Path p;
	p.push_back(start);
	p.push_back(goal);
	path = p.interpolate();
	worldModel.setPath(path);
	num_of_peds_world=0;

	worldModel.InitPedGoals();
	worldModel.InitRVO();
    AddObstacle();

	stateTracker=new WorldStateTracker(worldModel);

}
void POMDPSimulator::AddObstacle(){

	if(obstacle_file_name_ == "null"){
		/// for indian_cross
		std::vector<RVO::Vector2> obstacle[4];

	    obstacle[0].push_back(RVO::Vector2(-4,-4));
	    obstacle[0].push_back(RVO::Vector2(-30,-4));
	    obstacle[0].push_back(RVO::Vector2(-30,-30));
	    obstacle[0].push_back(RVO::Vector2(-4,-30));

	    obstacle[1].push_back(RVO::Vector2(4,-4));
	    obstacle[1].push_back(RVO::Vector2(4,-30));
	    obstacle[1].push_back(RVO::Vector2(30,-30));
	    obstacle[1].push_back(RVO::Vector2(30,-4));

		obstacle[2].push_back(RVO::Vector2(4,4));
	    obstacle[2].push_back(RVO::Vector2(30,4));
	    obstacle[2].push_back(RVO::Vector2(30,30));
	    obstacle[2].push_back(RVO::Vector2(4,30));

		obstacle[3].push_back(RVO::Vector2(-4,4));
	    obstacle[3].push_back(RVO::Vector2(-4,30));
	    obstacle[3].push_back(RVO::Vector2(-30,30));
	    obstacle[3].push_back(RVO::Vector2(-30,4));


	    int NumThreads=1/*Globals::config.NUM_THREADS*/;

	    for(int tid=0; tid<NumThreads;tid++){
		    for (int i=0; i<4; i++){
		 	   worldModel.ped_sim_[tid]->addObstacle(obstacle[i]);
			}

		    /* Process the obstacles so that they are accounted for in the simulation. */
		    worldModel.ped_sim_[tid]->processObstacles();
		}
	} else{

		int NumThreads=1/*Globals::config.NUM_THREADS*/;

		ifstream file;
	    file.open(obstacle_file_name_, std::ifstream::in);

	    if(file.fail()){
	        cout<<"open obstacle file failed !!!!!!"<<endl;
	        return;
	    }

	    std::vector<RVO::Vector2> obstacles[100];

	    std::string line;
	    int obst_num = 0;
	    while (std::getline(file, line))
	    {
	        std::istringstream iss(line);
	        
	        double x;
	        double y;
	        while (iss >> x >>y){
	            cout << x <<" "<< y<<endl;
	            obstacles[obst_num].push_back(RVO::Vector2(x, y));
	        }

	        for(int tid=0; tid<NumThreads;tid++){
			 	worldModel.ped_sim_[tid]->addObstacle(obstacles[obst_num]);			
			}

	        obst_num++;
	        if(obst_num > 99) break;
	    }

	    for(int tid=0; tid<NumThreads;tid++){
			worldModel.ped_sim_[tid]->processObstacles();			    
		}
	    
	    file.close();
	}

}

POMDPSimulator::~POMDPSimulator()
{

}

State* POMDPSimulator::Initialize(){
	cerr << "DEBUG: Initializing POMDPSimulator" << endl;

	// for tracking world state
	world_state.car.pos = start;
	world_state.car.vel = 0;
	world_state.car.heading_dir = -M_PI;
	world_state.num = n_peds;

	for(int i=0; i<n_peds; i++) {
		world_state.peds[i] = randomPed();
		world_state.peds[i].id = i;
	}

	num_of_peds_world = n_peds;

	double total_reward_dis = 0, total_reward_nondis=0;
	int step = 0;
	cout<<"LASER_RANGE= "<<ModelParams::LASER_RANGE<<endl;

	stateTracker->updateCar(world_state.car);

	//update the peds in stateTracker
	for(int i=0; i<num_of_peds_world; i++) {
		Pedestrian p(world_state.peds[i].pos.x, world_state.peds[i].pos.y, world_state.peds[i].id);
		stateTracker->updatePed(p);
	}

	target_speed_=0.0;
	real_speed_ = 0.0;
	steering_ = 0.0;

	b_update_il = true;

	cerr << "DEBUG: Initializing POMDPSimulator end" << endl;

}

bool POMDPSimulator::Connect(){
	cerr << "DEBUG: Connecting POMDPSimulator" << endl;

	IL_pub = nh.advertise<ped_is_despot::imitation_data>("il_data", 1);

	return true;
}

State* POMDPSimulator::GetCurrentState() const{

	cerr << "DEBUG: Getting current state" << endl;

	static PomdpStateWorld current_state;

	current_state.car.pos = world_state.car.pos;
	current_state.car.vel = world_state.car.vel;
	current_state.car.heading_dir = world_state.car.heading_dir;
	current_state.num = num_of_peds_world;
	std::vector<PedDistPair> sorted_peds = stateTracker->getSortedPeds();

	//update s.peds to the nearest n_peds peds
	for(int i=0; i<num_of_peds_world; i++) {
		//cout << sorted_peds[i].second.id << endl;
		if(i<sorted_peds.size())
			current_state.peds[i] = world_state.peds[sorted_peds[i].second.id];
	}
	return &current_state;
}

bool POMDPSimulator::ExecuteAction(ACT_TYPE action, OBS_TYPE& obs){
	cerr << "DEBUG: Executing action" << endl;

	double reward;
	stateTracker->updateCar(world_state.car);

	//update the peds in stateTracker
	for(int i=0; i<num_of_peds_world; i++) {
		Pedestrian p(world_state.peds[i].pos.x, world_state.peds[i].pos.y, world_state.peds[i].id);
		stateTracker->updatePed(p);
	}

#ifdef CROSS_CASE
	int new_ped_count=0;
	 while(new_ped_count<1 && numPedInCircle(world_state.peds, num_of_peds_world,world_state.car.pos.x,
			 world_state.car.pos.y)<n_peds && num_of_peds_world < ModelParams::N_PED_WORLD)
	{
		PedStruct new_ped= randomFarPed(world_state.car.pos.x, world_state.car.pos.y);
		new_ped.id = num_of_peds_world;
		world_state.peds[num_of_peds_world]=new_ped;

		num_of_peds_world++;
		world_state.num++;
		new_ped_count++;
		Pedestrian p(new_ped.pos.x, new_ped.pos.y, new_ped.id);
		stateTracker->updatePed(p); //add the new generated ped into stateTracker.ped_list
	}
#endif
	if(worldModel.isGlobalGoal(world_state.car)) {
		cout << "-------------------------------------------------------" << endl;
		cout << "goal_reached=1" << endl;

		target_speed_ = 0.0;
		return true;
	}

	/*cout << "state=[[" << endl;
#ifdef LINE_CASE
	//pomdp->PrintState(s);
#elif defined(CROSS_CASE)
	PrintWorldState(world_state);
#endif
	cout << "]]" << endl;*/

	int collision_peds_id=-1;
	if( world_state.car.vel > 0.001 && worldModel.inCollision(world_state,collision_peds_id) ) {
		cout << "-------------------------------------------------------" << endl;
		cout << "collision=1: " << collision_peds_id<<endl;
		target_speed_= 0.0;
	}
#ifdef LINE_CASE
	else if(worldModel.inCollision(s,collision_peds_id)) {

		target_speed_= 0.0;
		//cout << "close=1: " << collision_peds_id<<endl;
	}
#elif defined(CROSS_CASE)
	else if(worldModel.inCollision(world_state,collision_peds_id)) {
		target_speed_= 0.0 ;
	}
#endif

	bool terminate = static_cast<PedPomdp*>(model_)->Step(world_state,
			Random::RANDOM.NextDouble(),
			action, reward, obs);


/*
	cout<<"Raw world state for obs: [["<<endl;
	model_->PrintState(world_state);
	cout<<"]]"<<endl;*/


	obs=static_cast<PedPomdp*>(model_)->StateToIndex(GetCurrentState());
/*
	cout<<"Raw world state for obs: [["<<endl;
	model_->PrintState(world_state);
	cout<<"]]"<<endl;*/

	//cout << "obs= " << endl;
	step_reward_=reward;

	real_speed_ = world_state.car.vel;


	/* Do look-ahead to adress the latency in real_speed_ */
	cout<<"real speed: "<<real_speed_<<endl;

	float acc=static_cast<PedPomdp*>(model_)->GetAcceleration(action);
	target_speed_ = min(real_speed_ + acc, ModelParams::VEL_MAX);
	target_speed_ = max(target_speed_, 0.0);

	steering_=static_cast<PedPomdp*>(model_)->GetSteering(action);

	publishImitationData(world_state, action, step_reward_, target_speed_);


	if(terminate) {
		cout << "-------------------------------------------------------" << endl;
		cout << "simulation terminate=1" << endl;
		return true;
	}
	return false;
}

double POMDPSimulator::StepReward(PomdpStateWorld& state, ACT_TYPE action){
	double reward=0;
	if (worldModel.isGlobalGoal(state.car)) {
        reward = ModelParams::GOAL_REWARD;
		return reward;
	}

 	// Safety control: collision; Terminate upon collision
    if(state.car.vel > 0.001 && worldModel.inRealCollision(state) ) { /// collision occurs only when car is moving
		reward = ModelParams::CRASH_PENALTY * (state.car.vel * state.car.vel + ModelParams::REWARD_BASE_CRASH_VEL);  //, closest_ped, closest_dist);
		if(action == PedPomdp::ACT_DEC) reward += 0.1;
		return reward;
	}
	// Smoothness control
	reward +=  (action == PedPomdp::ACT_DEC || action == PedPomdp::ACT_ACC) ? -0.1 : 0.0;
	// Speed control: Encourage higher speed
	reward += ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
	//cout<< "   + vel reward:"<< reward << " REWARD_FACTOR_VEL, car vel, VEL_MAX:" <<ModelParams::REWARD_FACTOR_VEL<<","<< state.car.vel <<","<<ModelParams::VEL_MAX<< endl;

	return reward;
}

int POMDPSimulator::numPedInArea(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world)
{
	int num_inside = 0;

	for (int i=0; i<num_of_peds_world; i++)
	{
		if(peds[i].pos.x >= PED_X0 && peds[i].pos.x <= PED_X1 && peds[i].pos.y >= PED_Y0 && peds[i].pos.y <= PED_Y1) num_inside++;
	}

	return num_inside;
}

int POMDPSimulator::numPedInCircle(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world, double car_x, double car_y)
{
	int num_inside = 0;

	for (int i=0; i<num_of_peds_world; i++)
	{
		if((peds[i].pos.x - car_x)*(peds[i].pos.x - car_x) + (peds[i].pos.y - car_y)*(peds[i].pos.y - car_y) <= ModelParams::LASER_RANGE * ModelParams::LASER_RANGE) num_inside++;
	}

	return num_inside;
}

void POMDPSimulator::ImportPeds(std::string filename, PomdpStateWorld& world_state){
	ifstream fin;fin.open(filename, ios::in);
	assert(fin.is_open());
	if (fin.good())
	{
		int num_peds_infile=0;
		string str;
		getline(fin, str);
		istringstream ss(str);
		ss>>num_peds_infile;//throw headers
		cout<<"num_peds_infile"<<num_peds_infile<<endl;
		assert(num_peds_infile==n_peds);
		int i=0;
		while(getline(fin, str))
		{
			if(!str.empty() && i<n_peds)
			{
				istringstream ss(str);
				ss>> world_state.peds[i].id
				>> world_state.peds[i].goal
				>> world_state.peds[i].pos.x
				>> world_state.peds[i].pos.y
				>> world_state.peds[i].speed;
				i++;
			}
		}
		cout<<"peds imported"<<endl;
	}
	else
	{
		cout<<"Empty peds file!"<<endl;
		exit(-1);
	}
}
void POMDPSimulator::ExportPeds(std::string filename, PomdpStateWorld& world_state){
	std::ofstream fout;fout.open(filename, std::ios::trunc);
	assert(fout.is_open());

	fout<<n_peds<<endl;
	for(int i=0; i<n_peds; i++)
	{
		fout<<world_state.peds[i].id<<" "
				<<world_state.peds[i].goal<<" "
				<<world_state.peds[i].pos.x<<" "
				<<world_state.peds[i].pos.y<<" "
				<<world_state.peds[i].speed<<endl;
	}
	fout<<endl;
}


#ifdef LINE_CASE
PedStruct POMDPSimulator::randomPed() {
	int n_goals = worldModel.goals.size();
	int goal = Random::RANDOM.NextInt(n_goals);
	double x = Random::RANDOM.NextDouble(PED_X0, PED_X1);
	double y = Random::RANDOM.NextDouble(PED_Y0, PED_Y1);
	if(goal == n_goals-1) {
		// stop intention
		while(path.mindist(COORD(x, y)) < 1.0) {
			// dont spawn on the path
			x = Random::RANDOM.NextDouble(PED_X0, PED_X1);
			y = Random::RANDOM.NextDouble(PED_Y0, PED_Y1);
		}
	}
	int id = 0;
	return PedStruct(COORD(x, y), goal, id);
}
#elif defined(CROSS_CASE)

PedStruct POMDPSimulator::randomPed() {
   int goal;
   double goal0_x_min = /*14*/12, goal0_x_max = 19/*21*/;
   double goal0_y_min = 4.5, goal0_y_max = 11-0.5;

   double goal1_x_min = 6.5, goal1_x_max = 13.5;
   double goal1_y_min = -1, goal1_y_max = 4;

   if(Random::RANDOM.NextInt(100)>95) goal=worldModel.goals.size() - 1; //setting stop intention with 5% probability.
   else goal = Random::RANDOM.NextInt(worldModel.goals.size() - 1); //uniformly randomly select a goal from those that not is not stopping

   double x;
   double y;
   double speed=ModelParams::PED_SPEED;
   if(goal == 0){
	   x = Random::RANDOM.NextDouble(goal0_x_min, goal0_x_max);
	   y = Random::RANDOM.NextDouble(goal0_y_min, goal0_y_max);
   }
   else if(goal == 1){
	   x = Random::RANDOM.NextDouble(goal1_x_min, goal1_x_max);
	   y = Random::RANDOM.NextDouble(goal1_y_min, goal1_y_max);
   }
   else{// stop intention
	   speed=0;
	   if(Random::RANDOM.NextInt(2)==0){
		   x = Random::RANDOM.NextDouble(goal0_x_min, goal0_x_max);
		   y = Random::RANDOM.NextDouble(goal0_y_min, goal0_y_max);
	   }
	   else{
		   x = Random::RANDOM.NextDouble(goal1_x_min, goal1_x_max);
		   y = Random::RANDOM.NextDouble(goal1_y_min, goal1_y_max);
	   }
	   while(path.mindist(COORD(x, y)) < 1.0) {
		   // dont spawn on the path
		   if(Random::RANDOM.NextInt(2)==0){
			   x = Random::RANDOM.NextDouble(goal0_x_min, goal0_x_max);
			   y = Random::RANDOM.NextDouble(goal0_y_min, goal0_y_max);
		   }
		   else{
			   x = Random::RANDOM.NextDouble(goal1_x_min, goal1_x_max);
			   y = Random::RANDOM.NextDouble(goal1_y_min, goal1_y_max);
		   }
	   }
   }
   int id = 0;
   return PedStruct(COORD(x, y), goal, id, speed);
}

PedStruct POMDPSimulator::randomFarPed(double car_x, double car_y) { //generate pedestrians that are not close to the car
    int goal;
    double goal0_x_min = /*14*/28, goal0_x_max = /*21*/31;
    double goal0_y_min = /*4.5*/2, goal0_y_max = /*11-0.5*/12;

    double goal1_x_min = 6.5, goal1_x_max = 13.5;
    double goal1_y_min = /*-1*/-4, goal1_y_max = /*4*/1;

    double x;
    double y;

    if(Random::RANDOM.NextInt(2)==0){
        goal = 0;
        x = Random::RANDOM.NextDouble(goal0_x_min, goal0_x_max);
        y = Random::RANDOM.NextDouble(goal0_y_min, goal0_y_max);
    }
    else{
        goal = 1;
        x = Random::RANDOM.NextDouble(goal1_x_min, goal1_x_max);
        y = Random::RANDOM.NextDouble(goal1_y_min, goal1_y_max);
    }

    while(COORD::EuclideanDistance(COORD(car_x, car_y), COORD(x, y)) < 2.0) {
        if(Random::RANDOM.NextInt(2)==0){
            goal = 0;
            x = Random::RANDOM.NextDouble(goal0_x_min, goal0_x_max);
            y = Random::RANDOM.NextDouble(goal0_y_min, goal0_y_max);
        }
        else{
            goal = 1;
            x = Random::RANDOM.NextDouble(goal1_x_min, goal1_x_max);
            y = Random::RANDOM.NextDouble(goal1_y_min, goal1_y_max);
        }
    }

    int id = 0;
    return PedStruct(COORD(x, y), goal, id);
}
#endif


PedStruct POMDPSimulator::randomPedAtCircleEdge(double car_x, double car_y) {
	int n_goals = worldModel.goals.size();
	int goal = Random::RANDOM.NextInt(n_goals);
	double x, y;
	double angle;

	angle = Random::RANDOM.NextDouble(0, M_PI/2);

	if(goal==3) {
		x = car_x - ModelParams::LASER_RANGE * cos(angle);
		y = car_y - ModelParams::LASER_RANGE * sin(angle);
	} else if(goal == 4){
		x = car_x + ModelParams::LASER_RANGE * cos(angle);
		y = car_y - ModelParams::LASER_RANGE * sin(angle);
	} else if(goal == 2 || goal == 1){
		x = car_x + ModelParams::LASER_RANGE * cos(angle);
		y = car_y + ModelParams::LASER_RANGE * sin(angle);
	} else if(goal == 0 || goal == 5){
		x = car_x - ModelParams::LASER_RANGE * cos(angle);
		y = car_y + ModelParams::LASER_RANGE * sin(angle);
	} else{
		angle = Random::RANDOM.NextDouble(-M_PI, M_PI);
		x = car_x + ModelParams::LASER_RANGE * cos(angle);
		y = car_y + ModelParams::LASER_RANGE * sin(angle);
		if(goal == n_goals-1) {
			while(path.mindist(COORD(x, y)) < 1.0) {
				// dont spawn on the path
				angle = Random::RANDOM.NextDouble(-M_PI, M_PI);
				x = car_x + ModelParams::LASER_RANGE * cos(angle);
				y = car_y + ModelParams::LASER_RANGE * sin(angle);
			}
		}
	}
	int id = 0;
	return PedStruct(COORD(x, y), goal, id);
}

void POMDPSimulator::generateFixedPed(PomdpState &s) {

	s.peds[0] = PedStruct(COORD(38.1984, 50.6322), 5, 0);

	s.peds[1] = PedStruct(COORD(35.5695, 46.2163), 4, 1);

	s.peds[2] = PedStruct(COORD(41.1636, 49.6807), 4, 2);

	s.peds[3] = PedStruct(COORD(35.1755, 41.4558), 4, 3);

	s.peds[4] = PedStruct(COORD(37.9329, 35.6085), 3, 4);

	s.peds[5] = PedStruct(COORD(41.0874, 49.6448), 5, 5);
}

void POMDPSimulator::PrintWorldState(PomdpStateWorld state, ostream& out) {
    COORD& carpos = state.car.pos;

	out << "car pos / heading dir / vel = " << "(" << carpos.x<< ", " <<carpos.y << ") / "
        << state.car.heading_dir << " / "
        << state.car.vel << endl;
	out<< state.num << " pedestrians " << endl;
	int mindist_id=0;
    double min_dist = std::numeric_limits<int>::max();

	for(int i = 0; i < state.num; i ++) {
		if(COORD::EuclideanDistance(state.peds[i].pos, carpos)<min_dist)
		{
			min_dist=COORD::EuclideanDistance(state.peds[i].pos, carpos);
			mindist_id=i;
		}
		out << "ped " << i << ": id / pos / vel / goal / dist2car / infront =  " << state.peds[i].id << " / "
            << "(" << state.peds[i].pos.x << ", " << state.peds[i].pos.y << ") / "
            << state.peds[i].speed << " / "
            << state.peds[i].goal << " / "
            << COORD::EuclideanDistance(state.peds[i].pos, carpos) << "/"
			<< worldModel.inFront(state.peds[i].pos, state.car) << endl;
	}
    if (state.num > 0)
        min_dist = COORD::EuclideanDistance(carpos, state.peds[/*0*/mindist_id].pos);
	out << "MinDist: " << min_dist << endl;
}


void POMDPSimulator::publishImitationData(PomdpStateWorld& planning_state, ACT_TYPE safeAction, float reward, float cmd_vel)
{
	// car for publish
	ped_is_despot::car_info p_car;
	p_car.car_pos.x = planning_state.car.pos.x;
	p_car.car_pos.y = planning_state.car.pos.y;
	p_car.car_pos.z = 0;
	p_car.car_yaw = planning_state.car.heading_dir;

	p_IL_data.past_car = p_IL_data.cur_car;
    p_IL_data.cur_car = p_car;

	// path for publish
	nav_msgs::Path navpath;
	ros::Time plan_time = ros::Time::now();

	navpath.header.frame_id = global_frame_id;
	navpath.header.stamp = plan_time;
	
	for(const auto& s: path) {
		geometry_msgs::PoseStamped pose;
		pose.header.stamp = plan_time;
		pose.header.frame_id = global_frame_id;
		pose.pose.position.x = s.x;
		pose.pose.position.y = s.y;
		pose.pose.position.z = 0.0;
		pose.pose.orientation.x = 0.0;
		pose.pose.orientation.y = 0.0;
		pose.pose.orientation.z = 0.0;
		pose.pose.orientation.w = 1.0;
		navpath.poses.push_back(pose);
	}
	p_IL_data.plan = navpath; 

	// peds for publish
	ped_is_despot::peds_info p_ped;
	// only publish information for N_PED_IN peds for imitation learning
	for (int i = 0; i < ModelParams::N_PED_IN; i++){
		ped_is_despot::ped_info ped;
        ped.ped_id = planning_state.peds[i].id;
        ped.ped_goal_id = planning_state.peds[i].goal;
        ped.ped_speed = 1.2;
        ped.ped_pos.x = planning_state.peds[i].pos.x;
        ped.ped_pos.y = planning_state.peds[i].pos.y;
        ped.ped_pos.z = 0;
        p_ped.peds.push_back(ped);
    }

    p_IL_data.past_peds = p_IL_data.cur_peds;
    p_IL_data.cur_peds = p_ped;

	// ped belief for pushlish
	int i=0;
	ped_is_despot::peds_believes pbs;	
	for(auto & kv: beliefTracker->peds)
	{
		ped_is_despot::ped_belief pb;
		PedBelief belief = kv.second;
		pb.ped_x=belief.pos.x;
		pb.ped_y=belief.pos.y;
		pb.ped_id=belief.id;
		for(auto & v : belief.prob_goals)
			pb.belief_value.push_back(v);
		pbs.believes.push_back(pb);
	}
	pbs.cmd_vel=stateTracker->carvel;
	pbs.robotx=stateTracker->carpos.x;
	pbs.roboty=stateTracker->carpos.y;

	p_IL_data.believes = pbs.believes;


	// action for publish
	geometry_msgs::Twist p_action_reward;

    p_IL_data.action_reward.linear.y=reward;
    p_IL_data.action_reward.linear.z=cmd_vel;


    p_IL_data.action_reward.linear.x = static_cast<PedPomdp*>(model_)->GetAcceleration(safeAction);
	p_IL_data.action_reward.angular.x = static_cast<PedPomdp*>(model_)->GetSteering(safeAction);


    IL_pub.publish(p_IL_data);

}

