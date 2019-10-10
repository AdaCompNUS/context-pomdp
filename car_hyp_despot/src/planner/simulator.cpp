#include "WorldModel.h"
#include"state.h"
#include"Path.h"
#include "despot/solver/despot.h"
#include "custom_particle_belief.h"
#include "simulator.h"
#include "ped_pomdp.h"

#include <despot/GPUcore/CudaInclude.h>

#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;
using namespace despot;
//#define LINE_CASE
#define CROSS_CASE

WorldStateTracker* SimulatorBase::stateTracker;
WorldModel SimulatorBase::worldModel;     
bool SimulatorBase::agents_data_ready = false;
bool SimulatorBase::agents_path_data_ready = false;

int n_sim = 1;

#ifdef LINE_CASE
	const double AGENT_X0 = 35;/// used to generate peds' locations, where x is in (AGENT_X0, AGENT_X1), and y is in (AGENT_Y0, AGENT_Y1)
	const double AGENT_Y0 = 35;
	const double AGENT_X1 = 42;
	const double AGENT_Y1 = 52;
	const int n_peds = 6; // should be smaller than ModelParams::N_AGENT_IN
#elif defined(CROSS_CASE)
	const double AGENT_X0 = 0;/// used to generate peds' locations, where x is in (AGENT_X0, AGENT_X1), and y is in (AGENT_Y0, AGENT_Y1)
	const double AGENT_Y0 = 0;
	const double AGENT_X1 = 20;
	const double AGENT_Y1 = 15;
	const int n_peds = 20;//6; // should be smaller than ModelParams::N_AGENT_IN
#endif


Simulator::Simulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed)/// set the path to be a straight line
: SimulatorBase(_nh, ""), POMDPWorld(model, seed)
{
#ifdef LINE_CASE
	start.x=40;start.y=40;
	goal.x=40;goal.y=50;
#elif defined(CROSS_CASE)
	double shift_x = 0.0;
	double shift_y = 2.5;
	start.x = 20; start.y = 7.5 + shift_y;
	goal.x = 10.0 + shift_x; goal.y = /*-2.0*/0.0;
#endif

	Path p;
	p.push_back(start);
	p.push_back(COORD(13 + shift_x,7.5 + shift_y));
	p.push_back(COORD(12.6 + shift_x,7.4 + shift_y));
	p.push_back(COORD(12.2 + shift_x,7.25 + shift_y));
	p.push_back(COORD(11.8 + shift_x,7.08 + shift_y));
	p.push_back(COORD(11.4 + shift_x,6.8 + shift_y));
	p.push_back(COORD(11 + shift_x,6.45 + shift_y));
	p.push_back(COORD(10.6 + shift_x,5.9 + shift_y));
	p.push_back(COORD(10.28 + shift_x,5.25 + shift_y));
	p.push_back(COORD(10.1 + shift_x,4.5 + shift_y));
	p.push_back(COORD(10 + shift_x,3 + shift_y));
	p.push_back(goal);
	path = p.interpolate();
	worldModel.setPath(path);
	num_of_peds_world=0;
	stateTracker=new WorldStateTracker(worldModel);
	rand_ = new Random(Globals::config.root_seed);
}

Simulator::~Simulator()
{

}

State* Simulator::Initialize(){

	// for tracking world state
	world_state.car.pos = start;
	world_state.car.vel = 0;
	world_state.car.heading_dir = M_PI; // relative to the x axis
	world_state.num = n_peds;

	if(FIX_SCENARIO==0)
	{
		if(DESPOT::Debug_mode || Globals::config.experiment_mode)
		{
			//ImportPeds("Peds.txt", world_state);
			//cout << "[FIX_SCENARIO] load peds from "<< "Peds.txt" << endl;
		}
		else
		{
			for(int i=0; i<n_peds; i++) {
				world_state.agents[i] = randomPed();
				world_state.agents[i].id = i;
				world_state.agents[i].mode = random_ped_mode(world_state.agents[i]);
			}
			//if (!Globals::config.experiment_mode)
				//ExportPeds("Peds.txt",world_state);
		}
	}
	else if(FIX_SCENARIO==1)
	{
		//ImportPeds("Peds.txt", world_state);
		//cout << "[FIX_SCENARIO] load peds from "<< "Peds.txt" << endl;
	}
	else if(FIX_SCENARIO==2)
	{
		//generate initial n_peds peds
		for(int i=0; i<n_peds; i++) {
			world_state.agents[i] = randomPed();
			world_state.agents[i].id = i;
			world_state.agents[i].mode = random_ped_mode(world_state.agents[i]);
		}
		//ExportPeds("Peds.txt",world_state);
	}

	num_of_peds_world = n_peds;

	double total_reward_dis = 0, total_reward_nondis=0;
	int step = 0;
	cout<<"LASER_RANGE= "<<ModelParams::LASER_RANGE<<endl;

	worldModel.InitRVO();
}

State* Simulator::GetCurrentState(){
	static PomdpStateWorld current_state;

	//cout << "[GetCurrentState] current num peds in simulator: " << num_of_peds_world << endl;

	auto sorted_peds = stateTracker->getSortedAgents();


	current_state.car.pos = world_state.car.pos;
	current_state.car.vel = world_state.car.vel;
	current_state.car.heading_dir = world_state.car.heading_dir;
	current_state.num = /*num_of_peds_world*/sorted_peds.size();

	if (sorted_peds.size() ==0){
		current_state.num = world_state.num;

		for(int i=0; i<current_state.num; i++) {
			current_state.agents[i] = world_state.agents[i];
			current_state.agents[i].mode = world_state.agents[i].mode;
		}
	}
	else{
		//update s.agents to the nearest n_peds peds
		for(int i=0; i<current_state.num; i++) {
			if(i<sorted_peds.size()){
				current_state.agents[i] = world_state.agents[sorted_peds[i].second->id];
				current_state.agents[i].mode = world_state.agents[sorted_peds[i].second->id].mode;
			}
		}
	}
	return &current_state;
}

void Simulator::UpdateWorld(){

	cout << "[Simulator::UpdateWorld] \n";

	stateTracker->updateCar(world_state.car);

	//update the peds in stateTracker
	for(int i=0; i<num_of_peds_world; i++) {
		Pedestrian p(world_state.agents[i].pos.x, world_state.agents[i].pos.y, world_state.agents[i].id);
		stateTracker->updatePed(p);
	}

#ifdef CROSS_CASE
	if(FIX_SCENARIO==0){
		int new_ped_count=0;
		while(new_ped_count<1 && numPedInCircle(world_state.agents, num_of_peds_world,world_state.car.pos.x, world_state.car.pos.y)<n_peds
			&& num_of_peds_world < ModelParams::N_PED_WORLD)
		{
			AgentStruct new_ped= randomFarPed(world_state.car.pos.x, world_state.car.pos.y);
			new_ped.id = num_of_peds_world;
			world_state.agents[num_of_peds_world]=new_ped;
			world_state.agents[num_of_peds_world].mode = random_ped_mode(new_ped);

			num_of_peds_world++;
			world_state.num++;
			new_ped_count++;
			Pedestrian p(new_ped.pos.x, new_ped.pos.y, new_ped.id);
			stateTracker->updatePed(p); //add the new generated ped into stateTracker.ped_list

			cout << "[Simulator] Create new pedestrian "<< new_ped.id <<" in world simulator."<< endl;

		}

		cout << "[Simulator] Number of peds in laser range: "<< numPedInCircle(world_state.agents, num_of_peds_world,world_state.car.pos.x, world_state.car.pos.y)
				<< endl;
	}
#endif

	if(DESPOT::Debug_mode)
		static_cast<PedPomdp*>(model_)->PrintWorldState(world_state);
}


bool Simulator::ExecuteAction(ACT_TYPE action, OBS_TYPE& obs){

	if(worldModel.isGlobalGoal(world_state.car)) {
		cout << "-------------------------------------------------------" << endl;
		cout << "goal_reached=1" << endl;
		return true;
	}

	int collision_peds_id=-1;
	if( world_state.car.vel > 0.001 && worldModel.inRealCollision(world_state,collision_peds_id) ) {
		cout << "-------------------------------------------------------" << endl;
		cout << "collision=1: " << collision_peds_id<<endl;
	}
#ifdef LINE_CASE
	else if(worldModel.inCollision(s,collision_peds_id)) {
	}
#elif defined(CROSS_CASE)
	else if(worldModel.inCollision(world_state,collision_peds_id)) {
	}
#endif

	if(FIX_SCENARIO==1){
		cout << "[FIX_SCENARIO] act= " << action << endl;
		action=0;
		cout << "[FIX_SCENARIO] rewrite act to " << action << endl;
	}

	double reward;

	// action = 0; // debugging

	bool terminate = static_cast<PedPomdp*>(model_)->Step(world_state,
			rand_->NextDouble(),
			action, reward, obs);

	obs=static_cast<PedPomdp*>(model_)->StateToIndex(GetCurrentState());

	//cout << "[Simulator] Current world state:" << endl;
	//static_cast<PedPomdp*>(model_)->PrintWorldState(world_state);

	step_reward_=reward;

	if(terminate) {
		cout << "-------------------------------------------------------" << endl;
		cout << "simulation terminate=1" << endl;

		return true;

	}
	return false;
}

int Simulator::numPedInCircle(AgentStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world, double car_x, double car_y)
{
	int num_inside = 0;

	for (int i=0; i<num_of_peds_world; i++)
	{
		if((peds[i].pos.x - car_x)*(peds[i].pos.x - car_x) + (peds[i].pos.y - car_y)*(peds[i].pos.y - car_y) <= ModelParams::LASER_RANGE * ModelParams::LASER_RANGE) num_inside++;
	}

	return num_inside;
}
/*
void Simulator::ImportPeds(std::string filename, PomdpStateWorld& world_state){
	ifstream fin;fin.open(filename, ios::in);

	try
	{
		std::cout << "Working dir: " << GetCurrentWorkingDir() << std::endl;
		assert(fin.is_open());
	}
	catch (int e)
	{
		cout << __FUNCTION__ <<": Exception Nr. " << e << '\n';
		exit(1);
	}

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
				ss>> world_state.agents[i].id
				>> world_state.agents[i].intention
				>> world_state.agents[i].pos.x
				>> world_state.agents[i].pos.y
				>> world_state.agents[i].vel;
				i++;
			}
		}
		cout<<"peds imported"<<endl;

		for(int i=0; i< n_peds; i++)
			world_state.agents[i].mode = random_ped_mode(world_state.agents[i]);
	}
	else
	{
		cout<<"Empty peds file!"<<endl;
		exit(-1);
	}
}
void Simulator::ExportPeds(std::string filename, PomdpStateWorld& world_state){
	std::ofstream fout;fout.open(filename, std::ios::trunc);
	assert(fout.is_open());

	fout<<n_peds<<endl;
	for(int i=0; i<n_peds; i++)
	{
		fout<<world_state.agents[i].id<<" "
				<<world_state.agents[i].intention<<" "
				<<world_state.agents[i].pos.x<<" "
				<<world_state.agents[i].pos.y<<" "
				<<world_state.agents[i].vel<<endl;
	}
	fout<<endl;
}
*/

#ifdef LINE_CASE
AgentStruct Simulator::randomPed() {
	int n_goals = worldModel.goals.size();
	int goal = rand_->NextInt(n_goals);
	double x = rand_->NextDouble(AGENT_X0, AGENT_X1);
	double y = rand_->NextDouble(AGENT_Y0, AGENT_Y1);
	if(goal == n_goals-1) {
		// stop intention
		while(path.mindist(COORD(x, y)) < 1.0) {
			// dont spawn on the path
			x = rand_->NextDouble(AGENT_X0, AGENT_X1);
			y = rand_->NextDouble(AGENT_Y0, AGENT_Y1);
		}
	}
	int id = 0;
	return AgentStruct(COORD(x, y), goal, id);
}
#elif defined(CROSS_CASE)

AgentStruct Simulator::randomPed() {
   int goal;

   // goal 0: left
   double goal0_x_min = /*14*/3, goal0_x_max = 13/*21*/;
   double goal0_y_min = 5.5, goal0_y_max = 11-1.5;

   // goal 1: up
   double goal1_x_min = 7.5, goal1_x_max = 12.5;
   double goal1_y_min = /*-1*/5, goal1_y_max = 15;


   if(rand_->NextInt(100)>95) goal=worldModel.goals.size() - 1; //setting stop intention with 5% probability.
   else goal = rand_->NextInt(worldModel.goals.size() - 1); //uniformly randomly select a goal from those that not is not stopping

   double x;
   double y;
   double speed=ModelParams::PED_SPEED;
   if(goal == 0){
	   x = rand_->NextDouble(goal0_x_min, goal0_x_max);
	   y = rand_->NextDouble(goal0_y_min, goal0_y_max);
   }
   else if(goal == 1){
	   x = rand_->NextDouble(goal1_x_min, goal1_x_max);
	   y = rand_->NextDouble(goal1_y_min, goal1_y_max);
   }
   else{// stop intention
	   speed=0;
	   if(rand_->NextInt(2)==0){
		   x = rand_->NextDouble(goal0_x_min, goal0_x_max);
		   y = rand_->NextDouble(goal0_y_min, goal0_y_max);
	   }
	   else{
		   x = rand_->NextDouble(goal1_x_min, goal1_x_max);
		   y = rand_->NextDouble(goal1_y_min, goal1_y_max);
	   }
	   while(path.mindist(COORD(x, y)) < 1.0) {
		   // dont spawn on the path
		   if(rand_->NextInt(2)==0){
			   x = rand_->NextDouble(goal0_x_min, goal0_x_max);
			   y = rand_->NextDouble(goal0_y_min, goal0_y_max);
		   }
		   else{
			   x = rand_->NextDouble(goal1_x_min, goal1_x_max);
			   y = rand_->NextDouble(goal1_y_min, goal1_y_max);
		   }
	   }
   }
   int id = 0;
   return AgentStruct(COORD(x, y), goal, id, speed);
}

AgentStruct Simulator::randomFarPed(double car_x, double car_y) { //generate pedestrians that are not close to the car
    int goal;
    // goal 0: left
    double goal0_x_min = /*14*/28/*30*/, goal0_x_max = /*21*/31/*33*/;
    double goal0_y_min = /*4.5*/2, goal0_y_max = /*11-0.5*/12;

    // goal 1: up
    double goal1_x_min = /*6.5*/5.5, goal1_x_max = /*13.5*/14.5;
    double goal1_y_min = /*-1*//*-4*/-6, goal1_y_max = /*4*//*1*/-2;

    double x;
    double y;

    cout << __FUNCTION__ << " rand seed = " << rand_->seed() << endl;

    if(rand_->NextInt(2)==0){
        goal = 0;
        x = rand_->NextDouble(goal0_x_min, goal0_x_max);
        y = rand_->NextDouble(goal0_y_min, goal0_y_max);
    }
    else{
        goal = 1;
        x = rand_->NextDouble(goal1_x_min, goal1_x_max);
        y = rand_->NextDouble(goal1_y_min, goal1_y_max);
    }

    while(COORD::EuclideanDistance(COORD(car_x, car_y), COORD(x, y)) < 2.0) {
        if(rand_->NextInt(2)==0){
            goal = 0;
            x = rand_->NextDouble(goal0_x_min, goal0_x_max);
            y = rand_->NextDouble(goal0_y_min, goal0_y_max);
        }
        else{
            goal = 1;
            x = rand_->NextDouble(goal1_x_min, goal1_x_max);
            y = rand_->NextDouble(goal1_y_min, goal1_y_max);
        }
    }

    int id = 0;
    return AgentStruct(COORD(x, y), goal, id);
}

int Simulator::random_ped_mode(AgentStruct& ped) {
	if (ped.intention == worldModel.goals.size()-1)
		return AGENT_ATT;

	double prob = Random::RANDOM.NextDouble();

	if (prob < /*0.9*/ 0.3){
		return AGENT_ATT;
	}
	else
		return AGENT_DIS;
}
#endif



