#include"ped_pomdp.h"
//#include "despotstar.h"
#include "WorldModel.h"
#include"state.h"
#include"Path.h"
#include "despot/solver/despot.h"
#include "custom_particle_belief.h"
#include "simulator_hyp.h"
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


Simulator::Simulator(): SimpleTUI()/// set the path to be a straight line
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
}


Simulator::~Simulator()
{
}
int Simulator::numPedInArea(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world)
{
	int num_inside = 0;

	for (int i=0; i<num_of_peds_world; i++)
	{
		if(peds[i].pos.x >= PED_X0 && peds[i].pos.x <= PED_X1 && peds[i].pos.y >= PED_Y0 && peds[i].pos.y <= PED_Y1) num_inside++;
	}

	return num_inside;
}

int Simulator::numPedInCircle(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world, double car_x, double car_y)
{
	int num_inside = 0;

	for (int i=0; i<num_of_peds_world; i++)
	{
		if((peds[i].pos.x - car_x)*(peds[i].pos.x - car_x) + (peds[i].pos.y - car_y)*(peds[i].pos.y - car_y) <= ModelParams::LASER_RANGE * ModelParams::LASER_RANGE) num_inside++;
	}

	return num_inside;
}

int Simulator::run(int argc, char *argv[]) {

	const char *program = (argc > 0) ? argv[0] : "despot";

	argc -= (argc > 0);
	argv += (argc > 0); // skip program name argv[0] if present

	option::Stats stats(usage, argc, argv);
	option::Option *options = new option::Option[stats.options_max];
	option::Option *buffer = new option::Option[stats.buffer_max];
	option::Parser parse(usage, argc, argv, options, buffer);

	string solver_type = "DESPOT";
	bool search_solver;

	/* =========================
	* Parse required parameters
	* =========================*/
	int num_runs = 1;
	string simulator_type = "pomdp";
	string belief_type = "DEFAULT";
	int time_limit = -1;

	/* =========================================
	* Problem specific default parameter values
	*=========================================*/
	InitializeDefaultParameters();

	/* =========================
	* Parse optional parameters
	* =========================*/
	if (options[E_HELP]) {
		cout << "Usage: " << program << " [options]" << endl;
		option::printUsage(std::cout, usage);
		return 0;
	}

	OptionParse(options, num_runs, simulator_type, belief_type, time_limit,
			  solver_type, search_solver,Globals::config.useGPU);
	PrepareGPU();

	//cout << "====================" << endl;
	WorldStateTracker stateTracker(worldModel);
	WorldBeliefTracker beliefTracker(worldModel, stateTracker);
	PedPomdp * pomdp = new PedPomdp(worldModel);

	ScenarioLowerBound *lower_bound = pomdp->CreateScenarioLowerBound("SMART");
	ScenarioUpperBound *upper_bound = pomdp->CreateScenarioUpperBound("SMART", "SMART");

	Solver * solver = new DESPOT(pomdp, lower_bound, upper_bound, NULL, Globals::config.useGPU);

	PrepareGPUData(pomdp, solver);

	for(int i=0; i<num_runs; i++){
		cout<<"++++++++++++++++++++++ ROUND "<<i<<" ++++++++++++++++++++"<<endl;
		//for pomdp planning and print world info
		PomdpState s;
		// for tracking world state
		PomdpStateWorld world_state;

		world_state.car.pos = 0;
		world_state.car.vel = 0;
		world_state.car.dist_travelled = 0;
		world_state.num = n_peds;

		if(FIX_SCENARIO==0)
		{
			//generate initial n_peds peds
			if(DESPOT::Debug_mode/*false*/)
			{
				ImportPeds("Peds.txt", world_state);
				//pomdp->PrintWorldState(world_state);
			}
			else
			{
				for(int i=0; i<n_peds; i++) {
					world_state.peds[i] = randomPed();
					world_state.peds[i].id = i;
				}
			}
		}
		else if(FIX_SCENARIO==1)
		{
			ImportPeds("Peds.txt", world_state);
		}
		else if(FIX_SCENARIO==2)
		{
			//generate initial n_peds peds
			for(int i=0; i<n_peds; i++) {
				world_state.peds[i] = randomPed();
				world_state.peds[i].id = i;
			}
			ExportPeds("Peds.txt",world_state);
		}

		int num_of_peds_world = n_peds;

		double total_reward_dis = 0, total_reward_nondis=0;
		int step = 0;
		cout<<"LASER_RANGE= "<<ModelParams::LASER_RANGE<<endl;
		for(int step=0; step < /*180*/Globals::config.sim_len; step++) {
			cout << "====================" << "step= " << step <<"===================="<< endl;
			double reward;
			uint64_t obs;
			stateTracker.updateCar(path[world_state.car.pos], world_state.car.dist_travelled);
			stateTracker.updateVel(world_state.car.vel);

			//update the peds in stateTracker
			for(int i=0; i<num_of_peds_world; i++) {
				Pedestrian p(world_state.peds[i].pos.x, world_state.peds[i].pos.y, world_state.peds[i].id);
				stateTracker.updatePed(p);
			}

	#ifdef CROSS_CASE
			if(/*false*//*true*/FIX_SCENARIO==0 /*||*/ /*FIX_SCENARIO==2|| FIX_SCENARIO==1*/){
				int new_ped_count=0;
				 while(new_ped_count<1 && numPedInCircle(world_state.peds, num_of_peds_world,path[world_state.car.pos].x, path[world_state.car.pos].y)<n_peds && num_of_peds_world < ModelParams::N_PED_WORLD)
				{
					//PedStruct new_ped= randomPedAtCircleEdge(path[world_state.car.pos].x, path[world_state.car.pos].y);
					PedStruct new_ped= randomFarPed(path[world_state.car.pos].x, path[world_state.car.pos].y);
					new_ped.id = num_of_peds_world;
					world_state.peds[num_of_peds_world]=new_ped;

					num_of_peds_world++;
					world_state.num++;
					new_ped_count++;
					Pedestrian p(new_ped.pos.x, new_ped.pos.y, new_ped.id);
					stateTracker.updatePed(p); //add the new generated ped into stateTracker.ped_list
				}
			}
	#endif
			if(worldModel.isGlobalGoal(world_state.car)) {
				cout << "goal_reached=1" << endl;
				break;
			}

			s.car.pos = world_state.car.pos;
			s.car.vel = world_state.car.vel;
			s.car.dist_travelled = world_state.car.dist_travelled;
			s.num = n_peds;
			std::vector<PedDistPair> sorted_peds = stateTracker.getSortedPeds();

			//update s.peds to the nearest n_peds peds
			for(int i=0; i<n_peds; i++) {
				//cout << sorted_peds[i].second.id << endl;
				if(i<sorted_peds.size())
					s.peds[i] = world_state.peds[sorted_peds[i].second.id];
			}

			cout << "state=[[" << endl;
	#ifdef LINE_CASE
			pomdp->PrintState(s);
	#elif defined(CROSS_CASE)
			pomdp->PrintWorldState(world_state);
			if(FIX_SCENARIO==2)
			{
				ExportPeds("Peds.txt",world_state);
			}
			//pomdp->PrintState(s);
	#endif
			cout << "]]" << endl;

			int collision_peds_id=-1;
			if( world_state.car.vel > 0.001 && worldModel.inCollision(world_state,collision_peds_id) ) {
				cout << "collision=1: " << collision_peds_id<<endl;
			}
	#ifdef LINE_CASE
			else if(worldModel.inCollision(s,collision_peds_id)) {
				//cout << "close=1: " << collision_peds_id<<endl;
			}
	#elif defined(CROSS_CASE)
			else if(worldModel.inCollision(world_state,collision_peds_id)) {
				//cout << "close=1: " << collision_peds_id<<endl;
			}
	#endif
			beliefTracker.update();
			////vector<PomdpState> samples = beliefTracker.sample(Globals::config.num_scenarios);
			vector<PomdpState> samples = beliefTracker.sample(max(2000,5*Globals::config.num_scenarios));//samples are used to construct particle belief. num_scenarios is the number of scenarios sampled from particles belief to construct despot
			vector<State*> particles = pomdp->ConstructParticles(samples);
			ParticleBelief* pb = new ParticleBelief(particles, pomdp);
			//MaxLikelihoodScenario* pb = new MaxLikelihoodScenario(particles, pomdp);
			solver->belief(pb);
			Globals::config.silence = /*false*/true;
			int act = solver->Search().action;
			cout << "act= " << act << endl;

			if(FIX_SCENARIO==1)
				act=0;
			/*else if (FIX_SCENARIO==2)
				act=1;*/
			if(FIX_SCENARIO==1 /*|| FIX_SCENARIO==2*/)cout << "rewrite act to " << act << endl;

			bool terminate = pomdp->Step(world_state,
					Random::RANDOM.NextDouble(),
					act, reward, obs);
			cout << "obs= " << endl;
			cout << "reward= " << reward << endl;
			total_reward_nondis += reward /** Globals::Discount(step)*/;//Panpan: use non-discounted reward
			total_reward_dis+=reward * Globals::Discount(step);
			if(terminate) {
				cout << "terminate=1" << endl;
				break;
			}
		}
		cout << "final_state=[[" << endl;
		pomdp->PrintState(s);
		cout << "]]" << endl;
		cout << "total_reward= " << total_reward_dis << endl;
		cout << "total_nondiscounted_reward= " << total_reward_nondis << endl;

	}

	ReleaseGPUData(pomdp, solver);
   // cout << "====================" << endl;
	return 1;
}

void Simulator::ImportPeds(std::string filename, PomdpStateWorld& world_state){
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
				>> world_state.peds[i].vel;
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
void Simulator::ExportPeds(std::string filename, PomdpStateWorld& world_state){
	std::ofstream fout;fout.open(filename, std::ios::trunc);
	assert(fout.is_open());

	fout<<n_peds<<endl;
	for(int i=0; i<n_peds; i++)
	{
		fout<<world_state.peds[i].id<<" "
				<<world_state.peds[i].goal<<" "
				<<world_state.peds[i].pos.x<<" "
				<<world_state.peds[i].pos.y<<" "
				<<world_state.peds[i].vel<<endl;
	}
	fout<<endl;
}


#ifdef LINE_CASE
PedStruct Simulator::randomPed() {
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

PedStruct Simulator::randomPed() {
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

PedStruct Simulator::randomFarPed(double car_x, double car_y) { //generate pedestrians that are not close to the car
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


PedStruct Simulator::randomPedAtCircleEdge(double car_x, double car_y) {
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

void Simulator::generateFixedPed(PomdpState &s) {

	s.peds[0] = PedStruct(COORD(38.1984, 50.6322), 5, 0);

	s.peds[1] = PedStruct(COORD(35.5695, 46.2163), 4, 1);

	s.peds[2] = PedStruct(COORD(41.1636, 49.6807), 4, 2);

	s.peds[3] = PedStruct(COORD(35.1755, 41.4558), 4, 3);

	s.peds[4] = PedStruct(COORD(37.9329, 35.6085), 3, 4);

	s.peds[5] = PedStruct(COORD(41.0874, 49.6448), 5, 5);
}


/*int main(int argc, char** argv) {
    //if (argc >= 2) ModelParams::CRASH_PENALTY = -atof(argv[1]);
    //else
	ModelParams::CRASH_PENALTY = -1000;
    

    // TODO the seed should be initialized properly so that
    // different process as well as process on different machines
    // all get different seeds
    Seeds::root_seed(get_time_second());
    // Global random generator
    double seed = Seeds::Next();
    Random::RANDOM = Random(seed);
    cerr << "Initialized global random generator with seed " << seed << endl;

    //n_sim=2;
    ////if (argc >= 3) n_sim = atoi(argv[2]);
   // //if (argc >= 4) Globals::config.pruning_constant = double(atof(argv[3]));
    //Simulator sim;
    //for(int i=0; i<n_sim; i++){
   // 	cout<<"++++++++++++++++++++++ ROUND "<<i<<" ++++++++++++++++++++"<<endl;
   //     sim.run(argc, argv);
    //}
    Simulator sim;
    sim.run(argc, argv);
}*/
