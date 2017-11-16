#include "belief_update/belief_update_exact.h"
#include "belief_update/belief_update_particle.h"
#include "globals.h"
#include "lower_bound/lower_bound_policy_mode.h"
#include "lower_bound/lower_bound_policy_random.h"
#include "lower_bound/lower_bound_policy_suffix.h"
#include "model.h"
#include "optionparser.h"
#include "problems/pedestrian_changelane/pedestrian_changelane.h"
#include "problems/pedestrian_changelane/map.h"
#include "problems/pedestrian_changelane/window.h"
#include "problems/pedestrian_changelane/math_utils.h"
#include "problems/pedestrian_changelane/SFM.h"
#include "problems/pedestrian_changelane/param.h"
#include "solver.h"
#include "util_uniform.h"
#include "upper_bound/upper_bound_nonstochastic.h"
#include "upper_bound/upper_bound_stochastic.h"
#include "world.h"
#include <iomanip>
#include "problems/pedestrian_changelane/world_simulator.h"



using namespace std;

Model<PedestrianState>* Simulator;
ILowerBound<PedestrianState>* lb_global;
IUpperBound<PedestrianState>* ub_global;
BeliefUpdate<PedestrianState>* bu_global;
RandomStreams * streams_global;
option::Option* options_global;
/* The seeds used by different components of the system to generate random 
 * numbers are all derived from the root seed so that an experiment can be 
 * repeated deterministically. The seed for the random-number stream of 
 * particle i is given by config.n_particles ^ i (see RandomStreams). The
 * remaining seeds are generated as follows.
 */
int WorldSeed() {
  cout<<"root seed"<<Globals::config.root_seed<<endl;
  return Globals::config.root_seed ^ Globals::config.n_particles;
}

int BeliefUpdateSeed() {
  return Globals::config.root_seed ^ (Globals::config.n_particles + 1);
}

// Action selection for RandomPolicyLowerBound (if used):
int RandomActionSeed() {
  return Globals::config.root_seed ^ (Globals::config.n_particles + 2);
}

WorldSimulator world;
WorldSimulator*RealWorldPt;

Solver<PedestrianState>* solver;
ILowerBound<PedestrianState>* lb;
BeliefUpdate<PedestrianState>* bu;



void controlLoop();
void initRealSimulator()
{
  Globals::config.n_belief_particles=2000;
  Globals::config.n_particles=500;
  RandomStreams streams(Globals::config.n_particles, Globals::config.search_depth, 
		                        Globals::config.root_seed);


  cout<<"global particle "<<Globals::config.n_particles<<endl;
  cout<<"root seed "<<Globals::config.root_seed<<endl;
  cout<<"search depth"<<Globals::config.search_depth<<endl;

  ifstream fin("despot.config");
  fin >> Globals::config.pruning_constant;
  cerr << "Pruning constant = " << Globals::config.pruning_constant << endl;

  Simulator  = new Model<PedestrianState>(streams, "pedestrian.config");
  double control_freq=ModelParams::control_freq;
  cout<<"control freq "<<control_freq<<endl;

  Simulator->control_freq=control_freq;
  RealWorldPt->control_freq=control_freq;
  RealWorldPt->control_time=1.0/control_freq;


  int knowledge = 2;
  lb = new RandomPolicyLowerBound<PedestrianState>(
		  streams, knowledge, RandomActionSeed());

  //IUpperBound<PedestrianState>* ub =
      //new UpperBoundStochastic<PedestrianState>(streams, *model);


  bu = new ParticleFilterUpdate<PedestrianState>(BeliefUpdateSeed(), *Simulator);

  // int ret = Run(model, lb, ub, bu, streams);
  VNode<PedestrianState>::set_model(*Simulator);
  Simulator->rob_map=RealWorldPt->window.rob_map;
  Simulator->sfm=&RealWorldPt->sfm;
  int action;
  int i;

  PedestrianState ped_state=RealWorldPt->GetCurrState();

  cout<<"start state"<<endl;
  Simulator->PrintState(ped_state);

  Simulator->SetStartState(ped_state);
  solver=new Solver<PedestrianState>(*Simulator, Simulator->InitialBelief(), *lb, *Simulator, *bu, streams);	
  solver->Init();
  controlLoop();


}




void controlLoop()
{
		int i;
		for(i=0;i<1000;i++)
		{

			int n_trials;
			int safeAction=solver->Search(Globals::config.time_per_move,n_trials);

			cout<<"action "<<safeAction<<endl;

			/*

			Car world_car;
			world_car.w=out_pose.getOrigin().getX()*ModelParams::map_rln;
			world_car.h=out_pose.getOrigin().getY()*ModelParams::map_rln;
			RealWorldPt->UpdateRobPoseReal(world_car);
			*/
			if(RealWorldPt->OneStep(safeAction)) break;

			if(ModelParams::debug)
			{
				cout<<"State before shift window"<<endl;
				Simulator->PrintState(RealWorldPt->GetCurrState());
			}
			RealWorldPt->ShiftWindow();


			cout<<"here"<<endl;
			//if(RealWorldPt->NumPedInView()==0) return;   //no pedestrian detected yet
			Simulator->rob_map=RealWorldPt->window.rob_map;
			PedestrianState new_state_old=RealWorldPt->GetCurrObs();

			cout<<"world observation: ";//<<obs<<endl;
			Simulator->PrintState(new_state_old);


			PedestrianState ped_state=RealWorldPt->GetCurrState();

			if(ModelParams::debug)
			{
				cout<<"current state"<<endl;
				Simulator->PrintState(ped_state);
			}

			cout<<"vel after action "<<ped_state.Vel<<endl;
			cout<<"vel global after action "<<world.velGlobal<<endl;
			cout<<"car pose after action "<<world.robPos<<endl;
			cout<<"car x y after action "<<world.car.w<<" "<<world.car.h<<endl;




			double reward;
			solver->UpdateBelief(safeAction, ped_state,new_state_old);


		}
		if(world.InCollision())
		{
			cout << "\nTotal  fail reward after " << i << " steps = "<<endl; 
		}
		else
			cout << "\nTotal success reward after " << i << " steps = "<<endl;
}


void Plan()
{
	cout<<"start planning"<<endl;
	//TestSimulator();
	Solver<PedestrianState>* solver=0;// = new Solver<PedestrianState>(*Simulator, Simulator->InitialBelief(), *lb_global, *ub_global, *bu_global, *streams_global);

	bool crashed=false;
	Simulator->rob_map=world.window.rob_map;
	Simulator->sfm=&world.sfm;
	int action;
	int i;

	PedestrianState ped_state=world.GetCurrState();
	if(ModelParams::debug)
	{
		cout<<"Initial State"<<endl;
		Simulator->PrintState(world.GetCurrState());
	}



	Simulator->SetStartState(ped_state);
	solver=new Solver<PedestrianState>(*Simulator, Simulator->InitialBelief(), *lb_global, *ub_global, *bu_global, *streams_global);	
	solver->Init();


	for(i=0;i<1000;i++)  
	{
		cout<<"Run "<<i<<endl;
		int n_trials;
		action=solver->Search(Globals::config.time_per_move,n_trials);
		cout<<"action "<<action<<endl;
		if(world.OneStep(action)) break;	
		world.ShiftWindow();
		if(ModelParams::debug)
		{
			cout<<"after shift window"<<endl;
			Simulator->PrintState(world.GetCurrState(),cout);
			world.Display();
		}
		PedestrianState obs=world.GetCurrObs();
		Simulator->rob_map=world.window.rob_map;
		/*
		if(world.NumPedInView()==0&&solver) 
		{
			cout<<"clean problem"<<endl;
			delete solver;
			solver=0;
		}*/
		if(solver)   {
			PedestrianState ped_state=world.GetCurrState();
			solver->UpdateBelief(action,ped_state,obs);	
			//cout<<"solver finished "<<solver->Finished()<<endl;
			//if(solver->Finished()) break;
		}
	}
	/*
	if(world.InCollision())
	{
		{
			cout << "\nTotal  fail reward after " << i << " steps = "<<endl; 
		}
		cout << "\nTotal  danger reward after " << i << " steps = "<<endl; 
	}
	else
		cout << "\nTotal success reward after " << i << " steps = "<<endl;
		*/
}


void disableBufferedIO(void)
{
    setbuf(stdout, NULL);
    setbuf(stdin, NULL);
    setbuf(stderr, NULL);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
}


/*
int main(int argc,char**argv)
{

	cout<<"start run loop"<<endl;
	       
	BuildPriorTable();
	
	LoadMap();
	for(int i=0;i<1;i++)
	{
		
		LoadPath(i);	
		Plan();
		break;
	}
	cout<<"exit program"<<endl;
	return 0;
}*/


enum optionIndex { 
  UNKNOWN, HELP, PROBLEM, PARAMS_FILE, DEPTH, DISCOUNT, SEED, TIMEOUT, 
  NPARTICLES, PRUNE, SIMLEN, LBTYPE, BELIEF, KNOWLEDGE, APPROX_BOUNDS, NUMBER
};

// option::Arg::Required is a misnomer. The program won't complain if these 
// are absent, and required flags must be checked manually.
const option::Descriptor usage[] = {
 {UNKNOWN,       0, "",  "",              option::Arg::None,     "USAGE: despot [options]\n\nOptions:"},
 {HELP,          0, "",  "help",          option::Arg::None,     "  --help   \tPrint usage and exit."},
 {PROBLEM,       0, "q", "problem",       option::Arg::Required, "  -q <arg> \t--problem=<arg>  \tProblem name."},
 {PARAMS_FILE,   0, "m", "model-params",  option::Arg::Required, "  -m <arg> \t--model-params=<arg>  \tPath to model-parameters file, if any."},
 {DEPTH,         0, "d", "depth",         option::Arg::Required, "  -d <arg> \t--depth=<arg>  \tMaximum depth of search tree (default 90)."},
 {DISCOUNT,      0, "g", "discount",      option::Arg::Required, "  -g <arg> \t--discount=<arg>  \tDiscount factor (default 0.95)."},
 {SEED,          0, "r", "seed",          option::Arg::Required, "  -r <arg> \t--seed=<arg>  \tRandom number seed (default 42)."},
 {TIMEOUT,       0, "t", "timeout",       option::Arg::Required, "  -t <arg> \t--timeout=<arg>  \tSearch time per move, in seconds (default 1)."},
 {NPARTICLES,    0, "n", "nparticles",    option::Arg::Required, "  -n <arg> \t--nparticles=<arg>  \tNumber of particles (default 500)."},
 {PRUNE,         0, "p", "prune",         option::Arg::Required, "  -p <arg> \t--prune=<arg>  \tPruning constant (default no pruning)."},
 {SIMLEN,        0, "s", "simlen",        option::Arg::Required, "  -s <arg> \t--simlen=<arg>  \tNumber of steps to simulate. (default 90; 0 = infinite)."},
 {LBTYPE,        0, "l", "lbtype",        option::Arg::Required, "  -l <arg> \t--lbtype=<arg>  \tLower bound strategy, if applicable."},
 {BELIEF,        0, "b", "belief",        option::Arg::Required, "  -b <arg> \t--belief=<arg>  \tBelief update strategy, if applicable."},
 {KNOWLEDGE,     0, "k", "knowledge",     option::Arg::Required, "  -k <arg> \t--knowledge=<arg>  \tKnowledge level for random lower bound policy, if applicable."},
 {NUMBER,     0, "", "number",     option::Arg::Required, "--number=<arg>  \tNumber of pedestrians."},
 {APPROX_BOUNDS, 0, "a", "approx-bounds", option::Arg::None,     "  -a \t--approx-bounds  \tWhether initial lower/upper bounds are approximate or true (default false)."},
 {0,0,0,0,0,0}
};


template<typename T>
int RunMultiple(Model<T>* model, ILowerBound<T>* lb, IUpperBound<T>* ub, 
        BeliefUpdate<T>* bu, const RandomStreams& streams) {
	int num_ped = Globals::config.number;
  	cout<<"start run loop"<<endl;
	       
	
  	VNode<T>::set_model(*Simulator);
	//TestSimulator();
	Plan();


  /*
  VNode<T>::set_model(*model);
  World<T> world = World<T>(Globals::config.root_seed ^ Globals::config.n_particles, *model);
	vector<T> states = model->GetStartStates(num_ped);
	world.SetStartStates(states);

 
  vector<Solver<T>*> solvers;
	for(int i=0; i<num_ped; i++) {
		model->SetStartState(states[i]);
		Solver<T>* solver = new Solver<T>(*model, model->InitialBelief(), *lb, *ub, *bu, streams);
		solver->Init();
		solvers.push_back(solver);
	}

	cout << "Number of particles = " << Globals::config.n_particles << endl;
  cout << "\nSTARTING STATE:\n";
	for(int i=0; i<num_ped; i++) {
		cout << "Pedestrian " << i << endl;
		model->PrintState(states[i]);
	}

  int total_trials = 0, step;
	int order[] = {1, 0, 2};
	bool crashed = false;

  for (step = 0; (Globals::config.sim_len == 0 || step < Globals::config.sim_len); step++) {
		cout << "\nSTEP " << step + 1 << endl;
		int optimal = 1;
		for(int i=0; i<num_ped; i++) {
			Solver<T>* solver = solvers[i];
			int n_trials;
			int act = solver->Search(Globals::config.time_per_move, n_trials);

			if(order[act] > order[optimal])
				optimal = act;
		}

		vector<double> rewards; 
		vector<uint64_t> obss;
		world.StepMultiple(optimal, obss, rewards);

		bool finished = false;
		for(int i=0;i <num_ped; i++) {
			solvers[i]->UpdateBelief(optimal, obss[i]);

			if(solvers[i]->Finished())
				finished = true;

			if(rewards[i] == -50000)
				crashed = true;

			if(finished==true) break;
		}

		if(finished)
			break;
  }
  if(crashed)
	  cout << "\nTotal  fail reward after " << step+1 << " steps = " << world.TotalReward()
       << endl;
  else
  	  cout << "\nTotal success reward after " << step+1 << " steps = " << world.TotalReward()
       << endl;
  cout << "Undiscounted reward after " << step << " steps = " << world.TotalUndiscountedReward()
       << endl;
  cerr << "Average # of trials per move = "
       << (step == 0 ? 0 : (double)total_trials / step) << endl;
 
	for(int i=0; i<Globals::config.number; i++)
		delete solvers[i];

  return 0;*/
}


template<typename T>
int Run(Model<T>* model, ILowerBound<T>* lb, IUpperBound<T>* ub, 
        BeliefUpdate<T>* bu, const RandomStreams& streams) {
  VNode<T>::set_model(*model);
  World<T> world = World<T>(Globals::config.root_seed ^ Globals::config.n_particles, *model);

  Solver<T>* solver = 
      new Solver<T>(*model, model->InitialBelief(), *lb, *ub, *bu, streams);
  solver->Init();

  cout << "\nSTARTING STATE:\n";
  model->PrintState(model->GetStartState());

  int total_trials = 0, step;
  double reward; uint64_t obs;
  for (step = 0; 
       !solver->Finished() && (Globals::config.sim_len == 0 || step < Globals::config.sim_len);
       step++) {
		cout << "\nSTEP " << step + 1 << endl;
    int n_trials = 0;
    int act = solver->Search(Globals::config.time_per_move, n_trials); // each solver returns an action
    total_trials += n_trials;

    world.Step(act, obs, reward); 
    solver->UpdateBelief(act, obs);
  }
  if(reward==-50000)
	  cout << "\nTotal  fail reward after " << step << " steps = " << world.TotalReward()
       << endl;
  else
  	  cout << "\nTotal success reward after " << step << " steps = " << world.TotalReward()
       << endl;
  cout << "Undiscounted reward after " << step << " steps = " << world.TotalUndiscountedReward()
       << endl;
  
  delete solver;

  return 0;
}

int RunPedestrian(option::Option* options, const RandomStreams& streams) {
  if (!options[PARAMS_FILE]) {
    cerr << "pedestrian requires a params file\n";
    return 1;
  }
  auto model = new Model<PedestrianState>(streams, options[PARAMS_FILE].arg);
  Simulator  = new Model<PedestrianState>(streams, options[PARAMS_FILE].arg);

  ILowerBound<PedestrianState>* lb;
  string lb_type = options[LBTYPE] ? options[LBTYPE].arg : "random";
  if (lb_type == "random") {
    int knowledge = options[KNOWLEDGE] ? atoi(options[KNOWLEDGE].arg) : 2;
    lb = new RandomPolicyLowerBound<PedestrianState>(
        streams, knowledge, RandomActionSeed());
  }
  else {
    cerr << "pedestrian requires lower bound of type 'random'\n";
    return 1;
  }

  //IUpperBound<PedestrianState>* ub =
      //new UpperBoundStochastic<PedestrianState>(streams, *model);

  string bu_type = options[BELIEF] ? options[BELIEF].arg : "particle";
  BeliefUpdate<PedestrianState>* bu;
  if (bu_type == "particle")
    bu = new ParticleFilterUpdate<PedestrianState>(BeliefUpdateSeed(), *Simulator);
  else {
    cerr << "pedestrian requires belief update strategy of type 'particle'\n";
    return 1;
  }

  // int ret = Run(model, lb, ub, bu, streams);
  lb_global=lb;
  ub_global=Simulator;
  bu_global=bu;
  
  int ret = RunMultiple(model, lb, model, bu, streams);

  delete model;
  delete lb;
  // delete ub;
  delete bu;

  return ret;
}

int main(int argc, char* argv[]) {
  argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
  option::Stats stats(usage, argc, argv);
  option::Option* options = new option::Option[stats.options_max];
  option::Option* buffer = new option::Option[stats.buffer_max];
  option::Parser parse(usage, argc, argv, options, buffer);

  // Required parameters
  string problem;
  if (!options[PROBLEM]) {
    option::printUsage(std::cout, usage);
    return 0;
  }
  problem = options[PROBLEM].arg;

  // Optional parameters
  if (options[DEPTH]) Globals::config.search_depth = atoi(options[DEPTH].arg);
  if (options[DISCOUNT]) Globals::config.discount = atof(options[DISCOUNT].arg);
  if (options[SEED]) Globals::config.root_seed = atoi(options[SEED].arg);
  if (options[TIMEOUT]) Globals::config.time_per_move = atof(options[TIMEOUT].arg);
  if (options[NPARTICLES]) Globals::config.n_particles = atoi(options[NPARTICLES].arg);
  if (options[PRUNE]) Globals::config.pruning_constant = atof(options[PRUNE].arg);
  if (options[SIMLEN]) Globals::config.sim_len = atoi(options[SIMLEN].arg);
  if (options[APPROX_BOUNDS]) Globals::config.approximate_bounds = true;
	if (options[NUMBER]) Globals::config.number = atoi(options[NUMBER].arg);


	/*
  RandomStreams streams(Globals::config.n_particles, Globals::config.search_depth, 
                        Globals::config.root_seed);

  cout<<"option seed "<<Globals::config.root_seed<<endl;

  streams_global=&streams;
  */
  world.SetSeed(WorldSeed());
  cout<<"world seed "<<WorldSeed()<<endl;
  world.NumPedTotal=Globals::config.number;
  world.Init();
  RealWorldPt=&world;

  /*
  if (problem == "pedestrian")
    return RunPedestrian(options, streams);
  else
	  cout << "Problem must be one of tag, lasertag, rocksample, tiger, bridge "
          "and pocman.\n";*/

  initRealSimulator();

  return 1;
}
