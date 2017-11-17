#include <despot/simple_tui.h>

using namespace std;

namespace despot {

void disableBufferedIO(void) {
  setbuf(stdout, NULL);
  setbuf(stdin, NULL);
  setbuf(stderr, NULL);
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stdin, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
}

SimpleTUI::SimpleTUI() {

	//Globals::config.useGPU=true;
}

SimpleTUI::~SimpleTUI() {}

Solver *SimpleTUI::InitializeSolver(DSPOMDP *model, string solver_type,
                                    option::Option *options) {
  Solver *solver = NULL;
  // DESPOT or its default policy
  if (solver_type == "DESPOT" ||
      solver_type == "PLB") // PLB: particle lower bound
  {
    string blbtype = options[E_BLBTYPE] ? options[E_BLBTYPE].arg : "DEFAULT";
    string lbtype = options[E_LBTYPE] ? options[E_LBTYPE].arg : "DEFAULT";
    ScenarioLowerBound *lower_bound =
        model->CreateScenarioLowerBound(lbtype, blbtype);

    logi << "Created lower bound " << typeid(*lower_bound).name() << endl;

    if (solver_type == "DESPOT") {
      string bubtype = options[E_BUBTYPE] ? options[E_BUBTYPE].arg : "DEFAULT";
      string ubtype = options[E_UBTYPE] ? options[E_UBTYPE].arg : "DEFAULT";
      ScenarioUpperBound *upper_bound =
          model->CreateScenarioUpperBound(ubtype, bubtype);

      logi << "Created upper bound " << typeid(*upper_bound).name() << endl;

      solver = new DESPOT(model, lower_bound, upper_bound,NULL,Globals::config.useGPU);
    } else
      solver = lower_bound;
  } // AEMS or its default policy
  else if (solver_type == "AEMS" || solver_type == "BLB") {
    string lbtype = options[E_LBTYPE] ? options[E_LBTYPE].arg : "DEFAULT";
    BeliefLowerBound *lower_bound =
        static_cast<BeliefMDP *>(model)->CreateBeliefLowerBound(lbtype);

    logi << "Created lower bound " << typeid(*lower_bound).name() << endl;

    if (solver_type == "AEMS") {
      string ubtype = options[E_UBTYPE] ? options[E_UBTYPE].arg : "DEFAULT";
      BeliefUpperBound *upper_bound =
          static_cast<BeliefMDP *>(model)->CreateBeliefUpperBound(ubtype);

      logi << "Created upper bound " << typeid(*upper_bound).name() << endl;

      solver = new AEMS(model, lower_bound, upper_bound);
    } else
      solver = lower_bound;
  } // POMCP or DPOMCP
  else if (solver_type == "POMCP" || solver_type == "DPOMCP") {
    string ptype = options[E_PRIOR] ? options[E_PRIOR].arg : "DEFAULT";
    POMCPPrior *prior = model->CreatePOMCPPrior(ptype);

    logi << "Created POMCP prior " << typeid(*prior).name() << endl;

    if (options[E_PRUNE]) {
      prior->exploration_constant(Globals::config.pruning_constant);
    }

    if (solver_type == "POMCP")
      solver = new POMCP(model, prior);
    else
      solver = new DPOMCP(model, prior);
  } else { // Unsupported solver
    cerr << "ERROR: Unsupported solver type: " << solver_type << endl;
    exit(1);
  }
  return solver;
}

void SimpleTUI::OptionParse(option::Option *options, int &num_runs,
                            string &simulator_type, string &belief_type,
                            int &time_limit, string &solver_type,
                            bool &search_solver, bool& use_GPU) {
  if (options[E_SILENCE])
    Globals::config.silence = true;

  if (options[E_DEPTH])
    Globals::config.search_depth = atoi(options[E_DEPTH].arg);

  if (options[E_DISCOUNT])
    Globals::config.discount = atof(options[E_DISCOUNT].arg);

  if (options[E_SEED])
    Globals::config.root_seed = atoi(options[E_SEED].arg);
  else { // last 9 digits of current time in milli second
    long millis = (long)get_time_second() * 1000;
    long range = (long)pow((double)10, (int)9);
    Globals::config.root_seed =
        (unsigned int)(millis - (millis / range) * range);
  }

  if (options[E_TIMEOUT])
    Globals::config.time_per_move = atof(options[E_TIMEOUT].arg);

  if (options[E_NUMPARTICLES])
    Globals::config.num_scenarios = atoi(options[E_NUMPARTICLES].arg);

  if (options[E_PRUNE])
    Globals::config.pruning_constant = atof(options[E_PRUNE].arg);

  if (options[E_GAP])
    Globals::config.xi = atof(options[E_GAP].arg);

  if (options[E_SIM_LEN])
    Globals::config.sim_len = atoi(options[E_SIM_LEN].arg);

  if (options[E_EVALUATOR])
    simulator_type = options[E_EVALUATOR].arg;

  if (options[E_MAX_POLICY_SIM_LEN])
    Globals::config.max_policy_sim_len =
        atoi(options[E_MAX_POLICY_SIM_LEN].arg);

  if (options[E_DEFAULT_ACTION])
    Globals::config.default_action = options[E_DEFAULT_ACTION].arg;

  if (options[E_RUNS])
    num_runs = atoi(options[E_RUNS].arg);

  if (options[E_BELIEF])
    belief_type = options[E_BELIEF].arg;

  if (options[E_TIME_LIMIT])
    time_limit = atoi(options[E_TIME_LIMIT].arg);

  if (options[E_NOISE])
    Globals::config.noise = atof(options[E_NOISE].arg);

  search_solver = options[E_SEARCH_SOLVER];

  if (options[E_SOLVER])
    solver_type = options[E_SOLVER].arg;

  int verbosity = 0;
  if (options[E_VERBOSITY])
    verbosity = atoi(options[E_VERBOSITY].arg);
  logging::level(verbosity);

  if (options[E_GPU])
  {
	Globals::config.useGPU = atoi(options[E_GPU].arg);
    cout<<"use GPU: "<<Globals::config.useGPU<<endl;
  }

  if (options[E_GPUID])
  {
  	Globals::config.GPUid = atoi(options[E_GPUID].arg);
      cout<<"GPU ID: "<<Globals::config.GPUid<<endl;
  }
  if (options[E_CPU])
  {
	Globals::config.use_multi_thread_ = atoi(options[E_CPU].arg);
	  cout<<"use multi_thread: "<<Globals::config.use_multi_thread_<<endl;
  }

  if (options[E_THREAD_NUM])
  {
	Globals::config.NUM_THREADS = atoi(options[E_THREAD_NUM].arg);
	cout<<"number of CPU threads: "<<Globals::config.NUM_THREADS<<endl;
  }

  if (options[E_EXPLORATION])
  {
	  if(options[E_EXPLORATION].arg=="Vloss")
		  Globals::config.exploration_mode = VIRTUAL_LOSS;
	  else if(options[E_EXPLORATION].arg=="UCT")
		  Globals::config.exploration_mode = UCT;
	  cout<<"exploration scheme (0 virtual loss; 1 UCT bound): "<<Globals::config.exploration_mode<<endl;
  }

  if (options[E_EXPLORE_CONST])
  {
	  Globals::config.exploration_constant = atof(options[E_EXPLORE_CONST].arg);
	  cout<<"exploration constant: "<<Globals::config.exploration_constant<<endl;
  }

/*  if (options[E_ROLLOUT])
  {
	Globals::config.rollout_type = options[E_ROLLOUT].arg;
	cout<<"Roll-out type: "<<Globals::config.rollout_type<<endl;
  }*/
}

void SimpleTUI::InitializeEvaluator(Evaluator *&simulator,
                                    option::Option *options, DSPOMDP *model,
                                    Solver *solver, int num_runs,
                                    clock_t main_clock_start,
                                    string simulator_type, string belief_type,
                                    int time_limit, string solver_type) {

  if (time_limit != -1) {
    simulator =
        new POMDPEvaluator(model, belief_type, solver, main_clock_start, &cout,
                           EvalLog::curr_inst_start_time + time_limit,
                           num_runs * Globals::config.sim_len);
  } else {
    simulator =
        new POMDPEvaluator(model, belief_type, solver, main_clock_start, &cout);
  }
}

void SimpleTUI::DisplayParameters(option::Option *options, DSPOMDP *model) {

  string lbtype = options[E_LBTYPE] ? options[E_LBTYPE].arg : "DEFAULT";
  string ubtype = options[E_UBTYPE] ? options[E_UBTYPE].arg : "DEFAULT";
  default_out << "Model = " << typeid(*model).name() << endl
              << "Random root seed = " << Globals::config.root_seed << endl
              << "Search depth = " << Globals::config.search_depth << endl
              << "Discount = " << Globals::config.discount << endl
              << "Simulation steps = " << Globals::config.sim_len << endl
              << "Number of scenarios = " << Globals::config.num_scenarios
              << endl
              << "Search time per step = " << Globals::config.time_per_move
              << endl
              << "Regularization constant = "
              << Globals::config.pruning_constant << endl
              << "Lower bound = " << lbtype << endl
              << "Upper bound = " << ubtype << endl
              << "Policy simulation depth = "
              << Globals::config.max_policy_sim_len << endl
              << "Target gap ratio = " << Globals::config.xi << endl;
  // << "Solver = " << typeid(*solver).name() << endl << endl;
}

void SimpleTUI::RunEvaluator(DSPOMDP *model, Evaluator *simulator,
                             option::Option *options, int num_runs,
                             bool search_solver, Solver *&solver,
                             string simulator_type, clock_t main_clock_start,
                             int start_run) {
  // Run num_runs simulations
  vector<double> round_rewards(num_runs);

  for (int round = start_run; round < start_run + num_runs; round++) {
    default_out << endl
                << "####################################### Round " << round
                << " #######################################" << endl;

    if (search_solver) {
      if (round == 0) {
        solver = InitializeSolver(model, "DESPOT", options);
        default_out << "Solver: " << typeid(*solver).name() << endl;

        simulator->solver(solver);
      } else if (round == 5) {
        solver = InitializeSolver(model, "POMCP", options);
        default_out << "Solver: " << typeid(*solver).name() << endl;

        simulator->solver(solver);
      } else if (round == 10) {
        double sum1 = 0, sum2 = 0;
        for (int i = 0; i < 5; i++)
          sum1 += round_rewards[i];
        for (int i = 5; i < 10; i++)
          sum2 += round_rewards[i];
        if (sum1 < sum2)
          solver = InitializeSolver(model, "POMCP", options);
        else
          solver = InitializeSolver(model, "DESPOT", options);
        default_out << "Solver: " << typeid(*solver).name()
                    << " DESPOT:" << sum1 << " POMCP:" << sum2 << endl;
      }

      simulator->solver(solver);
    }

    simulator->InitRound();
    bool terminal=false;
    for (int i = 0; i < Globals::config.sim_len; i++) {
      /*
      default_out << "-----------------------------------Round " << round
                  << " Step " << i << "-----------------------------------"
                  << endl;*/
      double step_start_t = get_time_second();

      terminal = simulator->RunStep(i, round);

      if (terminal)
        break;

      double step_end_t = get_time_second();
      logi << "[main] Time for step: actual / allocated = "
           << (step_end_t - step_start_t) << " / " << EvalLog::allocated_time
           << endl;
      simulator->UpdateTimePerMove(step_end_t - step_start_t);
      logi << "[main] Time per move set to " << Globals::config.time_per_move
           << endl;
      logi << "[main] Plan time ratio set to " << EvalLog::plan_time_ratio
           << endl;
    //  default_out << endl;
    }

    default_out << "Simulation terminated in " << simulator->step() << " steps"
                << endl;
    double round_reward = simulator->EndRound(terminal);
    round_rewards[round] = round_reward;
  }

  if (simulator_type == "ippc" && num_runs != 30) {
    cout << "Exit without receiving reward." << endl
         << "Total time: Real / CPU = "
         << (get_time_second() - EvalLog::curr_inst_start_time) << " / "
         << (double(clock() - main_clock_start) / CLOCKS_PER_SEC) << "s"
         << endl;
    exit(0);
  }
}

void SimpleTUI::PrintResult(int num_runs, Evaluator *simulator,
                            clock_t main_clock_start) {
  ios::fmtflags old_settings = cout.flags();

  //int Width=7;
  int Prec=3;
  cout.precision(Prec);
  cout << "\nCompleted " << num_runs << " run(s)." << endl;
  cout.precision(Prec);
  cout << "Average total discounted reward (stderr) = "
       << simulator->AverageDiscountedRoundReward() << " ( "
       << simulator->StderrDiscountedRoundReward() << " )" << endl;
  cout.precision(Prec);
  cout << "Average total undiscounted reward (stderr) = "
       << simulator->AverageUndiscountedRoundReward() << " ( "
       << simulator->StderrUndiscountedRoundReward() << " )" << endl;
  cout.precision(Prec);
  cout << "Success rate = "
  	   << simulator->SuccessRate() << endl;
  cout.precision(5);
  cout << "Total time: Real / CPU = "
       << (get_time_second() - EvalLog::curr_inst_start_time) << " / "
       << (double(clock() - main_clock_start) / CLOCKS_PER_SEC) << " s" << endl;
  cout.precision(Prec);
  cout << "Action frequency= ";
  int path_length=0;
  for(int action =0;action<simulator->model()->NumActions();action++)
  {
	 cout<< simulator->AverageActionFrequency(action) << " ";
  	 path_length+=simulator->AverageActionFrequency(action);
  }
  cout<<endl;
  cout.precision(Prec);
  cout << "Average path length = "<< path_length << endl;
  cout<<endl;
  simulator->PrintStatisticResult();
  cout.flags(old_settings);
}

void SimpleTUI::ReleaseGPUData(DSPOMDP* model, Solver* solver) {
	if (Globals::config.useGPU) {
		model->DeleteGPUParticles(Globals::config.num_scenarios);
		DeleteGPUModel();
		DeleteGPUGlobals();
		if (Globals::config.rollout_type == "GRAPH")
			DeleteGPUPolicyGraph();

		solver->clearGPUHistory();
		DestroyRandGen();
	}
}

void SimpleTUI::PrepareGPUData(DSPOMDP* model, Solver* solver) {
	if (Globals::config.useGPU) {
		//logging::level(4);
		InitializedGPUModel(Globals::config.rollout_type, model); //GPU Panpan
		InitializeGPUGlobals();
		model->AllocGPUParticles(Globals::config.num_scenarios, 0);
		solver->initGPUHistory();
	}
	if (Globals::config.useGPU && Globals::config.rollout_type == "GRAPH") {
		InitializeGPUPolicyGraph(
				(PolicyGraph*) (((DESPOT*) (solver))->lower_bound()));
	}
}

void SimpleTUI::PrepareGPU() {
	if (Globals::config.useGPU) {
		SetUpCPUThreads(Globals::config.use_multi_thread_,
				Globals::config.NUM_THREADS);
		SetupGPU();
		InitRandGen();
	}
	if(use_multi_thread_)
		record=new unsigned long long int[Globals::config.NUM_THREADS];
	else
		record=new unsigned long long int;
}

int SimpleTUI::run(int argc, char *argv[]) {

  clock_t main_clock_start = clock();
  EvalLog::curr_inst_start_time = get_time_second();

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
  /* =========================
   * Global random generator
   * =========================*/
  Seeds::root_seed(Globals::config.root_seed);
  unsigned world_seed = Seeds::Next();
  unsigned seed = Seeds::Next();
  Random::RANDOM = Random(seed);

  /* =========================
   * initialize model
   * =========================*/
  DSPOMDP *model = InitializeModel(options);

  /* =========================
   * initialize solver
   * =========================*/
  Solver *solver = InitializeSolver(model, solver_type, options);
  assert(solver != NULL);
  PrepareGPUData(model, solver);
  /* =========================
   * initialize simulator
   * =========================*/
  Evaluator *simulator = NULL;
  InitializeEvaluator(simulator, options, model, solver, num_runs,
                      main_clock_start, simulator_type, belief_type, time_limit,
                      solver_type);
  simulator->world_seed(world_seed);
  //simulator->useGPU_=useGPU_;
  int start_run = 0;

  /* =========================
   * Display parameters
   * =========================*/
  DisplayParameters(options, model);

  /* =========================
   * run simulator
   * =========================*/
  RunEvaluator(model, simulator, options, num_runs, search_solver, solver,
               simulator_type, main_clock_start, start_run);

  simulator->End();

  PrintResult(num_runs, simulator, main_clock_start);

  ReleaseGPUData(model, solver);

  return 0;
}


} // namespace despot
