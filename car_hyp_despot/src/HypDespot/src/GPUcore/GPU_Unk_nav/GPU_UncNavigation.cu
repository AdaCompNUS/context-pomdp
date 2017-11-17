#include "GPU_UncNavigation.h"

#include "GPU_base_unc_navigation.h"
#include <base/base_unc_navigation.h>

using namespace std;

namespace despot {
static int NumObstacles;
static bool NewRound=true;
static DvcCoord* obstacle_pos;
//const DvcCoord Dvc_NavCompass::DIRECTIONS[] = {  DvcCoord(0, 1), DvcCoord(1, 0),DvcCoord(0, -1),
//	DvcCoord(-1, 0), DvcCoord(1, 1), DvcCoord(1, -1), DvcCoord(-1, -1), DvcCoord(-1, 1) };
//const string Dvc_NavCompass::CompassString[] = { "North", "East","South", "West",
//	"NE", "SE", "SW", "NW" };


/* =============================================================================
 * Dvc_UncNavigation class
 * =============================================================================*/

/*Dvc_UncNavigation::Dvc_UncNavigation(string map) :
	Dvc_UncNavigation(map) {
	half_efficiency_distance_ = 20;
}*/


DEVICE Dvc_UncNavigation::Dvc_UncNavigation(/*int size, int obstacles*/)// :
	//grid_(size, size),
	//size_(size),
	//num_obstacles_(obstacles)
{
	/*if (size == 4 && obstacles == 4) {
		Init_4_4();
	} else if (size == 5 && obstacles == 5) {
		Init_5_5();
	} else if (size == 5 && obstacles == 7) {
		Init_5_7();
	} else if (size == 7 && obstacles == 8) {
		Init_7_8();
	} else if (size == 11 && obstacles == 11) {
		Init_11_11();
	} else {
		InitGeneral();
	}*/
	// put obstacles in random positions
	//InitGeneral();
	// InitStates();

}


DEVICE Dvc_UncNavigation::~Dvc_UncNavigation()
{
	//half_efficiency_distance_ = 20;

}

/*__device__ float Obs_Prob_[5][500][16];


DEVICE bool Dvc_UncNavigation::Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
	OBS_TYPE& obs) {

	Dvc_UncNavigationState& nav_state = static_cast<Dvc_UncNavigationState&>(state);//copy contents, link cells to existing ones
	bool terminal=false;
	reward = 0;

	int obs_i=threadIdx.x;

	if(obs_i==0)
	{
		terminal=(nav_state.rob==nav_state.goal);

		reward=-0.1;// small cost for one step
		DvcCoord rob_pos=nav_state.rob;

		float prob=1.0f-STEP_NOISE;

		if (action < E_STAY && terminal!=true) { // Move
			rob_pos +=(rand_num<prob)? Dvc_Compass::GetDirections(action):DvcCoord(0,0);

			bool validmove=(nav_state.Inside(rob_pos) && nav_state.CollisionCheck(rob_pos)==false);

			nav_state.rob=validmove?rob_pos:nav_state.rob;
			reward=validmove?-0.1:-1;
			reward=(nav_state.rob==nav_state.goal)?GOAL_REWARD:reward;
		}

		if (action == E_STAY) { // Sample
			reward=-0.1;
		}
	}
		obs=Dvc_NumObservations()-1;
		__shared__ float Obs_Prob_1[8][16];
		__syncthreads();

			float prob=1;

			prob*=((obs_i&8)>>3==nav_state.Grid(nav_state.rob.x,nav_state.rob.y+1))?1-OBS_NOISE:OBS_NOISE;
			prob*=((obs_i&4)>>2==nav_state.Grid(nav_state.rob.x+1,nav_state.rob.y))?1-OBS_NOISE:OBS_NOISE;
			prob*=((obs_i&2)>>1==nav_state.Grid(nav_state.rob.x,nav_state.rob.y-1))?1-OBS_NOISE:OBS_NOISE;
			prob*=((obs_i&1)==nav_state.Grid(nav_state.rob.x-1,nav_state.rob.y))?1-OBS_NOISE:OBS_NOISE;

			Obs_Prob_1[threadIdx.y][obs_i]=prob;//Debug, test alignment, no enough cache space to use

		__syncthreads();

		prob=0;
		if(obs_i==0)
		{
			for(int i=0;i<Dvc_NumObservations();i++)//pick an obs according to the prob of each one
			{
				prob+=Obs_Prob_1[threadIdx.y][i];
				if(rand_num<=prob)
				{	obs=i;	break;	}
			}
		}

	if(obs_i==0)
	{
		if(terminal){reward=0;}
	}

	return terminal;//Debug,test time
}*/
static
__device__ float RandGeneration(unsigned long long int *record, float seed)
{
	//float value between 0 and 1
	//seed have to be within 0 and 1
	record[0]+=seed*ULLONG_MAX/1000.0;
	//unsigned long long int record_l=atomicAdd(record,t);//atomic add returns old value of record

	//record_l+=t;
	float record_f=0;
	/*randMethod(record_l,record_f);*/
	record[0]*=16807;
	record[0]=record[0]%2147483647;
	record_f=((double)record[0])/2147483647;
	return record_f;
}
DEVICE bool Dvc_UncNavigation::Dvc_Step(Dvc_State& state, float rand_num, int action, float& reward,
	OBS_TYPE& obs) {

	Dvc_UncNavigationState& nav_state = static_cast<Dvc_UncNavigationState&>(state);//copy contents, link cells to existing ones
	bool terminal=false;
	reward = 0;

	int dir=threadIdx.y;

	if(dir==0)
	{
		terminal=(nav_state.rob==nav_state.goal);

		reward=-0.1;// small cost for one step
		DvcCoord rob_pos=nav_state.rob;

		float prob=1.0f-STEP_NOISE;

		if (action < E_STAY && terminal!=true) { // Move
			// only succeed with 80% chance
			rob_pos +=(rand_num<prob)? Dvc_Compass::GetDirections(action):DvcCoord(0,0);

			bool validmove=(nav_state.Inside(rob_pos) && nav_state.CollisionCheck(rob_pos)==false);

			nav_state.rob=validmove?rob_pos:nav_state.rob;
			reward=validmove?-0.1:-1;
			reward=(nav_state.rob==nav_state.goal)?/*10*/GOAL_REWARD:reward;
		}

		if (action == E_STAY) { // Sample
			reward=-0.2;
		}

		obs=0;//Initialize obs
	}


		//__shared__ float Obs_Prob_1[8][4];
		//__syncthreads();

		OBS_TYPE obs_i=0;
		//int pos=nav_state.scenario_id;

		unsigned long long int Temp=123456;
		for(dir=0;dir<4;dir++)
		{
			switch(dir)
			{
			case 3:
				rand_num=RandGeneration(&Temp, rand_num);
				obs_i=(rand_num<1-OBS_NOISE)?nav_state.Grid(nav_state.rob.x,nav_state.rob.y+1):!nav_state.Grid(nav_state.rob.x,nav_state.rob.y+1);
				break;
			case 2:
				rand_num=RandGeneration(&Temp, rand_num);
				obs_i=(rand_num<1-OBS_NOISE)?nav_state.Grid(nav_state.rob.x+1,nav_state.rob.y):!nav_state.Grid(nav_state.rob.x+1,nav_state.rob.y);
				break;
			case 1:
				rand_num=RandGeneration(&Temp, rand_num);
				obs_i=(rand_num<1-OBS_NOISE)?nav_state.Grid(nav_state.rob.x,nav_state.rob.y-1):!nav_state.Grid(nav_state.rob.x,nav_state.rob.y-1);
				break;
			case 0:
				rand_num=RandGeneration(&Temp, rand_num);
				obs_i=(rand_num<1-OBS_NOISE)?nav_state.Grid(nav_state.rob.x-1,nav_state.rob.y):!nav_state.Grid(nav_state.rob.x-1,nav_state.rob.y);
				break;
			}
			obs=(obs|(obs_i<<dir));

			//printf("thread.x=%d, rand_num=%f, obs=%d\n",threadIdx.x,rand_num, obs);
		}
		//atomicOr((unsigned int*)(&obs), (unsigned int)(obs_i<<dir));
		//__syncthreads();

		if(obs>=Dvc_NumObservations())
			printf("Wrong obs %d", obs);
	if(threadIdx.y==0)
	{
		if(terminal){reward=0;obs=Dvc_NumObservations()-1;}
	}
	return terminal/*temp*/;//Debug,test time
}

DEVICE int Dvc_UncNavigation::NumActions() const {
	return /*5*/9;
}
/*DEVICE void Dvc_UncNavigation::TestObsProb(const Dvc_State& state) const
{
	double totalprob=0;
	for (int obs=0;obs<Dvc_NumObservations();obs++)
	{
		double prob=ObsProb(obs, state,E_STAY);
		logi<<"ObsProb "<<obs<<"="<<prob<<endl;
		totalprob+=prob;
	}
	logi<<"TotalProb="<<totalprob<<endl;
}*/

DEVICE float Dvc_UncNavigation::ObsProb(OBS_TYPE obs, const Dvc_State& state, int action) {
	//float temp=0;
	//clock_t start_time = clock();

	float prob=1;
	const Dvc_UncNavigationState* nav_state=static_cast<const Dvc_UncNavigationState*>(&state);

	int obs_North=obs/8;
	int obs_East=(obs-obs_North*8)/4;
	int obs_South=(obs-obs_North*8-obs_East*4)/2;
	int obs_West=(obs-obs_North*8-obs_East*4-obs_South*2);

	//PrintObs(state, obs,cout);
	//logi<<"Refracted as:"<< obs_North << obs_East <<obs_South<<obs_West<<endl;

	int truth_North,truth_East,truth_South,truth_West;
	truth_North=nav_state->Grid(nav_state->rob+Dvc_Compass::GetDirections(E_NORTH));
	truth_East=nav_state->Grid(nav_state->rob+Dvc_Compass::GetDirections(E_EAST));
	truth_South=nav_state->Grid(nav_state->rob+Dvc_Compass::GetDirections(E_SOUTH));
	truth_West=nav_state->Grid(nav_state->rob+Dvc_Compass::GetDirections(E_WEST));

	float Noise=OBS_NOISE;
	prob*=(obs_North==truth_North)?1-Noise:Noise;
	prob*=(obs_East==truth_East)?1-Noise:Noise;
	prob*=(obs_South==truth_South)?1-Noise:Noise;
	prob*=(obs_West==truth_West)?1-Noise:Noise;

	//clock_t stop_time = clock();

	//float runtime= (double)(stop_time - start_time)/1733500.0;
	//runtime++;
	//temp=runtime;

	return prob;
}

/*
DEVICE void Dvc_UncNavigation::PrintObs(const Dvc_State& state, OBS_TYPE observation,
	ostream& out) const {
	switch (observation) {
	case E_FN_FE_FS_FW:
		out << "N0 E0 S0 W0" << endl;
		break;
	case E_FN_FE_FS_OW:
		out << "N0 E0 S0 W1" << endl;
		break;
	case E_FN_FE_OS_FW:
		out << "N0 E0 S1 W0" << endl;
		break;
	case E_FN_FE_OS_OW:
		out << "N0 E0 S1 W1" << endl;
		break;
	case E_FN_OE_FS_FW:
		out << "N0 E1 S0 W0" << endl;
		break;
	case E_FN_OE_FS_OW:
		out << "N0 E1 S0 W1" << endl;
		break;
	case E_FN_OE_OS_FW:
		out << "N0 E1 S1 W0" << endl;
		break;
	case E_FN_OE_OS_OW:
		out << "N0 E1 S1 W1" << endl;
		break;
	case E_ON_FE_FS_FW:
		out << "N1 E0 S0 W0" << endl;
		break;
	case E_ON_FE_FS_OW:
		out << "N1 E0 S0 W1" << endl;
		break;
	case E_ON_FE_OS_FW:
		out << "N1 E0 S1 W0" << endl;
		break;
	case E_ON_FE_OS_OW:
		out << "N1 E0 S1 W1" << endl;
		break;
	case E_ON_OE_FS_FW:
		out << "N1 E1 S0 W0" << endl;
		break;
	case E_ON_OE_FS_OW:
		out << "N1 E1 S0 W1" << endl;
		break;
	case E_ON_OE_OS_FW:
		out << "N1 E1 S1 W0" << endl;
		break;
	case E_ON_OE_OS_OW:
		out << "N1 E1 S1 W1" << endl;
		break;
	}
}
*/



/*

DEVICE void Dvc_UncNavigation::RandGate(Dvc_UncNavigationState* nav_state) const
{
   DvcCoord pos;
   pos=nav_state->GateNorth();
   nav_state->GridOpen(pos) = (bool)Random::RANDOM.NextInt(2); // randomly put obstacles there
   pos=nav_state->GateEast();
   nav_state->GridOpen(pos) = (bool)Random::RANDOM.NextInt(2); // randomly put obstacles there
   pos=nav_state->GateWest();
   nav_state->GridOpen(pos) = (bool)Random::RANDOM.NextInt(2); // randomly put obstacles there
}

DEVICE void Dvc_UncNavigation::RandMap(Dvc_UncNavigationState* nav_state, float ObstacleProb, int skip) const
{
	//assign obstacle with prob ObstacleProb at 1/skip of the map
	for (int x=0;x<nav_state->sizeX_;x+=skip)
		for (int y=0;y<nav_state->sizeY_;y+=skip)
		{
			DvcCoord pos(x,y);
			if(nav_state->Grid(pos)==false
					&& pos!=nav_state->goal )//ignore existing obstacles
				nav_state->GridOpen(pos)=Random::RANDOM.NextDouble()<ObstacleProb? true:false;
		}
}
DEVICE void Dvc_UncNavigation::RandMap(Dvc_UncNavigationState* nav_state) const
{
	for(int i=0;i<NumObstacles;i++)
	{
		DvcCoord pos=obstacle_pos[i];
		nav_state->GridOpen(pos)=(bool)Random::RANDOM.NextInt(2);//put obstacle there
	}
}


DEVICE void Dvc_UncNavigation::CalObstacles(float prob) const
{
	NumObstacles=prob*(float)(size_*size_);
	if(NumObstacles>0)
	{
		obstacle_pos=new DvcCoord[NumObstacles];
		int ExistingObs=0;
		//allocate a temporary state
		Dvc_UncNavigationState* nav_state = Dvc_memory_pool_.Allocate();
		nav_state->InitCells(size_,size_);
		//put the goal first
		nav_state->FixedGoal();
		//generate obstacles
		DvcCoord pos;
		do {
			do {
				pos = DvcCoord(Random::RANDOM.NextInt(size_),
					Random::RANDOM.NextInt(size_));
			} while (nav_state->Grid(pos) == true || pos==nav_state->goal);// check for random free map position
			nav_state->GridOpen(pos)=true;//put obstacle there
			obstacle_pos[ExistingObs]=pos;
			ExistingObs++;
		}while(ExistingObs<NumObstacles);
	}
	else
		obstacle_pos=NULL;
}

DEVICE void Dvc_UncNavigation::FreeObstacles() const
{
	delete [] obstacle_pos;
}

DEVICE Dvc_State* Dvc_UncNavigation::CreateStartState(string type) const {
	if(NewRound)
	{
		CalObstacles(0.0);
		NewRound=false;
	}
	//Dvc_UncNavigationState state(size_, size_);
	Dvc_UncNavigationState* startState = Dvc_memory_pool_.Allocate();
	startState->InitCells(size_,size_);
	//put the goal first
	startState->FixedGoal();
	// put obstacles in fixed positions
	DvcCoord pos;
	if (num_obstacles_>0)
	{
		pos.x=size_/4; pos.y=3*size_/4;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	if (num_obstacles_>1)
	{
		pos.x=2*size_/4; pos.y=2*size_/4;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	if (num_obstacles_>2)
	{
		pos.x=size_/2-2; pos.y=0;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	if (num_obstacles_>3)
	{
		pos.x=0; pos.y=1*size_/4;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	if (num_obstacles_>4)
	{
		pos.x=size_-2; pos.y=1*size_/4+1;
		startState->GridOpen(pos) = true; // put the obstacle there
	}
	//RandGate(startState);
	RandMap(startState);//Generate map using the obstacle positions specified in obstacle_pos

	// pick a random position for the robot, always do this in the last step
	UniformRobPos(startState);
	//AreaRobPos(startState,2);
	return startState;
}
DEVICE void Dvc_UncNavigation::UniformRobPos(Dvc_UncNavigationState* startState) const
{
	DvcCoord pos;
	do {
		pos = DvcCoord(Random::RANDOM.NextInt(size_),
			Random::RANDOM.NextInt(size_));
	} while (startState->Grid(pos) == true || pos==startState->goal);// check for random free map position
	startState->rob=pos;//put robot there
}
DEVICE void Dvc_UncNavigation::AreaRobPos(Dvc_UncNavigationState* startState, int area_size) const
{
	// robot start from top middle block of area_size
	DvcCoord Robpos(size_/2-area_size/2-1,size_-area_size);
	DvcCoord pos;

	bool has_slot=false;
	for(int x=Robpos.x;x<Robpos.x+area_size;x++)
		for(int y=Robpos.y;y<Robpos.y+area_size;y++)
		{
			if(startState->Grid(x,y) == false)
			{
				has_slot=true;
				break;
			}
		}

	if(has_slot)
	{
		do {
			pos = Robpos+DvcCoord(Random::RANDOM.NextInt(area_size),
				Random::RANDOM.NextInt(area_size));
		} while (pos==startState->goal);// check for random free map position
		startState->GridOpen(pos) = false ;
		startState->rob=pos;//put robot there
	}
	else
	{
		AreaRobPos(startState,area_size+2);//enlarge the area and re-assign
	}
}
*/

/*

DEVICE Belief* Dvc_UncNavigation::InitialBelief(const Dvc_State* start, string type) const {
	int N = Globals::config.num_scenarios*10;


	vector<Dvc_State*> particles(N);
	for (int i = 0; i < N; i++) {
		particles[i] = CreateStartState();
		//particles[i] = CreateFixedStartState();
		particles[i]->weight = 1.0 / N;
	}
	FreeObstacles();
	NewRound=true;
	return new ParticleBelief(particles, this);
}

class Dvc_UncNavigationParticleUpperBound1: public Dvc_ParticleUpperBound {
protected:
	const Dvc_UncNavigation* rs_model_;
public:
	DEVICE Dvc_UncNavigationParticleUpperBound1(const Dvc_UncNavigation* model) :
		rs_model_(model) {
	}

	DEVICE double Value(const Dvc_State& state) const {
		const Dvc_UncNavigationState& nav_state =
			static_cast<const Dvc_UncNavigationState&>(state);
		int count_x=abs(nav_state.rob.x-nav_state.goal.x);
		int count_y=abs(nav_state.rob.y-nav_state.goal.y);
		double value1=0, value2=0;
		for (int i=0;i<count_x;i++)
		{
			value1+=Globals::Discount(i)*(-0.1);
		}
		for (int j=0;j<count_y;j++)
		{
			value1+=Globals::Discount(count_x+j)*(-0.1);
		}

		for (int j=0;j<count_y;j++)
		{
			value2+=Globals::Discount(j)*(-0.1);
		}
		for (int i=0;i<count_x;i++)
		{
			value2+=Globals::Discount(count_y+i)*(-0.1);
		}

		return max(value1,value2)+Globals::Discount(count_x+count_y-1)*(10);
	}
};
DEVICE Dvc_ScenarioUpperBound* Dvc_UncNavigation::CreateScenarioUpperBound(string name,
	string particle_bound_name) const {
	Dvc_ScenarioUpperBound* bound = NULL;
	if (name == "UB1") {
		bound = new Dvc_UncNavigationParticleUpperBound1(this);
	} else if (name == "DEFAULT" || "TRIVIAL") {
		bound = new Dvc_TrivialParticleUpperBound(this);
	} else if (name == "UB2") {
		bound = new UncNavigationParticleUpperBound2(this);
	} else if (name == "DEFAULT" || name == "MDP") {
		bound = new UncNavigationMDPParticleUpperBound(this);
	} else if (name == "APPROX") {
		bound = new UncNavigationApproxParticleUpperBound(this);
	} else {
		cerr << "Unsupported scenario upper bound: " << name << endl;
		exit(0);
	}
	return bound;
}

DEVICE Dvc_ScenarioLowerBound* Dvc_UncNavigation::CreateScenarioLowerBound(string name, string
	particle_bound_name) const {
	if (name == "TRIVIAL") {
		return new Dvc_TrivialParticleLowerBound(this);
	} else if (name == "DEFAULT" || name == "EAST") {
		// scenario_lower_bound_ = new BlindPolicy(this, Dvc_NavCompass::EAST);
		return new UncNavigationEastScenarioLowerBound(this);
	} else if (name == "DEFAULT" || name == "RANDOM") {
		return new RandomPolicy(this,
			CreateParticleLowerBound(particle_bound_name));
	} else if (name == "ENT") {
		return new UncNavigationENTScenarioLowerBound(this);
	} else if (name == "MMAP") {
		return new UncNavigationMMAPStateScenarioLowerBound(this);
	} else if (name == "MODE") {
		// scenario_lower_bound_ = new ModeStatePolicy(this);
		return NULL; // TODO
	} else {
		cerr << "Unsupported lower bound algorithm: " << name << endl;
		exit(0);
		return NULL;
	}
}

DEVICE void Dvc_UncNavigation::PrintState(const Dvc_State& state, ostream& out) const {
	Dvc_UncNavigationState navstate=static_cast<const Dvc_UncNavigationState&>(state);
	int Width=7;
	int Prec=1;
	out << endl;
	for (int x = 0; x < size_ + 2; x++)
	{
		out.width(Width);out.precision(Prec);	out << "# ";
	}
	out << endl;
	for (int y = size_ - 1; y >= 0; y--) {
		out.width(Width);out.precision(Prec);	out << "# ";
		for (int x = 0; x < size_; x++) {
			DvcCoord pos(x, y);
			int obstacle = navstate.Grid(pos);
			if (navstate.goal == DvcCoord(x, y))
			{
				out.width(Width);out.precision(Prec);	out << "G ";
			}
			else if (GetRobPos(&state) == DvcCoord(x, y))
			{
				out.width(Width);out.precision(Prec);	out << "R ";
			}
			else if (obstacle ==true)
			{
				out.width(Width);out.precision(Prec);	out << "X ";
			}
			else
			{
				out.width(Width);out.precision(Prec);	out << ". ";
			}
		}
		out.width(Width);out.precision(Prec);	out << "#" << endl;
	}
	for (int x = 0; x < size_ + 2; x++)
	{
		out.width(Width);out.precision(Prec);	out << "# ";
	}
	out << endl;

	//PrintBelief(*belief_);
}
DEVICE void Dvc_UncNavigation::PrintBeliefMap(float** Beliefmap, std::ostream& out) const
{
	ios::fmtflags old_settings = out.flags();
	//out.precision(5);
	//out.width(4);
	int Width=6;
	int Prec=1;
	out << endl;
	out.width(Width);out.precision(Prec);
	for (int x = 0; x < size_ + 2; x++)
	{
		out.width(Width);out.precision(Prec);	out << "# ";
	}
	out << endl;
	for (int y = size_ - 1; y >= 0; y--) {
		out.width(Width);out.precision(Prec); out << "# ";
		for (int x = 0; x < size_; x++) {
			out.width(Width);out.precision(Prec);
			if(Beliefmap[x][y]>0.2)
			{
				out.width(Width-3);
				out <<"<"<<Beliefmap[x][y]<<">"<<" ";
				out.width(Width);
			}
			else
				out <<Beliefmap[x][y]<<" ";
		}
		out.width(Width);out.precision(Prec); out << "#" << endl;
	}
	for (int x = 0; x < size_ + 2; x++)
		{out.width(Width);out.precision(Prec); out << "# ";}
	out << endl;
	out.flags(old_settings);
}
DEVICE void Dvc_UncNavigation::AllocBeliefMap(float**& Beliefmap) const
{
	Beliefmap =new float*[size_];
	for(int i=0;i<size_;i++)
	{
		Beliefmap[i]=new float[size_];
		memset((void*)Beliefmap[i], 0 , size_*sizeof(float));
	}
}
DEVICE void Dvc_UncNavigation::ClearBeliefMap(float**& Beliefmap) const
{
	for(int i=0;i<size_;i++)
	{
		delete [] Beliefmap[i];
	}
	delete [] Beliefmap;
}
DEVICE void Dvc_UncNavigation::PrintBelief(const Belief& belief, ostream& out) const {
	const vector<Dvc_State*>& particles =
		static_cast<const ParticleBelief&>(belief).particles();
	out << "Robot position belief:";
	float** Beliefmap;float** ObsBeliefmap;
	AllocBeliefMap(Beliefmap);AllocBeliefMap(ObsBeliefmap);
	float GateState[3];

	memset((void*)GateState, 0 , 3*sizeof(float));
	for (int i = 0; i < particles.size(); i++) {

		const Dvc_UncNavigationState* navstate = static_cast<const Dvc_UncNavigationState*>(particles[i]);

		GateState[0]+=((int)navstate->Grid(navstate->GateWest()))*navstate->weight;
		GateState[1]+=((int)navstate->Grid(navstate->GateNorth()))*navstate->weight;
		GateState[2]+=((int)navstate->Grid(navstate->GateEast()))*navstate->weight;

		for (int x=0; x<size_; x++)
			for (int y=0; y<size_; y++)
			{
				if(navstate->rob.x==x && navstate->rob.y==y)
					Beliefmap[x][y]+=navstate->weight;
				DvcCoord pos(x,y);
				ObsBeliefmap[x][y]+=((int)navstate->Grid(pos))*navstate->weight;
					//out << "Weight=" << particles[i]->weight<<endl;
			}
	}

	PrintBeliefMap(Beliefmap,out);
	out << "Map belief:";
	PrintBeliefMap(ObsBeliefmap,out);
	out << "Gate obstacle belief:"<<endl;
	for (int i=0;i<3;i++)
		out<<GateState[i]<<" ";
	out<<endl;
	ClearBeliefMap(Beliefmap);ClearBeliefMap(ObsBeliefmap);
}

DEVICE void Dvc_UncNavigation::PrintAction(int action, ostream& out) const {
	if (action < E_STAY)
		out << Dvc_NavCompass::CompassString[action] << endl;
	if (action == E_STAY)
		out << "Stay" << endl;
}
*/

DEVICE Dvc_State* Dvc_UncNavigation::Allocate(int state_id, double weight) const {
	//Dvc_UncNavigationState* state = Dvc_memory_pool_.Allocate();
	Dvc_UncNavigationState* state = new Dvc_UncNavigationState();
	state->state_id = state_id;
	state->weight = weight;

	return state;
}

DEVICE Dvc_State* Dvc_UncNavigation::Dvc_Get(Dvc_State* particles, int pos) {
	Dvc_UncNavigationState* particle_i= static_cast<Dvc_UncNavigationState*>(particles)+pos;

	return particle_i;
}

DEVICE Dvc_State* Dvc_UncNavigation::Dvc_Alloc( int num) {
	//Dvc_UncNavigationState* state = Dvc_memory_pool_.Allocate();
	Dvc_UncNavigationState* state = (Dvc_UncNavigationState*)malloc(num*sizeof(Dvc_UncNavigationState));

	for(int i=0;i<num;i++)
		state[i].SetAllocated();
	return state;
}

DEVICE Dvc_State* Dvc_UncNavigation::Dvc_Copy(const Dvc_State* particles, int pos) {
	//Dvc_UncNavigationState* state = Dvc_memory_pool_.Allocate();
	const Dvc_UncNavigationState* particle_i= static_cast<const Dvc_UncNavigationState*>(particles)+pos;
	Dvc_UncNavigationState* state = new Dvc_UncNavigationState();

	*state = *particle_i;
	state->SetAllocated();
	return state;
}
DEVICE void Dvc_UncNavigation::Dvc_Copy_NoAlloc(Dvc_State* des, const Dvc_State* src, int pos, bool offset_des) {
	/*Pass member values, assign member pointers to existing state pointer*/
	const Dvc_UncNavigationState* src_i= static_cast<const Dvc_UncNavigationState*>(src)+pos;
	if(!offset_des) pos=0;
	Dvc_UncNavigationState* des_i= static_cast<const Dvc_UncNavigationState*>(des)+pos;

	*des_i = *src_i;
	des_i->SetAllocated();
}
DEVICE void Dvc_UncNavigation::Dvc_Free(Dvc_State* particle) {
	//Dvc_memory_pool_.Free(static_cast<Dvc_UncNavigationState*>(particle));
	delete static_cast<Dvc_UncNavigationState*>(particle);
}
/*DEVICE void Dvc_UncNavigation::Dvc_FreeList(Dvc_State* particle) {
	//Dvc_memory_pool_.Free(static_cast<Dvc_UncNavigationState*>(particle));
	free(static_cast<Dvc_UncNavigationState*>(particle));
}*/
/*
DEVICE int Dvc_UncNavigation::NumActiveParticles() const {
	return Dvc_memory_pool_.num_allocated();
}
*/
DEVICE int Dvc_UncNavigation::Dvc_NumObservations() { // one dummy terminal state
	return 16;
}

/*
DEVICE DvcCoord Dvc_UncNavigation::GetRobPos(const Dvc_State* state) const {
	return static_cast<const Dvc_UncNavigationState*>(state)->rob;
}
*/

DEVICE OBS_TYPE Dvc_UncNavigation::Dvc_GetObservation(double rand_num,
	const Dvc_UncNavigationState& nav_state) {
	OBS_TYPE obs=Dvc_NumObservations()+10;
	double TotalProb=0;
	bool found=false;

	for(int i=0;i<Dvc_NumObservations();i++)//pick an obs according to the prob of each one
	{
		/*TotalProb+=ObsProb(i, nav_state, E_STAY);
		obs=(rand_num<=TotalProb && !found)?i:obs;
		found=(rand_num<=TotalProb)?true:found;*/
		TotalProb+=ObsProb(i, nav_state, E_STAY);
		if(rand_num<=TotalProb)
		{	obs=i;	break;	}
	}
	return obs;
}
/*DEVICE OBS_TYPE Dvc_UncNavigation::Dvc_GetObservation_parallel(double rand_num,
	const Dvc_UncNavigationState& nav_state) {
	float temp=0;

	clock_t start_time = clock();

	OBS_TYPE obs=Dvc_NumObservations()+10;
	double TotalProb=0;
	//bool found=false;

	int action=blockIdx.x;
	int PID=blockIdx.y*blockDim.y+threadIdx.y;
	int obs_i=threadIdx.x;
	clock_t start_time1 = clock();

	__syncthreads();

	Obs_Prob_[action][PID][obs_i]=ObsProb(obs_i, nav_state, E_STAY);

	__syncthreads();
	clock_t stop_time1 = clock();

	float runtime1= (float)(stop_time1 - start_time1)/1733500.0;

	for(int i=0;i<Dvc_NumObservations();i++)//pick an obs according to the prob of each one
	{
		TotalProb+=Obs_Prob_[action][PID][i];
		if(rand_num<=TotalProb)
		{	obs=i;	break;	}
	}
	clock_t stop_time = clock();

	float runtime= (float)(stop_time - start_time)/1733.500;
	//runtime++;
	temp=runtime;
	//__syncthreads();

	return obs;//Debug, test time
}*/
/*
DEVICE int Dvc_UncNavigation::GetX(const Dvc_UncNavigationState* state) const {
	return state->rob.x;
}

DEVICE void Dvc_UncNavigation::IncX(Dvc_UncNavigationState* state) const {
	state->rob.x+=1;
}

DEVICE void Dvc_UncNavigation::DecX(Dvc_UncNavigationState* state) const {
	state->rob.x-=1;
}

DEVICE int Dvc_UncNavigation::GetY(const Dvc_UncNavigationState* state) const {
	return state->rob.y;
}

DEVICE void Dvc_UncNavigation::IncY(Dvc_UncNavigationState* state) const {
	state->rob.y+=1;
}

DEVICE void Dvc_UncNavigation::DecY(Dvc_UncNavigationState* state) const {
	state->rob.y-=1;
}


DEVICE Dvc_UncNavigationState Dvc_UncNavigation::NextState(Dvc_UncNavigationState& s, int a) const {
	if (s.rob==s.goal)// terminal state is an absorbing state
		return s;

    double Rand=Random::RANDOM.NextDouble();
	DvcCoord rob_pos = s.rob;
	Dvc_UncNavigationState newState(s);
	if (a < E_STAY) {//movement actions
	    if(Rand<0.8)// only succeed with 80% chance
	    	rob_pos += Dvc_NavCompass::DIRECTIONS[a];
		if (s.Inside(rob_pos) && s.CollisionCheck(rob_pos)==false) {
			newState.rob=rob_pos;//move the robot
		} else {
			;// don't move the robot
		}
		return newState;
	} else if (a == E_STAY) {//stay action
		return newState;
	} else //unsupported action
		return s;
}

DEVICE double Dvc_UncNavigation::Reward(Dvc_UncNavigationState& s, int a) const {
	if (s.rob==s.goal)// at terminal state, no reward
		return 0;
	DvcCoord rob_pos = s.rob;
	if (a < E_STAY) {
	    double Rand=Random::RANDOM.NextDouble();
	    if(Rand<0.8)// only succeed with 80% chance
	    	rob_pos += Dvc_NavCompass::DIRECTIONS[a];
	    if(rob_pos==s.goal)// arrive goal
	    	return 10;
		if (s.Inside(rob_pos) && s.CollisionCheck(rob_pos)==false) {
			return -0.1;// small cost for each move
		} else
			return -1;//run into obstacles or walls
	} else if (a == E_STAY) {
		return -0.1;// small cost for each step
	} else //unsupported action
		return 0;
}
*/


} // namespace despot
