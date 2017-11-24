#include "ped_pomdp.h"

#include <despot/core/policy_graph.h>
#include <limits>
#include <despot/GPUutil/GPUlimits.h>
#include <despot/GPUcore/thread_globals.h>


class PedPomdpParticleLowerBound : public ParticleLowerBound {
private:
	const PedPomdp* ped_pomdp_;
public:
	PedPomdpParticleLowerBound(const DSPOMDP* model) :
		ParticleLowerBound(model),
		ped_pomdp_(static_cast<const PedPomdp*>(model))
	{
	}

    // IMPORTANT: Check after changing reward function.
	virtual ValuedAction Value(const std::vector<State*>& particles) const {
		PomdpState* state = static_cast<PomdpState*>(particles[0]);
		int min_step = numeric_limits<int>::max();
		auto& carpos = ped_pomdp_->world.path[state->car.pos];
		double carvel = state->car.vel;
	
		// Find mininum num of steps for car-pedestrian collision
		for (int i=0; i<state->num; i++) {
			auto& p = state->peds[i];
            // 3.25 is maximum distance to collision boundary from front laser (see collsion.cpp)
			int step = (p.vel + carvel<=1e-5)?min_step:int(ceil(ModelParams::control_freq
						* max(COORD::EuclideanDistance(carpos, p.pos) - 1.0, 0.0)
						/ ((p.vel + carvel))));
			/*if(step<0) printf("   step, vel_sum, ped_vel, car_vel = %d %f %f %f\n"
					,step, p.vel+carvel,p.vel, carvel);
			if(p.vel<=1e-5) printf("   ped: goal,id,pos_x, pos_y, vel = %d %d %f %f %f\n"
					,p.goal, p.id,p.pos.x,p.pos.y, p.vel);*/

			min_step = min(step, min_step);
		}

        double move_penalty = ped_pomdp_->MovementPenalty(*state);

        // Case 1, no pedestrian: Constant car speed
		double value = move_penalty / (1 - Globals::Discount());
        // Case 2, with pedestrians: Constant car speed, head-on collision with nearest neighbor
		if (min_step != numeric_limits<int>::max()) {
            double crash_penalty = ped_pomdp_->CrashPenalty(*state);
			value = (move_penalty) * (1 - Globals::Discount(min_step)) / (1 - Globals::Discount())
				+ crash_penalty * Globals::Discount(min_step);
			/*if(true)
				printf("   min_step,crash_penalty, value=%d %f %f\n"
						,min_step, crash_penalty,value);*/
		}
        /*if(true)
			printf("   min_step,num_peds,move_penalty, value=%d %d %f %f\n"
					,min_step,state->num, move_penalty,value);*/

		/*if(particles[0]->scenario_id==52){
			printf("Rollout bottom: bottom_value=%f, last discount depth=-- \n", value);
		}*/
		return ValuedAction(ped_pomdp_->ACT_CUR, State::Weight(particles) * value);
	}
};

PedPomdp::PedPomdp(WorldModel &model_) :
	world(model_),
	random_(Random((unsigned) Seeds::Next()))
{
	//particle_lower_bound_ = new PedPomdpParticleLowerBound(this);
}

const std::vector<int>& PedPomdp::ObserveVector(const State& state_) const {
	const PomdpState &state=static_cast<const PomdpState&>(state_);
	static std::vector<int> obs_vec;
	obs_vec.resize(state.num * 2 + 2);

	int i=0;
    obs_vec[i++] = state.car.pos;
	obs_vec[i++] = int((state.car.vel+1e-5) / ModelParams::vel_rln);//add some noise to make 1.5/0.003=50

	for(int j = 0; j < state.num; j ++) {
		obs_vec[i++] = int(state.peds[j].pos.x / ModelParams::pos_rln); 
		obs_vec[i++] = int(state.peds[j].pos.y / ModelParams::pos_rln);
	}

	if(CPUDoPrint){
		for (int j=0;j<i;j++)
			cout<<obs_vec[j]<<" ";
		cout<<endl;
	}
	return obs_vec;
}

uint64_t PedPomdp::Observe(const State& state) const {
	hash<std::vector<int>> myhash;
	return myhash(ObserveVector(state));
}

std::vector<State*> PedPomdp::ConstructParticles(std::vector<PomdpState> & samples) {
	int num_particles=samples.size();
	std::vector<State*> particles;
	for(int i=0;i<samples.size();i++) {
		PomdpState* particle = static_cast<PomdpState*>(Allocate(-1, 1.0/num_particles));
		(*particle) = samples[i];
		particle->SetAllocated();
		particle->weight = 1.0/num_particles;
		particles.push_back(particle);
	}

	/*if(static_cast<PomdpState*>(particles[0])->num>=ModelParams::N_PED_IN-1)
	    cout << "11 reached!"<<endl;*/
	return particles;
}

// Very high cost for collision
double PedPomdp::CrashPenalty(const PomdpState& state) const { // , int closest_ped, double closest_dist) const {
	// double ped_vel = state.ped[closest_ped].vel;
    return ModelParams::CRASH_PENALTY * (state.car.vel * state.car.vel + ModelParams::REWARD_BASE_CRASH_VEL); 
}
// Very high cost for collision
double PedPomdp::CrashPenalty(const PomdpStateWorld& state) const { // , int closest_ped, double closest_dist) const {
	// double ped_vel = state.ped[closest_ped].vel;
    return ModelParams::CRASH_PENALTY * (state.car.vel * state.car.vel + ModelParams::REWARD_BASE_CRASH_VEL); 
}

// Avoid frequent dec or acc
double PedPomdp::ActionPenalty(int action) const {
    return (action == ACT_DEC || action == ACT_ACC) ? -0.1 : 0.0;
}

// Less penalty for longer distance travelled
double PedPomdp::MovementPenalty(const PomdpState& state) const {
    return ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
    /*
    // with no pedestrians nearby, but lower speed with pedestrians nearby
    retrun //(min_dist > 3.0 || (min_dist < 3.0 && action != ACT_ACC)) ?
		(1.0 * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX);
		//: -1.5;
    */
}
// Less penalty for longer distance travelled
double PedPomdp::MovementPenalty(const PomdpStateWorld& state) const {
    return ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
    /*
    // with no pedestrians nearby, but lower speed with pedestrians nearby
    retrun //(min_dist > 3.0 || (min_dist < 3.0 && action != ACT_ACC)) ?
		(1.0 * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX);
		//: -1.5;
    */
}

bool PedPomdp::Step(State& state_, double rNum, int action, double& reward, uint64_t& obs) const {
	PomdpState& state = static_cast<PomdpState&>(state_);
	reward = 0.0;


	/*if(state_.scenario_id==CPUPrintPID){
		if(CPUDoPrint){
			printf("(CPU) Before step: scenario%d \n", state_.scenario_id);
			PomdpState* pedpomdp_state=static_cast<PomdpState*>(&state_);
			printf("Before step:\n");
			printf("car pox= %d ",pedpomdp_state->car.pos);
			printf("dist=%f\n",pedpomdp_state->car.dist_travelled);
			printf("car vel= %f\n",pedpomdp_state->car.vel);
			for(int i=0;i<pedpomdp_state->num;i++)
			{
				printf("ped %d pox_x= %f pos_y=%f\n",i,
						pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y);
			}
		}
	}*/
	// CHECK: relative weights of each reward component
	// Terminate upon reaching goal
	if (world.isLocalGoal(state)) {
        reward = ModelParams::GOAL_REWARD;
    	/*if(CPUDoPrint && state_.scenario_id==CPUPrintPID){
			printf("Reach goal\n");
    	}*/
		return true;
	}

	//int closest_front_ped;
	//double closest_front_dist;
	//int closest_side_ped;
	//double closest_side_dist;
	//world.getClosestPed(state, closest_front_ped, closest_front_dist,
			//closest_side_ped, closest_side_dist);

 	// Safety control: collision; Terminate upon collision
	//if (closest_front_dist < ModelParams::COLLISION_DISTANCE) {
    if(state.car.vel > 0.001 && world.inCollision(state) ) { /// collision occurs only when car is moving
		reward = CrashPenalty(state); //, closest_ped, closest_dist);
		/*if(CPUDoPrint && state_.scenario_id==CPUPrintPID){
			printf("Crash\n");
		}*/
		return true;
	}

	// Forbidden actions
/*    double carvel = state.car.vel;
	if (action == ACT_CUR && 0.1 < carvel && carvel < 0.6) {
		reward = CrashPenalty(state);
		return true;
	}
    if (action == ACT_ACC && carvel >= ModelParams::VEL_MAX) {
		reward = CrashPenalty(state);
		return true;
    }
    if (action == ACT_DEC && carvel <= 0.01) {
		reward = CrashPenalty(state);
		return true;
    }
*/
    // encourage speed up when there's no pedestrian
    //if (carvel < 0.1 && closest_side_dist > 1.5 && closest_front_dist > 3.5 &&
            //(closest_front_ped < 0 || world.isMovingAway(state, closest_front_ped))) {
        //reward += -10;
    //}


    // encourage slowdown when pedestrian is close
    //double min_dist_all_dirs = min(closest_front_dist, closest_side_dist);
	//if (min_dist_all_dirs<6.0 && state.car.vel>1.0) {
		//reward+=-4;
	//}

	/*
	if ((min_dist < 2.5 && (state.car.vel > 1.0 || action == ACT_ACC)) || (min_dist > 4.0 && state.car.vel < 0.5)) {
		reward += -1000;
	}
	*/

	// Smoothness control
	reward += ActionPenalty(action);
	/*if(state_.scenario_id==52){
		printf("+ActionPenalty (act %d)=%f\n",action, reward);
	}*/
	// Speed control: Encourage higher speed
	reward += MovementPenalty(state);
	/*if(state_.scenario_id==52){
		printf("+MovementPenalty=%f\n",reward);
	}*/
	// State transition
	//Random random(rNum);
	double acc = (action == ACT_ACC) ? ModelParams::AccSpeed :
		((action == ACT_CUR) ?  0 : (-ModelParams::AccSpeed));

	//use rNum directly to keep consistent with GPU codes
	if(use_multi_thread_)
	{
		record[MapThread(this_thread::get_id())]=123456;
	}
	else
		record[0]=123456;

	//if(action==0 && state_.scenario_id==0)cout<<rNum<<endl;
	//world.FixGPUVel(state.car);
	world.RobStep(state.car, rNum/*random*/);
	//if(action==0 && state_.scenario_id==0)cout<<"1-"<<rNum<<endl;
	/*if(state.scenario_id==52)
		printf("start rand=%f\n",rNum);*/
	world.RobVelStep(state.car, acc, rNum/*random*/);

	//if(action==0 && state_.scenario_id==0)cout<<"2-"<<rNum<<endl;
	for(int i=0;i<state.num;i++)
	{
		//assert(state.peds[i].pos.x==state.peds[i].pos.x);//debugging
		/*if(state.scenario_id==52 && i==8)
			printf("ped %d rand=%f\n",i, rNum);*/
		world.PedStep(state.peds[i], rNum/*random*/);

		assert(state.peds[i].pos.x==state.peds[i].pos.x);//debugging
	}


	/*if(CPUDoPrint && state.scenario_id==CPUPrintPID){
		if(true){
			PomdpState* pedpomdp_state=static_cast<PomdpState*>(&state_);
			printf("rand=%f, action=%d \n", rNum, action);

			printf("After step:\n");
			printf("Reward=%f\n",reward);

			printf("car pox= %d ",pedpomdp_state->car.pos);
			printf("dist=%f\n",pedpomdp_state->car.dist_travelled);
			printf("car vel= %f\n",pedpomdp_state->car.vel);
			for(int i=0;i<pedpomdp_state->num;i++)
			{
				printf("ped %d pox_x= %f pos_y=%f\n",i,
						pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y);
			}
		}
	}*/
	//if(action==0 && state_.scenario_id==0)cout<<"3-"<<rNum<<endl;
	// Observation
	obs = Observe(state);
	/*if(CPUDoPrint && state.scenario_id==CPUPrintPID){
		cout<< "observation="<<obs<<endl;
	}*/
	return false;
}

bool PedPomdp::Step(PomdpStateWorld& state, double rNum, int action, double& reward, uint64_t& obs) const {
	
	reward = 0.0;

	if (world.isLocalGoal(state)) {
        reward = ModelParams::GOAL_REWARD;
		return true;
	}

    if(state.car.vel > 0.001 && world.inCollision(state) ) { /// collision occurs only when car is moving
		reward = CrashPenalty(state); //, closest_ped, closest_dist);
		return true;
	}
/*
	// Forbidden actions
    double carvel = state.car.vel;
	if (action == ACT_CUR && 0.1 < carvel && carvel < 0.6) {
		reward = CrashPenalty(state);
		return true;
	}
    if (action == ACT_ACC && carvel >= ModelParams::VEL_MAX) {
		reward = CrashPenalty(state);
		return true;
    }
    if (action == ACT_DEC && carvel <= 0.01) {
		reward = CrashPenalty(state);
		return true;
    } 
*/
	// Smoothness control
	reward += ActionPenalty(action);

	// Speed control: Encourage higher speed
	reward += MovementPenalty(state);

	// State transition
	Random random(rNum);
	double acc = (action == ACT_ACC) ? ModelParams::AccSpeed :
		((action == ACT_CUR) ?  0 : (-ModelParams::AccSpeed));
	world.RobStep(state.car, random);
	world.RobVelStep(state.car, acc, random);
	for(int i=0;i<state.num;i++)
		world.PedStep(state.peds[i], random);

	return false;
}

bool PedPomdp::ImportanceSamplingStep(State& state_, double rNum, int action, double& reward, uint64_t& obs) const {
	PomdpState& state = static_cast<PomdpState&>(state_);
	reward = 0.0;
	// CHECK: relative weights of each reward component
	// Terminate upon reaching goal
	if (world.isLocalGoal(state)) {
        reward = ModelParams::GOAL_REWARD;
		return true;
	}

	//int closest_front_ped;
	//double closest_front_dist;
	//int closest_side_ped;
	//double closest_side_dist;
	//world.getClosestPed(state, closest_front_ped, closest_front_dist,
			//closest_side_ped, closest_side_dist);

 	// Safety control: collision; Terminate upon collision
	//if (closest_front_dist < ModelParams::COLLISION_DISTANCE) {
    if(state.car.vel > 0.001 && world.inCollision(state) ) { /// collision occurs only when car is moving
		reward = CrashPenalty(state); //, closest_ped, closest_dist);
		return true;
	}
/*
	// Forbidden actions

    double carvel = state.car.vel;
	if (action == ACT_CUR && 0.1 < carvel && carvel < 0.6) {
		reward = CrashPenalty(state);
		return true;
	}
    if (action == ACT_ACC && carvel >= ModelParams::VEL_MAX) {
		reward = CrashPenalty(state);
		return true;
    }
    if (action == ACT_DEC && carvel <= 0.01) {
		reward = CrashPenalty(state);
		return true;
    }
*/
    // encourage speed up when there's no pedestrian
    //if (carvel < 0.1 && closest_side_dist > 1.5 && closest_front_dist > 3.5 &&
            //(closest_front_ped < 0 || world.isMovingAway(state, closest_front_ped))) {
        //reward += -10;
    //}


    // encourage slowdown when pedestrian is close
    //double min_dist_all_dirs = min(closest_front_dist, closest_side_dist);
	//if (min_dist_all_dirs<6.0 && state.car.vel>1.0) {
		//reward+=-4;
	//}

	/*
	if ((min_dist < 2.5 && (state.car.vel > 1.0 || action == ACT_ACC)) || (min_dist > 4.0 && state.car.vel < 0.5)) {
		reward += -1000;
	}
	*/

	// Smoothness control
	reward += ActionPenalty(action);

	// Speed control: Encourage higher speed
	reward += MovementPenalty(state);

	// State transition
	Random random(rNum);
	double acc = (action == ACT_ACC) ? ModelParams::AccSpeed :
		((action == ACT_CUR) ?  0 : (-ModelParams::AccSpeed));
	world.RobStep(state.car, random);

	//state.weight *= world.ISRobVelStep(state.car, acc, random);
	world.RobVelStep(state.car, acc, random);
	
	for(int i=0;i<state.num;i++)
		world.PedStep(state.peds[i], random);
		//state.weight *= world.ISPedStep(state.car, state.peds[i], random);

	// Observation
	obs = Observe(state);

	return false;
}

double PedPomdp::ObsProb(uint64_t obs, const State& s, int action) const {
	return obs == Observe(s);
}

std::vector<std::vector<double>> PedPomdp::GetBeliefVector(const std::vector<State*> particles) const {
	std::vector<std::vector<double>> belief_vec;
	return belief_vec;
}

Belief* PedPomdp::InitialBelief(const State* start, string type) const {
	assert(false);
	return NULL;
}

/// output the probability of the intentions of the pedestrians
void PedPomdp::Statistics(const std::vector<PomdpState*> particles) const {
	double goal_count[10][10]={{0}};
	cout << "Current Belief" << endl;
	if(particles.size() == 0)
		return;

	PrintState(*particles[0]);
	PomdpState* state_0 = particles[0];
	for(int i = 0; i < particles.size(); i ++) {
		PomdpState* state = particles[i];
		for(int j = 0; j < state->num; j ++) {
			goal_count[j][state->peds[j].goal] += particles[i]->weight;
		}
	}

	for(int j = 0; j < state_0->num; j ++) {
		cout << "Ped " << j << " Belief is ";
		for(int i = 0; i < world.goals.size(); i ++) {
			cout << (goal_count[j][i] + 0.0) <<" ";
		}
		cout << endl;
	}
}

/*/// output the probability of the intentions of the pedestrians
void PedPomdp::PrintParticles(const std::vector<State*> particles, ostream& out) const {
	cout<<"******** scenario belief********"<<endl;

	double goal_count[10][10]={{0}};
	cout << "Current Belief" << endl;
	if(particles.size() == 0)
		return;
	const PomdpState* pomdp_state=static_cast<const PomdpState*>(particles.at(0));

	//PrintState(*pomdp_state);

	//PomdpState* state_0 = particles[0];
	//const PomdpState* state_0=static_cast<const PomdpState*>(particles.at(0));
	for(int i = 0; i < particles.size(); i ++) {
		//PomdpState* state = particles[i];
		const PomdpState* pomdp_state=static_cast<const PomdpState*>(particles.at(i));
		for(int j = 0; j < pomdp_state->num; j ++) {
			goal_count[j][pomdp_state->peds[j].goal] += particles[i]->weight;
		}
	}

	for(int j = 0; j < 6; j ++) {
		cout << "Ped " << pomdp_state->peds[j].id << " Belief is ";
		for(int i = 0; i < world.goals.size(); i ++) {
			cout << (goal_count[j][i] + 0.0) <<" ";
		}
		cout << endl;
	}

	cout<<"******** end of scenario belief********"<<endl;
}*/

ValuedAction PedPomdp::GetMinRewardAction() const {
	return ValuedAction(0, 
			ModelParams::CRASH_PENALTY * (ModelParams::VEL_MAX*ModelParams::VEL_MAX + ModelParams::REWARD_BASE_CRASH_VEL));
}

class PedPomdpSmartScenarioLowerBound : public Policy {
protected:
	const PedPomdp* ped_pomdp_;

public:
	PedPomdpSmartScenarioLowerBound(const DSPOMDP* model, ParticleLowerBound* bound) :
		Policy(model,bound),
		//ped_pomdp_(model)
		ped_pomdp_(static_cast<const PedPomdp*>(model))
	{
	}

	int Action(const std::vector<State*>& particles,
			RandomStreams& streams, History& history) const {
		return ped_pomdp_->world.defaultPolicy(particles);
	}
};

class PedPomdpSmartGraphScenarioLowerBound : public PolicyGraph {
protected:
	WorldModel* world_;
	enum {
		CLOSE_STATIC,
		CLOSE_MOVING,
		MEDIUM_FAST,
		MEDIUM_MED,
		MEDIUM_SLOW,
		FAR_MAX,
		FAR_NOMAX,
	};
public:

	PedPomdpSmartGraphScenarioLowerBound(const DSPOMDP* model, ParticleLowerBound* bound,WorldModel* world,
			Belief* belief=NULL):
			PolicyGraph(model, bound, belief), world_(world) {

	}
	~PedPomdpSmartGraphScenarioLowerBound(){
		ClearGraph();
	}
	void ConstructGraph(int size, int branch){
		size = 3;//3 action nodes
		branch= 7; //7 obs edges
		cout<<"FIX_SCENARIO,Load_Graph="<<FIX_SCENARIO<<" "<<Load_Graph<<endl;
		if(FIX_SCENARIO==1 || Load_Graph==true)
		{
			ifstream fin;fin.open("Graph.txt", ios::in);

			assert(fin.is_open());
			/*if(!fin.is_open())
			{
				throw string("No Graph.txt!");
				exit(-1);
			}*/
			ImportGraph(fin,size,branch);
			fin.close();
		}
		else
		{
			cout<<"Generating graph"<<endl;
			graph_size_=size;
			num_edges_per_node_=branch;
			action_nodes_.resize(size);

			//Set random actions for nodes
			for (int i = 0; i < graph_size_; i++)
			{
				action_nodes_[i]=i;
			}
			//Link to random nodes for edges
			for (int i = 0; i < num_edges_per_node_; i++)
			{
				switch(i){
				case CLOSE_STATIC:
					for (int j = 0; j < graph_size_; j++){
						obs_edges_[(OBS_TYPE)i].push_back(0);
					}
					break;
				case CLOSE_MOVING:
					for (int j = 0; j < graph_size_; j++){
						obs_edges_[(OBS_TYPE)i].push_back(2);
					}
					break;
				case MEDIUM_FAST:
					for (int j = 0; j < graph_size_; j++){
						obs_edges_[(OBS_TYPE)i].push_back(2);
					}
					break;
				case MEDIUM_MED:
					for (int j = 0; j < graph_size_; j++){
						obs_edges_[(OBS_TYPE)i].push_back(0);
					}
					break;
				case MEDIUM_SLOW:
					for (int j = 0; j < graph_size_; j++){
						obs_edges_[(OBS_TYPE)i].push_back(1);
					}
					break;
				case FAR_MAX:
					for (int j = 0; j < graph_size_; j++){
						obs_edges_[(OBS_TYPE)i].push_back(0);
					}
					break;
				case FAR_NOMAX:
					for (int j = 0; j < graph_size_; j++){
						obs_edges_[(OBS_TYPE)i].push_back(1);
					}
					break;
				}
			}
			current_node_=0;
		}

		if(FIX_SCENARIO==2 && !Load_Graph)
		{
			ofstream fout;fout.open("Graph.txt", ios::trunc);
			ExportGraph(fout);
			fout.close();
		}
	}

	ValuedAction Value(const vector<State*>& particles,
		RandomStreams& streams, History& history) const {
		vector<State*> copy;
		for (int i = 0; i < particles.size(); i++)
			copy.push_back(model_->Copy(particles[i]));

		initial_depth_ = history.Size();
		int MaxDepth=min(Globals::config.max_policy_sim_len+initial_depth_,streams.Length());
		int depth;
		if(FIX_SCENARIO)
			entry_node_=0;/*Debug*/
		else
			entry_node_=0;
		int Action_decision=action_nodes_[entry_node_];
		double TotalValue=0;
		int init_pos=streams.position();

		//cout<<" [Vnode] raw lb for "<<copy.size()<<" particles: ";
		for (int i = 0; i < copy.size(); i++)
		{
			current_node_=entry_node_;

			streams.position(init_pos);
			State* particle = copy[i];
			vector<State*> local_particles;
			local_particles.push_back(particle);
			bool terminal=false;
			double value = 0;

			for(depth=initial_depth_-1;depth<MaxDepth;depth++)
			{
				int action = action_nodes_[current_node_];

				if(depth==initial_depth_)
					Action_decision=action;
				terminal=false;
				if(depth>=initial_depth_){
					OBS_TYPE obs;
					double reward;

					/*if(i==0){
						if(!terminal){
							PomdpState* pedpomdp_state=static_cast<PomdpState*>(particle);
							printf("Before step:\n");
							for(int i=0;i<pedpomdp_state->num;i++)
							{
								printf("ped %d pox_x= %f pos_y=%f\n",i,
										pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y);
							}
						}
					}*/
					terminal = model_->Step(*particle,
						streams.Entry(particle->scenario_id), action, reward, obs);

					value += reward * particle->weight * Globals::Discount(depth-initial_depth_);
					/*if(particle->scenario_id==418 )
						printf("terminal,reward, value, depth, init_depth=%d %f %f %d %d \n"
							,terminal,reward,value/particle->weight,depth,initial_depth_);*/
					/*if(history.Size()>1)
						printf("terminal,reward, value, depth, init_depth=%d %f %f %d %d \n"
							,terminal,reward,value/particle->weight,depth,initial_depth_);*/
					streams.Advance();
					/*if(i==0){
						if(!terminal){
							PomdpState* pedpomdp_state=static_cast<PomdpState*>(particle);
							printf("After step:\n");
							for(int i=0;i<pedpomdp_state->num;i++)
							{
								printf("ped %d pox_x= %f pos_y=%f\n",i,
										pedpomdp_state->peds[i].pos.x,pedpomdp_state->peds[i].pos.y);
							}
						}
						printf("Act at depth %d = %d, get reward %f\n",depth,action,reward);
					}*/

					/*if(particle->scenario_id==418)
						printf("Act at depth %d = %d, get reward %f\n",depth,action,reward);*/
				}
				if(terminal)
					break;
				else
				{
					int edge=world_->defaultGraphEdge(particle, (particle->scenario_id==418)?true:false);
					//if(particle->scenario_id==418) printf("Trace edge %d\n",edge);
					//vector<int> link= obs_edges_[obs];
					current_node_=obs_edges_[(OBS_TYPE)edge][current_node_];
					if(current_node_>10000)
						cout<<"error"<<endl;
				}
			}

			if(!terminal)
			{
				value += Globals::Discount(depth-initial_depth_+1) * particle_lower_bound_->Value(local_particles).value;
				/*if( particle->scenario_id==418 )
					printf("terminal,va.value, depth, init_depth=%d %f %d %d \n"
							,terminal,value/particle->weight,depth,initial_depth_);*/
				/*if(history.Size()>1)
					printf("terminal,va.value, depth, init_depth=%d %f %d %d \n"
						,terminal,value/particle->weight,depth,initial_depth_);*/
			}
			//cout.precision(3);
			//cout<< value<<" ";
			TotalValue+=value;
		}
		//cout<<endl;
		for (int i = 0; i < copy.size(); i++)
			model_->Free(copy[i]);

		return ValuedAction(Action_decision, TotalValue);
	}
};

/*void PedPomdp::InitializeScenarioLowerBound(string name, RandomStreams& streams) {
	// name = "TRIVIAL";
	name="SMART";
	if(name == "TRIVIAL") {
		scenario_lower_bound_ = new TrivialScenarioLowerBound(this);
	} else if(name == "RANDOM") {
		scenario_lower_bound_ = new RandomPolicy(this);
	} else if (name == "SMART") {
		scenario_lower_bound_ = new PedPomdpSmartScenarioLowerBound(this);
	} else {
		cerr << "Unsupported scenario lower bound: " << name << endl;
		exit(0);
	}
}*/

ScenarioLowerBound* PedPomdp::CreateScenarioLowerBound(string name,
		string particle_bound_name) const {
	// name = "TRIVIAL";
	//name="SMART_GRAPH";
	name="SMART";

	if(name == "TRIVIAL") {
		return new TrivialParticleLowerBound(this);
	} else if(name == "RANDOM") {
		//return new RandomPolicy(this);
		//return new RandomPolicy(this, CreateParticleLowerBound(particle_bound_name)); 
		return new RandomPolicy(this, new PedPomdpParticleLowerBound(this));
		//return NULL;
	} else if (name == "SMART") {
		Globals::config.rollout_type="INDEPENDENT";
		cout<<"Smart policy independent rollout"<<endl;
		//return new PedPomdpSmartScenarioLowerBound(this, CreateParticleLowerBound(particle_bound_name));
		return new PedPomdpSmartScenarioLowerBound(this, new PedPomdpParticleLowerBound(this));
	} else if (name == "SMART_GRAPH") {
		Globals::config.rollout_type="GRAPH";
		cout<<"Smart policy graph rollout"<<endl;

		PedPomdpSmartGraphScenarioLowerBound* tmp=new PedPomdpSmartGraphScenarioLowerBound(this,
				new PedPomdpParticleLowerBound(this), &world);
		tmp->ConstructGraph(3, 7);
		tmp->SetEntry(0);
		return tmp;
	} else {
		cerr << "Unsupported scenario lower bound: " << name << endl;
		exit(0);
	}
}

double PedPomdp::GetMaxReward() const {
	return 0;
}

class PedPomdpSmartParticleUpperBound : public ParticleUpperBound {
protected:
	const PedPomdp* ped_pomdp_;
public:
	PedPomdpSmartParticleUpperBound(const DSPOMDP* model) :
		//ParticleUpperBound(model),
		ped_pomdp_(static_cast<const PedPomdp*>(model))
	{
	}

    // IMPORTANT: Check after changing reward function.
	double Value(const State& s) const {
		const PomdpState& state = static_cast<const PomdpState&>(s);
        /*
        double min_dist = ped_pomdp_->world.getMinCarPedDist(state);
        if (min_dist < ModelParams::COLLISION_DISTANCE) {
            return ped_pomdp_->CrashPenalty(state);
        }
        */
        if (ped_pomdp_->world.inCollision(state))
            return ped_pomdp_->CrashPenalty(state);

        int min_step = ped_pomdp_->world.minStepToGoal(state);
		return ModelParams::GOAL_REWARD * Globals::Discount(min_step);
	}
};

/*void PedPomdp::InitializeParticleUpperBound(string name, RandomStreams& streams) {
	name = "SMART";
	if (name == "TRIVIAL") {
		particle_upper_bound_ = new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		particle_upper_bound_ = new PedPomdpSmartParticleUpperBound(this);
	} else {
		cerr << "Unsupported particle upper bound: " << name << endl;
		exit(0);
	}
}

void PedPomdp::InitializeScenarioUpperBound(string name, RandomStreams& streams) {
	//name = "SMART";
	name = "TRIVIAL";
	if (name == "TRIVIAL") {
		scenario_upper_bound_ = new TrivialScenarioUpperBound(this);
	} else {
		cerr << "Unsupported scenario upper bound: " << name << endl;
		exit(0);
	}
}*/

ParticleUpperBound* PedPomdp::CreateParticleUpperBound(string name) const{
	name = "SMART";
	if (name == "TRIVIAL") {
		return new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		return new PedPomdpSmartParticleUpperBound(this);
	} else {
		cerr << "Unsupported particle upper bound: " << name << endl;
		exit(0);
	}
}

ScenarioUpperBound* PedPomdp::CreateScenarioUpperBound(string name,
		string particle_bound_name) const{
	//name = "SMART";
	name = "TRIVIAL";
	if (name == "TRIVIAL") {
		cout<<"Trivial upper bound"<<endl;
		return new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		cout<<"Smart upper bound"<<endl;
		return new PedPomdpSmartParticleUpperBound(this);
	}
	else {
		cerr << "Unsupported scenario upper bound: " << name << endl;
		exit(0);
	}
}

void PedPomdp::PrintState(const State& s, ostream& out) const {
	const PomdpState & state=static_cast<const PomdpState&> (s);
    COORD& carpos = world.path[state.car.pos];

	out << "car pos / dist_trav / vel = " << "(" << carpos.x<< ", " <<carpos.y << ") / " 
        << state.car.dist_travelled << " / "
        << state.car.vel << endl;
	out<< state.num << " pedestrians " << endl;
	for(int i = 0; i < state.num; i ++) {
		out << "ped " << i << ": id / pos / vel / goal / dist2car / infront =  " << state.peds[i].id << " / "
            << "(" << state.peds[i].pos.x << ", " << state.peds[i].pos.y << ") / "
            << state.peds[i].vel << " / "
            << state.peds[i].goal << " / "
            << COORD::EuclideanDistance(state.peds[i].pos, carpos) << "/"
			<< world.inFront(state.peds[i].pos, state.car.pos) << endl;
	}
    double min_dist = -1;
    if (state.num > 0)
        min_dist = COORD::EuclideanDistance(carpos, state.peds[0].pos);
	out << "MinDist: " << min_dist << endl;
}
void PedPomdp::PrintWorldState(PomdpStateWorld state, ostream& out) {
    COORD& carpos = world.path[state.car.pos];

	out << "car pos / dist_trav / vel = " << "(" << carpos.x<< ", " <<carpos.y << ") / "
        << state.car.dist_travelled << " / "
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
            << state.peds[i].vel << " / "
            << state.peds[i].goal << " / "
            << COORD::EuclideanDistance(state.peds[i].pos, carpos) << "/"
			<< world.inFront(state.peds[i].pos, state.car.pos) << endl;
	}
    if (state.num > 0)
        min_dist = COORD::EuclideanDistance(carpos, state.peds[/*0*/mindist_id].pos);
	out << "MinDist: " << min_dist << endl;
}
void PedPomdp::PrintObs(const State&state, uint64_t obs, ostream& out) const {
	out << obs << endl;
}

void PedPomdp::PrintAction(int action, ostream& out) const {
	out << action << endl;
}

void PedPomdp::PrintBelief(const Belief& belief, ostream& out ) const {
	
}

/// output the probability of the intentions of the pedestrians
void PedPomdp::PrintParticles(const vector<State*> particles, ostream& out) const {
	cout<<"Particles for planning:"<<endl;
	double goal_count[ModelParams::N_PED_IN][10]={{0}};
    double q_goal_count[ModelParams::N_PED_IN][10]={{0}};//without weight, it is q;
    double q_single_weight;
    q_single_weight=1.0/particles.size();
	cout << "Current Belief" << endl;
	if(particles.size() == 0)
		return;
	const PomdpState* pomdp_state=static_cast<const PomdpState*>(particles.at(0));

	//PrintState(*pomdp_state);

	//PomdpState* state_0 = particles[0];
	//const PomdpState* state_0=static_cast<const PomdpState*>(particles.at(0));
	for(int i = 0; i < particles.size(); i ++) {
		//PomdpState* state = particles[i];
		const PomdpState* pomdp_state=static_cast<const PomdpState*>(particles.at(i));
		for(int j = 0; j < pomdp_state->num; j ++) {
			goal_count[j][pomdp_state->peds[j].goal] += particles[i]->weight;
            q_goal_count[j][pomdp_state->peds[j].goal] += q_single_weight;
		}
	}

	for(int j = 0; j < 6; j ++) {
		cout << "Ped " << pomdp_state->peds[j].id << " Belief is ";
		for(int i = 0; i < world.goals.size(); i ++) {
			cout << (goal_count[j][i] + 0.0) <<" ";
		}
		cout << endl;
	}

    cout<<"<><><> q:"<<endl;
    for(int j = 0; j < 6; j ++) {
        cout << "Ped " << pomdp_state->peds[j].id << " Belief is ";
        for(int i = 0; i < world.goals.size(); i ++) {
            cout << (q_goal_count[j][i] + 0.0) <<" ";
        }
        cout << endl;
    }

	cout<<"******** end of scenario belief********"<<endl;

}

State* PedPomdp::Allocate(int state_id, double weight) const {
	//num_active_particles ++;
	PomdpState* particle = memory_pool_.Allocate();
	particle->state_id = state_id;
	particle->weight = weight;
	return particle;
}

State* PedPomdp::Copy(const State* particle) const {
	//num_active_particles ++;
	PomdpState* new_particle = memory_pool_.Allocate();
	*new_particle = *static_cast<const PomdpState*>(particle);

	//debugging
	/*for(int i=0;i<new_particle->num;i++){
		PedStruct ped = new_particle -> peds[i];
		assert(ped.pos.x == ped.pos.x);
	}*/
	new_particle->SetAllocated();
	return new_particle;
}

void PedPomdp::Free(State* particle) const {
	//num_active_particles --;
	memory_pool_.Free(static_cast<PomdpState*>(particle));
}


int PedPomdp::NumActiveParticles() const {
	return memory_pool_.num_allocated();
}

double PedPomdp::ImportanceScore(PomdpState* state) const
{	
	double score = 1.8; //0.3 * 6; 0.3 basic score for each pedestrian
	for(int i=0;i<state->num;i++){
		PedStruct ped = state -> peds[i];
		CarStruct car = state -> car;
		COORD ped_pos = ped.pos;

		const COORD& goal = world.goals[ped.goal];
		double move_dw, move_dh;
		if (goal.x == -1 && goal.y == -1) {  //stop intention 
			move_dw = 0; //stop intention does not change ped pos, hence movement is 0;
			move_dh = 0;
		}
		else {
			MyVector goal_vec(goal.x - ped_pos.x, goal.y - ped_pos.y);
		    double a = goal_vec.GetAngle();
		    MyVector move(a, ped.vel*1.0, 0); //movement in unit time
		    move_dw = move.dw;
		    move_dh = move.dh;
		}

	    int count = 0;
	    
		for(int t=1; t<=5; t++){
			ped_pos.x += move_dw;
		    ped_pos.y += move_dh;
		  
		    double dist = car.vel; // car.vel * 1;
		    int nxt = world.path.forward(car.pos, dist);
		    car.pos = nxt;

		    double d = COORD::EuclideanDistance(world.path[car.pos], ped_pos);

		    if(d <= 1 && count < 3) {count ++; score += 4;}
		    else if(d <= 2 && count < 3) {count ++; score += 2;}
		    else if(d <= 3 && count < 3) {count ++; score += 1;}
		}
	}

	return score;
}

std::vector<double> PedPomdp::ImportanceWeight(std::vector<State*> particles) const
{
 	double total_weight=State::Weight(particles);
 	double new_total_weight=0;
	int particles_num=particles.size();

	std::vector<PomdpState*> pomdp_state_particles;

	std::vector <double> importance_weight;

	bool use_is_despot = false;
	if(use_is_despot == false){
		for(int i=0; i<particles_num;i++){
			importance_weight.push_back(particles[i]->weight);
		}
		return importance_weight;
	}

	for(int i=0; i<particles_num;i++){
		pomdp_state_particles.push_back(static_cast<PomdpState*>(particles[i]));
	}

	for(int i=0; i<particles_num;i++){

		importance_weight.push_back(pomdp_state_particles[i]->weight * ImportanceScore(pomdp_state_particles[i]));
		new_total_weight += importance_weight[i];
	}

	//normalize to total_weight
	for(int i=0; i<particles_num;i++){
		importance_weight[i]=importance_weight[i]*total_weight/new_total_weight;
		assert(importance_weight[i]>0);
	}

	return importance_weight;
}


int PedPomdp::NumObservations() const{
	//cout<<__FUNCTION__<<": Obs space too large! INF used instead"<<endl;
	return std::numeric_limits<int>::max();
}
int PedPomdp::ParallelismInStep() const{
	return ModelParams::N_PED_IN;
}
void PedPomdp::ExportState(const State& state, std::ostream& out) const{
	PomdpState cardriveState=static_cast<const PomdpState&>(state);
	ios::fmtflags old_settings = out.flags();

	int Width=7;
	int Prec=3;
	//out << "Head ";
	out << cardriveState.scenario_id <<" ";
	out << cardriveState.weight <<" ";
	out << cardriveState.num<<" ";
	out << cardriveState.car.dist_travelled <<" "
			<<cardriveState.car.pos<<" "
			<<cardriveState.car.vel<<" ";
	for (int i=0;i<ModelParams::N_PED_IN;i++)
		out << cardriveState.peds[i].goal <<" "<<cardriveState.peds[i].id
		<<" "<<cardriveState.peds[i].pos.x<<" "<<cardriveState.peds[i].pos.y
		<<" "<<cardriveState.peds[i].vel<<" ";

	out /*<< "End"*/<<endl;

	out.flags(old_settings);
}
State* PedPomdp::ImportState(std::istream& in) const{
	PomdpState* cardriveState = memory_pool_.Allocate();

	if (in.good())
	{
		string str;
		while(getline(in, str))
		{
			if(!str.empty())
			{
				istringstream ss(str);

				ss >> cardriveState->scenario_id;
				ss >> cardriveState->weight;
				ss >> cardriveState->num;
				ss >> cardriveState->car.dist_travelled >>cardriveState->car.pos>>cardriveState->car.vel;
				for (int i=0;i<ModelParams::N_PED_IN;i++)
					ss >> cardriveState->peds[i].goal >>cardriveState->peds[i].id
					>>cardriveState->peds[i].pos.x>>cardriveState->peds[i].pos.y
					>>cardriveState->peds[i].vel;
			}
		}
	}

	return cardriveState;
}
void PedPomdp::ImportStateList(std::vector<State*>& particles, std::istream& in) const{
	if (in.good())
	{
		int PID=0;
		string str;
		getline(in, str);
		//cout<<str<<endl;
		istringstream ss(str);
		int size;
		ss>>size;
		particles.resize(size);
		while(getline(in, str))
		{
			if(!str.empty())
			{
				if(PID>=particles.size())
					cout<<"Import particles error: PID>=particles.size()!"<<endl;

				PomdpState* cardriveState = memory_pool_.Allocate();

				istringstream ss(str);

				ss >> cardriveState->scenario_id;
				ss >> cardriveState->weight;
				ss >> cardriveState->num;
				ss >> cardriveState->car.dist_travelled >>cardriveState->car.pos>>cardriveState->car.vel;
				for (int i=0;i<ModelParams::N_PED_IN;i++)
					ss >> cardriveState->peds[i].goal >>cardriveState->peds[i].id
					>>cardriveState->peds[i].pos.x>>cardriveState->peds[i].pos.y
					>>cardriveState->peds[i].vel;
				particles[PID]=cardriveState;
				PID++;

			}
		}
	}
}
