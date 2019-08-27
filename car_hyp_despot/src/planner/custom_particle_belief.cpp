#include "custom_particle_belief.h"
#include <despot/core/particle_belief.h>
#include "WorldModel.h"
#include "ped_pomdp.h"
#include <iostream>
#include "world_simulator.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

// static WorldStateTracker stateTracker (Simulator::worldModel);
// static WorldBeliefTracker beliefTracker(Simulator::worldModel, stateTracker);
static AgentStruct sorted_peds_list[ModelParams::N_PED_WORLD];
static AgentStruct reordered_peds_list[ModelParams::N_PED_WORLD];


MaxLikelihoodScenario::MaxLikelihoodScenario(vector<State*> particles, const DSPOMDP* model,
		Belief* prior, bool split): ParticleBelief(particles, model,
		prior, split){

}

vector<State*> MaxLikelihoodScenario::SampleCustomScenarios(int num, vector<State*> particles,
	const DSPOMDP* model) const{
	//return particles;
	vector<State*> sample;

	cout<<"++++++++=========++++++++"<<endl;

	double goal_count[10][10]={{0}};
	if(particles.size() == 0)
		return sample;
	

	for(int i = 0; i < particles.size(); i ++) {
		const PomdpState* pomdp_state=static_cast<const PomdpState*>(particles.at(i));
		for(int j = 0; j < pomdp_state->num; j ++) {
			goal_count[j][pomdp_state->agents[j].intention] += particles[i]->weight;
		}
	}

	const PomdpState* pomdp_state=static_cast<const PomdpState*>(particles.at(0));
	for(int j = 0; j < pomdp_state->num; j ++) {
		cout << "Ped " << pomdp_state->agents[j].id << " Belief is ";
		for(int i = 0; i < 7; i ++) {
			cout << (goal_count[j][i] + 0.0) <<" ";
		}
		cout << endl;
	}
	
	PomdpState* particle = static_cast<PomdpState*>(model->Copy(pomdp_state));

	for(int j = 0; j < pomdp_state->num; j ++) {
		int max_likelihood_goal = -1;
		double max_likelihood = -1;
		for(int i = 0; i < 7; i ++) {
			if(goal_count[j][i] > max_likelihood) {
				max_likelihood = goal_count[j][i];
				max_likelihood_goal = i;
			}
		}
		cout<<"** "<<particle->agents[j].id<<"   goal: "<<max_likelihood_goal;
		particle->agents[j].intention = max_likelihood_goal;
		cout << endl;
	}

	particle -> weight = 1.0;
	sample.push_back(particle);
	
	return sample;
}

AgentStruct NewDummyAgent(int id){
	return AgentStruct(COORD(dummy_pos_value, dummy_pos_value), -1, id);
}

PedPomdpBelief::PedPomdpBelief(vector<State*> particles, const DSPOMDP* model):
	ParticleBelief( particles, model),
	world_model_(SimulatorBase::worldModel){

	stateTracker = SimulatorBase::stateTracker;
	beliefTracker = new WorldBeliefTracker(SimulatorBase::worldModel, *stateTracker);
//	logd << "[PedPomdpBelief::PedPomdpBelief] Belief tracker initialized: " << beliefTracker << endl;

	agentPredictionPub_ = nh.advertise<sensor_msgs::PointCloud>("ped_prediction", 1);
	markers_pub = nh.advertise<visualization_msgs::MarkerArray>("pomdp_belief",1);
	believesPub_ = nh.advertise<car_hyp_despot::peds_believes>("peds_believes",1);
	plannerAgentsPub_=nh.advertise<sensor_msgs::PointCloud>("planner_peds",1); // for visualization
}

Belief* PedPomdpBelief::MakeCopy() const{
	std::vector<State*> particles = particles_;
	return new PedPomdpBelief(particles, static_cast<const DSPOMDP*>(model_));
}


void PedPomdpBelief::UpdateState(const PomdpStateWorld* src_world_state, WorldModel& world_model){

//	stateTracker->removePeds();
//	stateTracker->updateCar(src_world_state->car);
//
//	cout << "[UpdateState] \n";
//
//	raise(SIGABRT);
//
//	//update the agents in stateTracker
//	for(int i=0; i<src_world_state->num; i++) {
//		Pedestrian p(src_world_state->agents[i].pos.x, src_world_state->agents[i].pos.y, src_world_state->agents[i].id);
//		stateTracker->updatePed(p, false); // true: print debug info
//	}
}

void PedPomdpBelief::SortAgents(PomdpState* sorted_search_state, const PomdpStateWorld* src_world_state){
	std::vector<WorldStateTracker::AgentDistPair> sorted_agents = stateTracker->getSortedAgents();

	//update s.agents to the nearest n_peds agents
	sorted_search_state->num=min(src_world_state->num, ModelParams::N_PED_IN);
	for(int i=0; i<sorted_search_state->num; i++) {
		if(i < sorted_agents.size())
		{
			int agentid = sorted_agents[i].second->id;
			// Search for the ped in src_world_state
			int j=0;
			for(;j<src_world_state->num;j++) {
				if (agentid==src_world_state->agents[j].id) {
					//found the corresponding ped, record the order
					sorted_search_state->agents[i]=src_world_state->agents[j];
					break;
				}
			}
			if(j==src_world_state->num)//ped not found
			{
				sorted_search_state->agents[i]=NewDummyAgent(-1);
			}
		}
	}
}

void PedPomdpBelief::ReorderAgents(PomdpState* target_search_state,
		const PomdpState* ref_search_state,
		const PomdpStateWorld* src_history_state){
	//update s.agents to the nearest n_peds agents
	target_search_state->num = ref_search_state->num;
	for(int i=0; i < ref_search_state->num; i++) {
		int agentid = ref_search_state->agents[i].id;
		//Search for the ped in world_state
		int j=0;
		for(;j<src_history_state->num;j++) {
			if (agentid==src_history_state->agents[j].id) {
				//found the corresponding ped, record the order
				target_search_state->agents[i]=src_history_state->agents[j];
				break;
			}
		}
		if (j >= src_history_state->num) {
			//not found, new ped
			target_search_state->agents[i]=NewDummyAgent(-1);
		}
	}
}


bool PedPomdpBelief::DeepUpdate(const std::vector<const State*>& state_history,
		std::vector<State*>& state_history_for_search,
		const State* cur_state,
		State* cur_state_for_search,
		ACT_TYPE action){
	//Update current belief

	try{

		PomdpState* cur_state_search = static_cast<PomdpState*>(cur_state_for_search);

		bool reorder = false;

		if (reorder){
			const PomdpStateWorld* cur_world_state = static_cast<const PomdpStateWorld*>(cur_state);

			for (int hi=0;hi< state_history.size(); hi++){
			   const PomdpStateWorld* hist_state =
					   static_cast<const PomdpStateWorld*>(state_history[hi]);
			   PomdpState* hist_state_search =
					   static_cast<PomdpState*>(state_history_for_search[hi]);

			   ReorderAgents(hist_state_search, cur_state_search, hist_state);
			}
		}

		// DEBUG("Update and reorder the belief distributions for agents");
		//
		beliefTracker->update();

		if (Globals::config.silence == false && DESPOT::Debug_mode)
			beliefTracker->printBelief();

		// DEBUG("publish planner peds topic");
		publishPlannerPeds(*cur_state_search);
	}
	catch (exception e) {
		cerr << "Exception caught in " << __FUNCTION__ << " " << e.what() << endl;
	}
}

bool PedPomdpBelief::DeepUpdate(const State* cur_state){

	if (cur_state == NULL)
		return false;

	//Update current belief
	const PomdpStateWorld* cur_world_state = static_cast<const PomdpStateWorld*>(cur_state);

	//Sort pedestrians in the current state and update the current search state
//	UpdateState(cur_world_state, world_model_);

	//Update and reorder the belief distributions for agents
	beliefTracker->update();

	return true;
}

void PedPomdpBelief::ResampleParticles(const PedPomdp* model, bool do_prediction){

	logd << "[PedPomdpBelief::ResampleParticles] Sample from belief tracker " 
		<< beliefTracker << endl;

	assert(beliefTracker);

	// bool do_prediction = true;
	bool use_att_mode = model->use_rvo_in_search;
	vector<PomdpState> samples = beliefTracker->sample(
		max(2000,5*Globals::config.num_scenarios), do_prediction, use_att_mode);

	logd << "[PedPomdpBelief::ResampleParticles] Construct raw particles" << endl;

	vector<State*> particles = model->ConstructParticles(samples);

	if(DESPOT::Debug_mode)
		std::srand(0);

	logd << "[PedPomdpBelief::ResampleParticles] Construct final particles" << endl;

	// TODO: free old particles
	for (int i = 0; i < particles_.size(); i++)
		model_->Free(particles_[i]);
	particles_ = particles;

	if (fabs(State::Weight(particles) - 1.0) > 1e-6) {
			loge << "[PedPomdpBelief::PedPomdpBelief] Particle weights sum to " << State::Weight(particles) << " instead of 1" << endl;
			exit(1);
		}

	bool split = true;
	if (split) {
		// Maintain more particles to avoid degeneracy
		while (2 * num_particles_ < 5000)
			num_particles_ *= 2;
		if (particles_.size() < num_particles_) {
			logi << "[PedPomdpBelief::ResampleParticles] Splitting " << particles_.size()
				<< " particles into " << num_particles_ << " particles." << endl;
			vector<State*> new_particles;
			int n = num_particles_ / particles_.size();
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < particles_.size(); j++) {
					State* particle = particles_[j];
					State* copy = model_->Copy(particle);
					copy->weight /= n;
					new_particles.push_back(copy);
				}
			}

			for (int i = 0; i < particles_.size(); i++)
				model_->Free(particles_[i]);

			particles_ = new_particles;
		}
	}

	if (fabs(State::Weight(particles_) - 1.0) > 1e-6) {
		loge << "[PedPomdpBelief::ResampleParticles] Particle weights sum to " << State::Weight(particles_) << " instead of 1" << endl;
		exit(1);
	}

	random_shuffle(particles_.begin(), particles_.end());
	// cerr << "Number of particles in initial belief: " << particles_.size() << endl;

	if (prior_ == NULL) {
		for (int i = 0; i < initial_particles_.size(); i++)
			model_->Free(initial_particles_[i]);
		initial_particles_.resize(0);
		for (int i = 0; i < particles_.size(); i++)
			initial_particles_.push_back(model_->Copy(particles_[i]));
	}

	// for (int i =0; i < 10; i++){
	// 	static_cast<const PedPomdp*>(model_)->validate_state(
	// 		*static_cast<PomdpState*>(particles_[i]));
	// }
}

State* PedPomdpBelief::GetParticle(int i){
	return particles_[i];
}

void PedPomdpBelief::Update(ACT_TYPE action, OBS_TYPE obs) {
	logi << "[PedPomdpBelief::Update] Doing nothing " << endl;

	return ;
}

void PedPomdpBelief::publishAgentsPrediciton() {
    vector<AgentStruct> agents = beliefTracker->predictAgents();
    sensor_msgs::PointCloud pc;
    pc.header.frame_id=ModelParams::rosns + "/map";
    pc.header.stamp=ros::Time::now();
    for(const auto& ped: agents) {
        geometry_msgs::Point32 p;
        p.x = ped.pos.x;
        p.y = ped.pos.y;
        p.z = 1.0;
        pc.points.push_back(p);
    }
    agentPredictionPub_.publish(pc);
}


void PedPomdpBelief::publishBelief()
{
	//vector<vector<double> > ped_beliefs=RealSimulator->GetBeliefVector(solver->root_->particles());	
	//cout<<"belief vector size "<<ped_beliefs.size()<<endl;
	int i=0;
	car_hyp_despot::peds_believes pbs;	
	for(auto & kv: beliefTracker->agent_beliefs)
	{
		publishMarker(i++,kv.second);
		car_hyp_despot::ped_belief pb;
		AgentBelief belief = kv.second;
		pb.ped_x=belief.pos.x;
		pb.ped_y=belief.pos.y;
		pb.ped_id=belief.id;

		for (auto& prob_goals : belief.prob_modes_goals)
			for(auto & v : prob_goals)
				pb.belief_value.push_back(v);

		pbs.believes.push_back(pb);
	}
	pbs.cmd_vel=stateTracker->carvel;
	pbs.robotx=stateTracker->carpos.x;
	pbs.roboty=stateTracker->carpos.y;
	believesPub_.publish(pbs);
	markers_pub.publish(markers);
	markers.markers.clear();
}


void PedPomdpBelief::publishMarker(int id,AgentBelief & ped)
{
	//cout<<"belief vector size "<<belief.size()<<endl;
	std::vector<double> belief = ped.prob_modes_goals[0];
	uint32_t shape = visualization_msgs::Marker::CUBE;
    uint32_t shape_text=visualization_msgs::Marker::TEXT_VIEW_FACING;
	for(int i=0;i<belief.size();i++)
	{
		visualization_msgs::Marker marker;
		visualization_msgs::Marker marker_text;

		marker.header.frame_id=ModelParams::rosns + "/map";;
		marker.header.stamp=ros::Time::now();
		marker.ns="basic_shapes";
		marker.id=id*ped.prob_modes_goals[0].size()+i;
		marker.type=shape;
		marker.action = visualization_msgs::Marker::ADD;

		marker_text.header.frame_id=ModelParams::rosns + "/map";;
		marker_text.header.stamp=ros::Time::now();
		marker_text.ns="basic_shapes";
		marker_text.id=id*ped.prob_modes_goals[0].size()+i+1000;
		marker_text.type=shape_text;
		marker_text.action = visualization_msgs::Marker::ADD;


		double px=0,py=0;
		px=ped.pos.x;
		py=ped.pos.y;
		marker_text.pose.position.x = px;
		marker_text.pose.position.y = py;
		marker_text.pose.position.z = 0.5;
		marker_text.pose.orientation.x = 0.0;
		marker_text.pose.orientation.y = 0.0;
		marker_text.pose.orientation.z = 0.0;
		marker_text.pose.orientation.w = 1.0;
	//	cout<<"belief entries "<<px<<" "<<py<<endl;
		// Set the scale of the marker -- 1x1x1 here means 1m on a side
		//if(marker.scale.y<0.2) marker.scale.y=0.2;
		marker_text.scale.z = 1.0;
		//
		// Set the color -- be sure to set alpha to something non-zero!
		//marker.color.r = 0.0f;
		//marker.color.g = 1.0f;
		//marker.color.b = 0.0f;
		//marker.color.a = 1.0;
		marker_text.color.r = 0.0;
        marker_text.color.g = 0.0;
		marker_text.color.b = 0.0;//marker.lifetime = ros::Duration();
		marker_text.color.a = 1.0;
        marker_text.text = to_string(ped.id); 

		ros::Duration d1(1.0/ModelParams::control_freq);
		marker_text.lifetime=d1;


		px=0,py=0;
		px=ped.pos.x;
		py=ped.pos.y;
		marker.pose.position.x = px+i*0.7;
		marker.pose.position.y = py+belief[i]*2;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
	//	cout<<"belief entries "<<px<<" "<<py<<endl;
		// Set the scale of the marker -- 1x1x1 here means 1m on a side
		marker.scale.x = 0.5;
		marker.scale.y = belief[i]*4;
		//if(marker.scale.y<0.2) marker.scale.y=0.2;
		marker.scale.z = 0.2;
		//
		// Set the color -- be sure to set alpha to something non-zero!
		//marker.color.r = 0.0f;
		//marker.color.g = 1.0f;
		//marker.color.b = 0.0f;
		//marker.color.a = 1.0;
		marker.color.r = marker_colors[i][0];
        marker.color.g = marker_colors[i][1];
		marker.color.b = marker_colors[i][2];//marker.lifetime = ros::Duration();
		marker.color.a = 1.0;

		ros::Duration d2(1.0/ModelParams::control_freq);
		marker.lifetime=d2;
		markers.markers.push_back(marker);
        //markers.markers.push_back(marker_text);
	}
}


void PedPomdpBelief::publishPlannerPeds(const State &s)
{
	const PomdpState & state=static_cast<const PomdpState&> (s);
	sensor_msgs::PointCloud pc;
	pc.header.frame_id=ModelParams::rosns+"/map";
	pc.header.stamp=ros::Time::now();
	for(int i=0;i<state.num;i++) {
		geometry_msgs::Point32 p;
		p.x=state.agents[i].pos.x;
		p.y=state.agents[i].pos.y;
		p.z=1.5;
		pc.points.push_back(p);
	}
	plannerAgentsPub_.publish(pc);	
}
