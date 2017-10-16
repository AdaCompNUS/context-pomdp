#include <ped_path_predictor/world_state.h>

/**
* WORLD STATE HISTORY CLASS
*/
WorldStateHistroy::WorldStateHistroy(ros::NodeHandle &nh){
	
	
    ped_sub_ = nh.subscribe("pedestrian_array", 1, &WorldStateHistroy::PedCallback, this);
    
	starttime = get_time_second();
	last_time = get_time_second();

//	ros::spin();
}

void WorldStateHistroy::UpdatePrefVel(){
    mtx.lock();
    int history_size = history.size();

    std::vector<double> accu_weights; //accumulate weights; used for computing the weighted average vel, i.e., vel_pref
    accu_weights.resize(state_cur.peds.size());
    
    for(int i=0; i<state_cur.peds.size(); i++){
        state_cur.peds[i].vel_pref=state_cur.peds[i].vel_cur;
        accu_weights[i]=1.0;
    }

    for (int i=0; i< state_cur.peds.size(); i++){
        double weight = 1.0;
        for (int step = history_size-1; step >=0; step--){
            weight *= ModelParams::WEIGHT_DISCOUNT;
            for(int j=0; j<history[step].peds.size(); j++){
                if(state_cur.peds[i].id == history[step].peds[j].id){
                    state_cur.peds[i].vel_pref += (history[step].peds[i].vel_cur * weight);
                    accu_weights[i] += weight;
                    break;
                }
            }
        }
    }

    for(int i=0; i<state_cur.peds.size(); i++){
        state_cur.peds[i].vel_pref /= accu_weights[i];
    }
    mtx.unlock();
}


/*void WorldStateHistroy::UpdateVel(){
    mtx.lock();
    int history_size = history.size();
	state_cur = history[history_size-1];

    std::vector<double> accu_weights; //accumulate weights; used for computing the weighted average vel, i.e., vel_pref
    accu_weights.resize(state_cur.peds.size());
    

	for(int i=0; i<state_cur.peds.size(); i++){
		state_cur.peds[i].vel_pref=state_cur.peds[i].vel_cur;
        accu_weights[i]=1.0;
	}

    for (int i=0; i< state_cur.peds.size(); i++){
        double weight = 1.0;
	    for (int step = history_size-2; step >=0; step--){
            weight *= ModelParams::WEIGHT_DISCOUNT;
	        for(int j=0; j<history[step].peds.size(); j++){
                if(state_cur.peds[i].id == history[step].peds[j].id){
                    state_cur.peds[i].vel_pref += (history[step].peds[i].vel_cur * weight);
                    accu_weights[i] += weight;
                    break;
                }
           }
	    }
	}

    for(int i=0; i<state_cur.peds.size(); i++){
        state_cur.peds[i].vel_pref /= accu_weights[i];
    }
	mtx.unlock();
}
*/


double WorldStateHistroy::get_time_second() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

WorldState WorldStateHistroy::get_current_state(){
	    mtx.lock();
	    WorldState tmp = state_cur;
	    mtx.unlock();
	    return tmp;
}

double WorldStateHistroy::Timestamp() {
    return get_time_second()-starttime;
}

void WorldStateHistroy::CleanPed() {
    mtx.lock();
    vector<PedStruct> peds_new;
    for(int i=0;i<state_cur.peds.size();i++)
    {
        bool insert=true;
        double x1,y1;
        x1=state_cur.peds[i].pos.x;
        y1=state_cur.peds[i].pos.y;
        for(int j=0; j<peds_new.size(); j++) {
            double x2,y2;
            x2=peds_new[j].pos.x;
            y2=peds_new[j].pos.y;
            if (fabs(x1-x2)<=ModelParams::OVERLAP_THRESHOLD 
                    && fabs(y1-y2)<=ModelParams::OVERLAP_THRESHOLD) { //two pedestrians overlap; do not insert
                insert=false;
                break;
            }
        }
        // remove pedestrians that have not been updated for a long time because they are probably out of view
        if (Timestamp() - state_cur.peds[i].last_update > ModelParams::OUTDATED_THRESHOLD) insert=false; 
        if (insert)
            peds_new.push_back(state_cur.peds[i]);
    }
    state_cur.peds=peds_new;
    mtx.unlock();
}

void WorldStateHistroy::UpdatePed(const PedStruct& ped){
    mtx.lock();
    int i=0;
    for(;i<state_cur.peds.size();i++) {
        if (state_cur.peds[i].id==ped.id) {
            //found the corresponding ped,update the pose
            state_cur.peds[i].vel_cur = (ped.pos - state_cur.peds[i].pos)/(Timestamp() - state_cur.peds[i].last_update);
            if(fabs(state_cur.peds[i].vel_cur.x) < ModelParams::VEL_ZERO_THRESHOLD) state_cur.peds[i].vel_cur.x = 0;
            if(fabs(state_cur.peds[i].vel_cur.y) < ModelParams::VEL_ZERO_THRESHOLD) state_cur.peds[i].vel_cur.y = 0;

            state_cur.peds[i].pos.x=ped.pos.x;
            state_cur.peds[i].pos.y=ped.pos.y;
            //state_cur.peds[i].vel = state_cur.peds[i].vel * ALPHA + dist * ModelParams::control_freq * (1-ALPHA);
            state_cur.peds[i].last_update = Timestamp();

            break;
        }
        if (fabs(state_cur.peds[i].pos.x-ped.pos.x)<=ModelParams::OVERLAP_THRESHOLD 
                && fabs(state_cur.peds[i].pos.y-ped.pos.y)<=ModelParams::OVERLAP_THRESHOLD)   //overlap
        {
          	mtx.unlock();
           	return;
        }
    }
    if (i==state_cur.peds.size()) {
        //not found, new ped
        state_cur.peds.push_back(ped);
        state_cur.peds.back().last_update = Timestamp();
    }
    mtx.unlock();
}

void WorldStateHistroy::UpdatePed(WorldState& state, const PedStruct& ped){
    int i=0;
    for(;i<state.peds.size();i++) {
        if (state.peds[i].id==ped.id) {
            //found the corresponding ped,update the pose
            state.peds[i].vel_cur = (ped.pos - state_cur.peds[i].pos)/(Timestamp() - state_cur.peds[i].last_update);
            state.peds[i].pos.x=ped.pos.x;
            state.peds[i].pos.y=ped.pos.y;
            //state.peds[i].vel = state.peds[i].vel * ALPHA + dist * ModelParams::control_freq * (1-ALPHA);
            state.peds[i].last_update = Timestamp();
            break;
        }
        if (fabs(state.peds[i].pos.x-ped.pos.x)<=ModelParams::OVERLAP_THRESHOLD 
                && fabs(state.peds[i].pos.y-ped.pos.y)<=ModelParams::OVERLAP_THRESHOLD)   //overlap
            return;
    }
    if (i==state.peds.size()) {
        //not found, new ped
        state.peds.push_back(ped);
        state.peds.back().last_update = Timestamp();
    }
}

void WorldStateHistroy::PrintState(WorldState& state){
	for(int i=0;i<state.peds.size();i++) {
		std::cout<<"id:  "<<state.peds[i].id<<"  pos: "<<state.peds[i].pos<<"  cur_vel: "<<state.peds[i].vel_cur<<"   pref_vel: "<<state.peds[i].vel_pref<<endl;
    }
    std::cout<<std::endl;
}

void WorldStateHistroy::AddState(WorldState new_state){
    mtx.lock();
    history.push_back(new_state);
    if(history.size()>=ModelParams::N_HISTORY_STEP){
        history.erase(history.begin());
    }
    mtx.unlock();
}

void WorldStateHistroy::PedCallback(cluster_assoc::pedestrian_arrayConstPtr ped_array)
{
	if (get_time_second() - last_time <=0.25) return; // to control the frequency of callback function being called.

    for (int i=0; i<ped_array->pd_vector.size(); i++){
		//if(pd_vector[i].confidence > 1.0)
		{
			PedStruct ped;
			ped.id = ped_array->pd_vector[i].object_label;
			ped.pos.x = ped_array->pd_vector[i].global_centroid.x;
			ped.pos.y = ped_array->pd_vector[i].global_centroid.y;
			UpdatePed(ped);
		}

	}
    
    CleanPed();
    UpdatePrefVel();
    AddState(state_cur);
//    PrintState(state_cur);

    last_time = get_time_second();
}