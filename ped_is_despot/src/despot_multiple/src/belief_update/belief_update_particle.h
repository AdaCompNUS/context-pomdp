#ifndef PARTICLE_FILTER_UPDATE
#define PARTICLE_FILTER_UPDATE

#include "belief_update/belief_update.h"

#include "problems/pedestrian_changelane/pedestrian_state.h"

PedestrianState ObsToState(uint64_t obs);

//#include "problems/pedestrian/pedestrian.h"

/* This class implements a strategy for sequential importance resampling of
 * particles. Its time and space complexity is linear in the number of
 * particles.
 */
template<typename T>
class ParticleFilterUpdate : public BeliefUpdate<T> {
 public:
  ParticleFilterUpdate(unsigned belief_update_seed, const Model<T>& model) 
      : BeliefUpdate<T>(belief_update_seed, model)
  {}

 protected:
  static constexpr double NUM_EFF_PARTICLE_FRACTION = 0.10;

 private:
  vector<Particle<T>*> UpdateImpl(
      const vector<Particle<T>*>& particles,
      int N,
      int act,
	  T obs_state,
	  T obs_state_old);
};

template<typename T>
constexpr double ParticleFilterUpdate<T>::NUM_EFF_PARTICLE_FRACTION;

template<typename T>
vector<Particle<T>*> ParticleFilterUpdate<T>::UpdateImpl(
    const vector<Particle<T>*>& particles,
    int N,
    int act,
	T obs_state,
	T obs_state_old) {

  vector<Particle<T>*> ans;
  double reward;

  if(ModelParams::debug)
  {
	  cout<<"Belief before update "<<endl;
	  this->model_.Statistics(particles);
  }

  PedestrianState new_state = obs_state_old;
  for (auto p: particles) {
    PedestrianState& old_state = p->state;

    if (new_state.num != old_state.num) { // Sanity check
		cout << "old state" << endl;
		this->model_.PrintState(old_state);
		cout << "new  state" << endl;
		this->model_.PrintState(new_state);
	}
    
	assert(new_state.num == old_state.num);
	// Copy goal and id
    for (int i=0; i<old_state.num; i++) {
        new_state.PedPoses[i].second = old_state.PedPoses[i].second;
        new_state.PedPoses[i].third = old_state.PedPoses[i].third;
    }
    double obs_prob = this->model_.TransProbJoint(old_state, new_state, act);
	//this->model_.PrintState(new_state);
	// cout << "obs_prob = " << obs_prob << endl;
	assert(obs_prob > 0);
    if (obs_prob > 0) {
        Particle<T>* new_particle = this->model_.Copy(p);
        new_particle->wt = (p->wt + 1e-6) * obs_prob;
		//new_particle->wt = (p->wt) * obs_prob;
        new_particle->state = new_state;
        ans.push_back(new_particle);
    }
  }

  // Step forward all particles
  /*
  int lcount=0;
  while(ans.size()<100&&lcount<100){
		  for (auto p: particles) {
		  double random_num = (double)rand_r(&(this->belief_update_seed_)) / RAND_MAX;
		  Particle<T>* new_particle = this->model_.Copy(p);
		  this->model_.Step(new_particle->state, random_num, act, reward);
		  double obs_prob = this->model_.ObsProb(obs, new_particle->state, act);
		  //this->model_.PrintState(new_particle->state);
		  //cout<<"observation "<<obs<<endl;
		  if (obs_prob) {
		  new_particle->wt = p->wt * obs_prob;
		  ans.push_back(new_particle);

		  }
		  else
		  this->model_.Free(new_particle);

		  }
		  lcount++;
  }
  */


//  if (ans.empty()) {
  if (ans.size()<10) {
    // No resulting state is consistent with the given observation, so create
    // states randomly until we have enough that are consistent.
    cerr << "WARNING: Particle filter empty. Bootstrapping with random states"
         << endl;
    cout << "WARNING: Particle filter empty. Bootstrapping with random states"
         << endl;
	
	
	for(int i=0;i<ans.size();i++)
		this->model_.Free(ans[i]);

	ans.clear();
	cout<<"N= "<<N<<endl;
	int n_sampled=0;
	double obs_prob=1.0;
	for (auto p: particles) {
        Particle<T>* new_particle = this->model_.Allocate();
        *new_particle = {p->state, n_sampled++, obs_prob};
        ans.push_back(new_particle);
		if(ans.size()>=N) break;
	}

	/*	
    int n_sampled = 0;
    while (n_sampled < N) {
      T s = this->model_.RandomState(this->belief_update_seed_, obs_state);
      double obs_prob = this->model_.ObsProb(obs, s, act);
	  //this->model_.PrintState(s);
      //double obs_prob=1;
	  if (obs_prob) {
        Particle<T>* new_particle = this->model_.Allocate();
        *new_particle = {s, n_sampled++, obs_prob};
        ans.push_back(new_particle);
      }
    }
	*/
    //this->Normalize(ans);
   // return ans;
  }

  /*
  cout<<"particle ids after update"<<endl;
  for(int i=0;i<ans.size();i++)
  {
  	cout<<ans[i]->id<<" ";	
  }
*/
  this->Normalize(ans);
  if(ModelParams::debug)
  {
	  cout<<"After update "<<endl;
	  this->model_.Statistics(ans);
  }

  // Remove all particles below a threshold
	/* // This is buggy
  auto new_last = remove_if (ans.begin(),
                            ans.end(),
                            [&](const Particle<T>* p) { 
                              return p->wt < this->PARTICLE_WT_THRESHOLD; 
                            });
  if (new_last != ans.end()) {
    for (auto it = new_last; it != ans.end(); it++)
      this->model_.Free(*it); // actually freeing particles which have been moved to front
    ans = decltype(ans)(ans.begin(), new_last);

    this->Normalize(ans);
  }
	*/
  /*
	int cur = 0;
	auto last = ans.begin();
	for(auto particle : ans) {
		if(particle->wt < this->PARTICLE_WT_THRESHOLD)
			this->model_.Free(particle);
		else {
			ans[cur++] = particle;
			last ++;
		}
	}
	ans = decltype(ans)(ans.begin(), last);
	*/

  // Resample if we have < N particles or # effective particles drops below the threshold
  double num_eff_particles = 0;
  for (auto it: ans)
    num_eff_particles += it->wt * it->wt;
  num_eff_particles = 1 / num_eff_particles;
  cerr << "num_eff_particles " << num_eff_particles << endl;
  if(num_eff_particles<30) {
	cout<<"print particle weight"<<endl;
  	for(auto it:ans)
		cout<<it->wt<<endl;
  }
  if (num_eff_particles < N * NUM_EFF_PARTICLE_FRACTION || ans.size() < N) {

	for (int i=0; i<N; i++) {
        Particle<T>* new_particle = this->model_.Allocate();
		T s = this->model_.RandomState(this->belief_update_seed_, obs_state_old);
        *new_particle = {s, i, 0.1 / N};
        ans.push_back(new_particle);
	}
    this->Normalize(ans);

    auto resampled_ans = this->Sample(ans, N);
    for (auto it: ans)
      this->model_.Free(it);
    ans = resampled_ans;
	cerr << "DEBUG: Resampled " << ans.size() << " particles in belief" << endl;
	if(ModelParams::debug)
	{
		cout<<"After resample "<<endl;
		this->model_.Statistics(ans);
	}
  } 

  
  this->model_.ModifyObsStates(ans,obs_state,this->belief_update_seed_);
  if(ModelParams::debug)
  {
	  cout<<"After Modify "<<endl;
	  this->model_.Statistics(ans);
  }
  /*
  cout<<"particle ids after modify"<<endl;
  for(int i=0;i<ans.size();i++)
  {
  	cout<<ans[i]->id<<" ";	
  }
  cout<<endl;
  */

  return ans;
}

#endif
