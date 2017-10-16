#include "custom_particle_belief.h"

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
			goal_count[j][pomdp_state->peds[j].goal] += particles[i]->weight;
		}
	}

	const PomdpState* pomdp_state=static_cast<const PomdpState*>(particles.at(0));
	for(int j = 0; j < pomdp_state->num; j ++) {
		cout << "Ped " << pomdp_state->peds[j].id << " Belief is ";
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
		cout<<"** "<<particle->peds[j].id<<"   goal: "<<max_likelihood_goal;
		particle->peds[j].goal = max_likelihood_goal;
		cout << endl;
	}

	particle -> weight = 1.0;
	sample.push_back(particle);
	
	return sample;
}