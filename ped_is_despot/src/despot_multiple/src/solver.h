#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include "vnode.h"
#include "belief_update/belief_update.h"
#include "globals.h"
#include "history.h"
#include "lower_bound/lower_bound.h"
#include "model.h"
#include "qnode.h"
#include "random_streams.h"
#include "upper_bound/upper_bound.h"
#include "lower_bound/lower_bound_policy_suffix.h"

// This class implements the core of the algorithm.
template<typename T>
class Solver {
 public:
  Solver(const Model<T>& model,
         const vector<pair<T, double>>& initial_belief,
         ILowerBound<T>& lb,
         IUpperBound<T>& ub,
         BeliefUpdate<T>& bu,
         const RandomStreams& random_streams)
      : model_(model),
        initial_belief_(initial_belief),
        lb_(lb),
        ub_(ub),
        bu_(bu),
        streams_(random_streams),
        history_(History())
  {}


  unique_ptr<VNode<T>> root_; // Root of the search tree
  void Init();

  // Builds a search tree using at most @maxTime CPU time. This time does not
  // include the time for pruning. Returns the best action to take.
  int Search(double max_time, int& n_trials);

  // True iff all particles are in the terminal state
  bool Finished() const;

  // Updates the agent's current belief and initializes a new tree rooted at
  // the new belief.
  void UpdateBelief(int act, T  new_state,T new_state_old);

  // Resets the solver to its initial state (as if it were just constructed and
  // Init()-ialized). Useful to run experiments without reconstructing the 
  // solver every time, avoiding the additional overhead.
  void Reset();

  const RandomStreams& random_streams() const { return streams_; }

	// Print the tree roooted at node 
	void PrintTree(unique_ptr<VNode<T>>& node);

	// Record all the (history, optimal action) pairs for the suffix policy
	void RetrieveHistoryActionMapping(SuffixPolicyLowerBound<T>* suffix_policy);
	// Helper function
	private:	void RetrieveHistoryActionMapping(unique_ptr<VNode<T>>& node,
			History& history,
			SuffixPolicyLowerBound<T>* suffix_policy);

 private:
  // Performs one trial and backup. Params:
  // @debug: Whether to print debugging output.
  // @currNode: current VNode.
  // Return value: number of new search tree nodes added during this trial
  int Trial(unique_ptr<VNode<T>>& node, bool debug=false);
        
  // Expands a fringe belief node one more level
  void ExpandOneStep(unique_ptr<VNode<T>>& node, bool debug=false);

  const Model<T>& model_;
  vector<pair<T, double>> initial_belief_;
  vector<Particle<T>*> belief_;
  ILowerBound<T>& lb_;
  IUpperBound<T>& ub_;
  BeliefUpdate<T>& bu_;
	const RandomStreams& streams_;
	History history_; // Updated after each belief update and during search

  int default_action_; // Action for the lower bound at the root
};

template<typename T>
void Solver<T>::Init() {
  // Construct pool
  vector<Particle<T>*> pool;
  for (auto& it: initial_belief_) {
    Particle<T>* particle = model_.Allocate();
    *particle = {it.first, 0, it.second};
    pool.push_back(particle);
  }
  random_shuffle(pool.begin(), pool.end());

  belief_ = bu_.Sample(pool, Globals::config.n_belief_particles);
  cerr << "DEBUG: Initialized initial belief with " << belief_.size() << endl;

  auto particles = bu_.Sample(belief_, Globals::config.n_particles);
  cerr << "DEBUG: Sampled " << particles.size() << " particles from initial belief" << endl; 

  pair<double, int> lb = lb_.LowerBound(particles, model_, 0, history_);
  default_action_ = lb.second;
  double            ub = ub_.UpperBound(particles, model_, 0, history_);
  Globals::ValidateBounds(lb.first, ub);
  root_ = unique_ptr<VNode<T>>(
      new VNode<T>(std::move(particles), lb.first, ub, 0, 1, false));

  // Destruct pool
  for (auto p: pool)
    model_.Free(p);
}

template<typename T>
void Solver<T>::Reset() {
  bu_.Reset();
  history_.Truncate(0);
  Init();
}

template<typename T>
int Solver<T>::Search(double max_time, int& n_trials) {
	cerr << "Starting despot search..." << endl;

  clock_t begin = clock();
  n_trials = 0;
  while ((double)(clock() - begin) / CLOCKS_PER_SEC < max_time
        && Globals::ExcessUncertainty(root_->lbound(), root_->ubound(), root_->lbound(),
                             root_->ubound(), 0) > 1e-6) {
    Trial(root_);
    n_trials++;
  }
  cout<<"number of trials "<<n_trials<<endl;

	lb_.CollectSearchInformation(this);

  if (Globals::config.pruning_constant) {
    // Number of non-child belief nodes in the search tree that are pruned
    int total_pruned = 0;
    root_->Prune(total_pruned);
    int act = root_->pruned_action(); 
    cout << "Action = " << (act == -1 ? default_action_ : act) << endl;
	cout<<"start printing tree "<<endl;
	//PrintTree(root_);
    return act == -1 ? default_action_ : act;
  }
  else if (!(root_->in_tree())) {
	cout << "Default" << endl;
	cout << "Action = " << default_action_ << endl;
    return default_action_;
  } else {
	cout << "Optimal" << endl;
	cout << "Action = " << root_->OptimalAction() << endl;
	cout<<"start printing tree "<<endl;
//	PrintTree(root_);
    return root_->OptimalAction();
  }
  cerr << "Despot search done!" << endl;
}

template<typename T>
void Solver<T>::ExpandOneStep(unique_ptr<VNode<T>>& node, bool debug) {
  auto& particles = node->particles();
  vector<QNode<T>>& qnodes = node->Children();
  qnodes.reserve(model_.NumActions());
  double q_star = -Globals::INF;

  for (int a = 0; a < model_.NumActions(); a++) {
    if (debug)
      cerr << "a = " << a << endl; 

		//cout << "Expanding for a = " << a << endl;
    double first_step_reward = 0;
    // Map from observation to set of particles that produced that observation
    MAP<uint64_t, vector<Particle<T>*>> obs_to_particles;

    for (Particle<T>* p: particles) {
      Particle<T>* new_particle = model_.Copy(p);
      uint64_t obs; 
      double reward;
			//cout << "Before" << endl;
			//model_.PrintState(new_particle->state);
      model_.Step(new_particle->state, streams_.Entry(p->id, node->depth()), 
                  a, reward, obs);
			//cout << "After" << endl;
			//model_.PrintState(new_particle->state);
			//cout << "obs = " << obs << endl;
      if (model_.IsTerminal(new_particle->state))
        assert(obs == model_.TerminalObs());
      obs_to_particles[obs].push_back(new_particle);
      first_step_reward += reward * p->wt;
    }
    first_step_reward /= node->weight();

    if (debug) 
      cerr << "node weight = " << node->weight() << endl;

    qnodes.push_back(QNode<T>(obs_to_particles, node->depth(), a, 
                       first_step_reward, history_, model_, lb_, ub_, false));
    auto& qnode = qnodes.back();
    double remaining_reward = qnode.UpperBound();
    if (first_step_reward + Globals::config.discount * remaining_reward >
        q_star + Globals::TINY) {
      q_star = first_step_reward + Globals::config.discount * remaining_reward;
      node->set_best_ub_action(a);
    }

    if (debug) { 
      cerr << "first_step_reward = " << first_step_reward << endl;
      cerr << "remaining_reward = " << remaining_reward << endl;
    }
  }

  assert(node->best_ub_action() != -1);
}

template<typename T>
int Solver<T>::Trial(unique_ptr<VNode<T>>& node, bool debug) {
  if (node->depth() >= Globals::config.search_depth || 
      model_.IsTerminal(node->particles()[0]->state))
    return 0;
/*
	for(auto particle : node->particles())
		model_.PrintState(particle->state);*/

  if (debug) 
    cerr << "TRIAL\n" << "depth = " << node->depth() << "\n";

  if (node->Children().empty())
    ExpandOneStep(node, debug);

  int a_star = node->best_ub_action();
  if (debug)
    cerr << "a_star = " << a_star << endl;

  vector<QNode<T>>& qnodes = node->Children();
  int num_nodes_added = 0;

  pair<uint64_t, double> best_WEUO = qnodes[a_star].BestWEUO(root_);
  if (best_WEUO.second > 0) {
    if (debug) 
      cerr << "o_star = " << best_WEUO.first << endl;
		history_.Add(a_star, best_WEUO.first);
    num_nodes_added = Trial(qnodes[a_star].Belief(best_WEUO.first), debug);
		history_.RemoveLast();
  }
  node->set_n_tree_nodes(node->n_tree_nodes() + num_nodes_added);

  // Backup
  if (debug) 
    cerr << "Backing up\n";

  double new_lbound = max(node->lbound(), 
      qnodes[a_star].first_step_reward() +
      Globals::config.discount * qnodes[a_star].LowerBound());
  node->set_lbound(new_lbound);

  // As the upper bound of the action a_star may drop below the upper bound
  // of another action, we need to check all actions, unlike the lower bound.
  node->set_ubound(-Globals::INF);
  for (int a = 0; a < model_.NumActions(); a++) {
    double ub = qnodes[a].first_step_reward() +
                Globals::config.discount *
                qnodes[a].UpperBound();
    if (ub > node->ubound()) {
      node->set_ubound(ub);
      node->set_best_ub_action(a);
    }
  }

  // Sanity check
  if (node->lbound() > node->ubound() + Globals::TINY) {
    cerr << "depth = " << node->depth() << endl;
	cerr << "Larger = " << (node->lbound() > node->ubound() + Globals::TINY) << endl;
    cerr << node->lbound() << " " << node->ubound() << endl;
    assert(false);
  }

  if (!node->in_tree()) {
    node->set_in_tree(true);
    node->set_n_tree_nodes(node->n_tree_nodes() + 1);
    num_nodes_added++;
  }

  return num_nodes_added;
}

template<typename T>
void Solver<T>::PrintTree(unique_ptr<VNode<T>>& node) {
	if(node->depth() > 1 || !node->in_tree()) return;

  vector<QNode<T>>& qnodes = node->Children();

  int n_actions = model_.NumActions();

  for(int a = 0; a < n_actions; a++) {
		QNode<T>& qnode = qnodes[a];
		vector<uint64_t> labels = qnode.BranchLabels();

		for(int j=0; j<node->depth(); j++)
			cout << "  ";
		cout << "a=" << a << " (" << (qnode.first_step_reward() + qnode.LowerBound()) << " " << (qnode.first_step_reward() + qnode.UpperBound()) << ")" << endl;

		for(int i=0; i<labels.size(); i++) {
			for(int j=0; j<node->depth(); j++)
				cout << "  ";
			cout << " o=" << labels[i] << endl;
			PrintTree(qnode.Belief(labels[i]));
		}
  }
}

template<typename T>
void Solver<T>::RetrieveHistoryActionMapping(SuffixPolicyLowerBound<T>* suffix_policy) {
	History history = history_;
	RetrieveHistoryActionMapping(root_, history, suffix_policy);
}

template<typename T>
void Solver<T>::RetrieveHistoryActionMapping(unique_ptr<VNode<T>>& node,
		History& history,
		SuffixPolicyLowerBound<T>* suffix_policy) {
	if(!node->in_tree()) return;

	// The history should be long enough so that current belief does not depend
	// much on the initial belief.
	// TODO: may need to use a longer history
	if(history.Size() > 0 && node->best_ub_action() != -1)
		suffix_policy->Put(history, node->OptimalAction()); 

  vector<QNode<T>>& qnodes = node->Children();

  int n_actions = model_.NumActions();

  for(int action = 0; action < n_actions; action++) {
		QNode<T>& qnode = qnodes[action];
		vector<uint64_t> labels = qnode.BranchLabels();

		for(int i=0; i<labels.size(); i++) {
			history.Add(action, labels[i]);
			RetrieveHistoryActionMapping(qnode.Belief(labels[i]), history,
					suffix_policy);
			history.RemoveLast();
		}
  }
}

template<typename T>
bool Solver<T>::Finished() const {
  for (auto p: root_->particles())
    if (!model_.IsTerminal(p->state))
      return false;
  return true;
}

template<typename T>
void Solver<T>::UpdateBelief(int act,T new_state, T new_state_old) {
  belief_ = bu_.Update(belief_, Globals::config.n_belief_particles, act, new_state, new_state_old);
  cerr << "DEBUG: Current number of particles in belief: " << belief_.size() << endl;
  vector<Particle<T>*> particles = bu_.Sample(belief_, Globals::config.n_particles);
  cerr << "DEBUG: Sampled " << particles.size() << " particles from initial belief" << endl; 

  int obs_dummy=0;
  history_.Add(act, obs_dummy);

	//this->model_.ModifyObsStates(particles,new_state,bu_.belief_update_seed_);

  // Clear memory and renew root
  //cout<<"act obs "<<act<<" "<<obs<<endl;

  pair<double, int> lb = lb_.LowerBound(particles, model_, 0, history_);

  default_action_ = lb.second;
  double ub       = ub_.UpperBound(particles, model_, 0, history_);

  Globals::ValidateBounds(lb.first, ub);

  root_ = unique_ptr<VNode<T>>(
      new VNode<T>(std::move(particles), lb.first, ub, 0, 1, false));
}

#endif
