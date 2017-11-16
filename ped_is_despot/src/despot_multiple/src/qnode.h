#ifndef OBSSET_H
#define OBSSET_H

#include "vnode.h"
#include "globals.h"
#include "history.h"
#include "lower_bound/lower_bound.h"
#include "model.h"
#include "upper_bound/upper_bound.h"

/* This class represents an Q-node/AND-node (child of a belief node) of the search tree.
 */
template<typename T>
class QNode {
 public:
  // Params:
  // @obs_to_particles: mapping from observation to the particles (after the
  // transition) that generated the observation. The particles for each
  // observation become the representative set for the corresponding belief
  // node at the next level.
  // @depth: depth of the belief node *above* this node.
  // @action: action that led to this obsset.
  // @first_step_reward: The average first step reward of particles when they
  // transitioned from the belief node above this node.
  // @history: history upto the belief node *above* this node.
  // @debug: Flag controlling debugging output.
  QNode(MAP<uint64_t, vector<Particle<T>*>>& obs_to_particles,
         int depth,
         int action,
         double first_step_reward,
         History& history,
         const Model<T>& model,
         const ILowerBound<T>& lb,
         const IUpperBound<T>& ub,
         bool debug=false);

  double UpperBound() const;

  double LowerBound() const;

  double Prune(int& total_pruned) const;

  // Returns the observation with the highest weighted excess uncertainty
  // ("WEU") along with this highest value.
  // @root: Root of the search tree, passed to facilitate computation of the 
  //        excess uncertainty
  pair<uint64_t, double> BestWEUO(const unique_ptr<VNode<T>>& root) const;

	vector<uint64_t> BranchLabels() const {
		vector<uint64_t> obss;
		for(auto& it: obs_to_node_)
			obss.push_back(it.first);
		return obss;
	}

  // Returns the belief node corresponding to a given observation
  unique_ptr<VNode<T>>& Belief(uint64_t obs) {
    return obs_to_node_[obs];
  }

  double first_step_reward() const { return first_step_reward_; }

 private:
  int depth_; // Depth of the belief node *above* this node
  double weight_sum_; // The combined weight of particles at this node
  double first_step_reward_;
  MAP<uint64_t, unique_ptr<VNode<T>>> obs_to_node_; // Mapping from obs to
                                                         // belief node
};

template<typename T>
QNode<T>::QNode(
    MAP<uint64_t, vector<Particle<T>*>>& obs_to_particles, int depth, 
    int action, double first_step_reward, History& history, 
    const Model<T>& model, const ILowerBound<T>& lb, const IUpperBound<T>& ub,
    bool debug)
    : depth_(depth),
      first_step_reward_(first_step_reward) {
  weight_sum_ = 0;
  for (auto& r: obs_to_particles) {
    double obs_ws = 0;
    for (auto p: r.second)
      obs_ws += p->wt;
    weight_sum_ += obs_ws;

    history.Add(action, r.first);
    double l = lb.LowerBound(r.second, model, depth + 1, history).first;
    double u = ub.UpperBound(r.second, model, depth + 1, history);
    history.RemoveLast();
    Globals::ValidateBounds(l, u);
    obs_to_node_[r.first] = unique_ptr<VNode<T>>(
      new VNode<T>(std::move(r.second), l, u, depth + 1, obs_ws, false));
  }
}

template<typename T>
double QNode<T>::UpperBound() const {
  double ans = 0;
  for (auto& it: obs_to_node_)
    ans += it.second->ubound() * it.second->weight() / weight_sum_;
  return ans;
}

template<typename T>
double QNode<T>::LowerBound() const {
  double ans = 0;
  for (auto& it: obs_to_node_)
    ans += it.second->lbound() * it.second->weight() / weight_sum_;
  return ans;
}

template<typename T>
double QNode<T>::Prune(int& total_pruned) const {
  double cost = 0;
  for (auto& it: obs_to_node_)
    cost += it.second->Prune(total_pruned);
  return cost;
}

template<typename T>
pair<uint64_t, double> QNode<T>::BestWEUO(
    const unique_ptr<VNode<T>>& root) const {
  double weighted_eu_star = -Globals::INF;
  uint64_t o_star = 0;
  for (auto& it: obs_to_node_) {
    double weighted_eu = it.second->weight() / weight_sum_ *
                         Globals::ExcessUncertainty(
                           it.second->lbound(), it.second->ubound(),
                           root->lbound(), root->ubound(), depth_ + 1);
    if (weighted_eu > weighted_eu_star) {
      weighted_eu_star = weighted_eu;
      o_star = it.first;
    }
  }
  return {o_star, weighted_eu_star};
}

#endif
