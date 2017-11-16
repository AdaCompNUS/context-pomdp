#ifndef VNODE_H
#define VNODE_H

#include "globals.h"

template<typename T>
class QNode;

/* This class encapsulates a belief/value/AND node.
 * It stores the set of particles associated with the node
 * and some bookkeeping information.
 */
template<typename T>
class VNode {
 public:
  // Note: The ctor move-constructs the particles instead of copying them
  // for efficiency.
  VNode(vector<Particle<T>*>&& particles, double l, double u, int depth,
             double weight, bool it=false)
      : particles_(particles),
        lbound_(l),
        ubound_(u),
        depth_(depth),
        default_value_(l),
        pruned_action_(-1),
        weight_(weight),
        best_ub_action_(-1),
        in_tree_(it),
        n_tree_nodes_(it) {
    node_count_++;
  }

  ~VNode() { 
    for (auto it: particles_)
      model_->Free(it);
    node_count_--; 
  }

  int pruned_action() const { return pruned_action_; }

  double weight() const { return weight_; }

  double lbound() const { return lbound_; }
  void set_lbound(double val) { lbound_ = val; }

  double ubound() const { return ubound_; }
  void set_ubound(double val) { ubound_ = val; }

  int depth() const { return depth_; }

  int best_ub_action() const { return best_ub_action_; }
  void set_best_ub_action(int val) { best_ub_action_ = val; }

  bool in_tree() const { return in_tree_; }
  void set_in_tree(bool val) { in_tree_ = val; }

  int n_tree_nodes() const { return n_tree_nodes_; }
  void set_n_tree_nodes(int val) { n_tree_nodes_ = val; }

  const vector<Particle<T>*>& particles() { return particles_; }

  static int node_count() { return node_count_; }

  double Prune(int& total_pruned);

  vector<QNode<T>>& Children() { return qnodes_; }

  static void set_model(const Model<T>& model) { model_ = &model; }

  // Returns the optimal action computed during search. Since we need the 
  // optimal action only at the root of the search tree, we avoid storing it 
  // as a property of the node and recompute it in this method when needed.
  // Prerequisite: The node must have children to select the best action from.
  int OptimalAction() const;

 private:
  vector<Particle<T>*> particles_;
  double lbound_;
  double ubound_;
  const int depth_;
  const double default_value_; // Value under default policy (= lbound value
                               // before any backups)
  int pruned_action_;    // Best action at the node after pruning
  const double weight_;  // Sum of weights of particles at this belief
  int best_ub_action_;   // Action that gives the highest upper bound
  bool in_tree_;         // True if the node is visited by Solver::trial().
                         // In order to determine if a node is a fringe node 
                         // of the belief tree, we need to expand it one level.
                         // The nodes added during this expansion of a fringe
                         // node are not considered to be within the tree, so
                         // we use this indicator variable. 
  int n_tree_nodes_;     // Number of nodes with in_tree_ = true in the subtree
                         // rooted at this node
  static int node_count_;      // Total number of nodes created
  vector<QNode<T>> qnodes_; // Vector of children q-nodes
  static const Model<T>* model_;
};

template<typename T> int VNode<T>::node_count_;
template<typename T> const Model<T>* VNode<T>::model_;

template<typename T>
double VNode<T>::Prune(int& total_pruned) {
  // Cost if node were pruned
  double cost = pow(Globals::config.discount, depth_) * weight_ * default_value_ 
                - Globals::config.pruning_constant;
  if (!(in_tree_)) { // Leaf
    assert(n_tree_nodes_ == 0);
    return cost;
  }
  for (int a = 0; a < qnodes_.size(); a++) {
    double first_step_reward = pow(Globals::config.discount, depth_) * weight_ 
                               * qnodes_[a].first_step_reward();
    double best_child_value = qnodes_[a].Prune(total_pruned);

    // "-Globals::config.pruning_cost" to include the cost of the current node
    double new_cost = first_step_reward + best_child_value 
                      - Globals::config.pruning_constant;
    if (new_cost > cost) {
      cost = new_cost;
      pruned_action_ = a;
    }
  }
  if (pruned_action_ == -1)
    total_pruned++;
  return cost;
}

template<typename T>
int VNode<T>::OptimalAction() const {
  int a_star = -1;
  double q_star = -Globals::INF;
  for (int a = 0; a < qnodes_.size(); a++) {
    auto& qnode = qnodes_[a];
    double remaining_reward = qnode.LowerBound();
    if (qnode.first_step_reward() + Globals::config.discount * remaining_reward > 
       q_star + Globals::TINY) {
      q_star = qnode.first_step_reward() + Globals::config.discount *
               remaining_reward;
      a_star = a;
    }
  }
  return a_star;
}

#endif
