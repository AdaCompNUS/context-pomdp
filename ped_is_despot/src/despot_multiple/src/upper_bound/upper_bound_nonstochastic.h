#ifndef UPPER_BOUND_NONSTOCHASTIC_H
#define UPPER_BOUND_NONSTOCHASTIC_H

#include "upper_bound/upper_bound.h"

/* This class computes a generic upper bound for non-stochastic models. The 
 * computation is similar to the stochastic version. The difference
 * is that we can take advantage of the deterministic state transitions to 
 * reduce the complexity:
 *
 * Space complexity: O(|S|)
 * Time complexity for pre-processing: O(search depth * |S| * |A|)
 * Time complexity for query: O(#particles).
 */
template<typename T>
class UpperBoundNonStochastic : public IUpperBound<T> {
 public:
  UpperBoundNonStochastic(const RandomStreams& streams, const Model<T>& model); 

  double UpperBound(const vector<Particle<T>*>& particles,
                    const Model<T>& model,
                    int stream_position,
                    History& history) const;

  const vector<int>& UpperBoundAct() const { return upper_bound_act_; }

 private:
  // We use dynamic programming to compute the best upper bound action/value 
  // for a state, with horizon = maximum depth of the tree. We use this value
  // regardless of the depth at which the upper bound is requested, as it 
  // is will always be a tighter bound. We couldn't do this in the
  // non-stochastic case because the upper bound depended on the sequence 
  // of random numbers.
  vector<int> upper_bound_act_;
  vector<double> upper_bound_memo_;
};

template<typename T>
UpperBoundNonStochastic<T>::UpperBoundNonStochastic(
    const RandomStreams& streams, const Model<T>& model)
    : IUpperBound<T>(streams) {
  int S = model.NumStates(), A = model.NumActions();
  upper_bound_act_.resize(S);
  upper_bound_memo_.resize(S, -Globals::INF);
  vector<double> upper_bound_memo_tmp(S);
  for (int s = 0; s < S; s++)
    upper_bound_memo_tmp[s] = model.FringeUpperBound(s);
  // Maintain pointers into 2 levels of the array and swap them each iteration
  auto* ptr1 = &upper_bound_memo_, *ptr2 = &upper_bound_memo_tmp;
  double r;
  auto particle = model.Allocate();
  for (int i = 0; i < Globals::config.search_depth; i++) { // Length of horizon
    for (int s = 0; s < S; s++) {
      for (int a = 0; a < A; a++) {
        particle->state = s;
        model.Step(particle->state, 0/*dummy*/, a, r);
        double &ref = (*ptr1)[s];
        double maybe_better = r + Globals::config.discount * (*ptr2)[particle->state];
        if (maybe_better > ref) {
          ref = maybe_better;
          if (i == Globals::config.search_depth - 1)
            // Set best actions when last level is being computed
            upper_bound_act_[s] = a;
        }
      }
    }
    fill(ptr2->begin(), ptr2->end(), -Globals::INF);
    swap(ptr1, ptr2);
  }
  model.Free(particle);
  if (Globals::config.search_depth % 2 == 0)
    // The last computation is stored in the wrong level
    upper_bound_memo_.swap(upper_bound_memo_tmp);
}

template<typename T>
double UpperBoundNonStochastic<T>::UpperBound(
    const vector<Particle<T>*>& particles,
    const Model<T>& model,
    int stream_position,
    History& history) const {
  double ws = 0, total_cost = 0;
  for (auto p: particles) {
    ws += p->wt;
    total_cost += p->wt * upper_bound_memo_[p->state];
  }
  return total_cost / ws;
}

#endif
