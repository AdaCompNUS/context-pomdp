#ifndef LOWERBOUND_H
#define LOWERBOUND_H

#include "globals.h"
#include "history.h"
#include "model.h"
#include "particle.h"
#include "random_streams.h"

template<typename T>
class Solver;

// This is the interface implemented by lower bound strategies.
template<typename T>
class ILowerBound {
 public:
  ILowerBound(const RandomStreams& streams) : streams_(streams) {}

  virtual ~ILowerBound() {}
  
  // Input: a set of particles, the model, their position in the random number
  // stream (= depth in the tree), and the history seen so far.
  // Output: The lower bound and the first step action that achieves the bound.
  virtual pair<double, int> LowerBound(const vector<Particle<T>*>& particles,
                                       const Model<T>& model,
                                       int stream_position,
                                       History& history) const = 0;

	virtual void CollectSearchInformation(Solver<T>* solver) {
	}

 protected:
  const RandomStreams& streams_;
};

#endif
