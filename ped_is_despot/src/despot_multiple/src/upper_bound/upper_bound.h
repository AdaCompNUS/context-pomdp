#ifndef UPPER_BOUND_H
#define UPPER_BOUND_H

#include "globals.h"
#include "history.h"
#include "model.h"
#include "particle.h"
#include "random_streams.h"

// This is the interface implemented by all upper bound strategies.
template<typename T>
class IUpperBound {
 public:
  IUpperBound(const RandomStreams& streams) : streams_(streams) {}

  virtual ~IUpperBound() {};

  // Input: a set of particles, the model, the position of the particles in the
  // random number stream (= depth in the tree), and the history seen so far.
  // Output: upper bound
  virtual double UpperBound(const vector<Particle<T>*>& particles,
                            const Model<T>& model,
                            int stream_position,
                            History& history) const = 0;
 protected:
  const RandomStreams& streams_;
};

#endif
