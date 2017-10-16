#ifndef RANDOM_STREAMS_H
#define RANDOM_STREAMS_H

#include <vector>
#include <stdlib.h>

using namespace std;

/* This class encapsulates the streams of random numbers required for state
 * transitions during simulations.
 */
class RandomStreams {
 public:
	RandomStreams(int num_streams, int length, unsigned seed);

	int NumStreams() const { return streams_.size(); }

	int Length() const { return streams_.size() > 0 ? streams_[0].size() : 0; }

	double Entry(int stream, int pos) const {
    return streams_[stream][pos];
  }

private:
  vector<vector<double>> streams_; // Each particle is associated with a single
                                   // stream of numbers.
};

#endif
