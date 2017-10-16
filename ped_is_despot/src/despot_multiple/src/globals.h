#ifndef SHARED_H
#define SHARED_H

#include <map>
#include <memory>
#include <set>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <list>
#include <algorithm>
#include <ctime>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <inttypes.h>

using namespace std;

#define MAP map
#define UMAP unordered_map
typedef std::pair<int, int> pii;

namespace Globals {

extern const double INF;
extern const double TINY;

struct Config {
  // Maximum depth of search tree; also the maximum number of steps for which
  // the simulation runs.
  int search_depth;
  double discount;
  // The seed of life.
  unsigned int root_seed;
  // The amount of CPU time available to construct the search tree, not
  // including the the time taken to prune the tree and update the belief.
  double time_per_move;
  // Number of particles used in belief
  int n_belief_particles;
  // Number of starting states.
  int n_particles;
  // Whether the search tree should be pruned after construction.
  double pruning_constant;
  // Parameter such that eta * width(root) is the target uncertainty at the
  // root. Determines the termination criterion for a trial.
  double eta;
  // Number of steps to run the simulation for.
  int sim_len;
  // Whether the initial lower and upper bounds are approximate or true. If
  // they are approximate, the solver won't choke when lower > upper.
  bool approximate_bounds;
	int number;

  Config() : 
    search_depth(90),
    discount(0.95),
    root_seed(42),
    time_per_move(1),
	n_belief_particles(1000),
    n_particles(500),
    pruning_constant(0),
    eta(0.95),
    sim_len(90),
    approximate_bounds(false),
		number(1)
  {}
};

extern Config config;

inline vector<string> Tokenize(string line, char delim) {
  // Tokenizes a string on a delim and strips whitespace from the tokens
  vector<string> tokens;
  stringstream ss(line);
  string item;
  while (getline(ss, item, delim)) {
    while (isspace(item.front()))
      item.erase(item.begin());
    while (isspace(item.back()))
      item.erase(item.end()-1);
    tokens.push_back(item);
  }
  return tokens;
}

inline bool Fequals(double a, double b) { 
  return fabs(a - b) < TINY;
}

inline double ExcessUncertainty(double l, double u, double root_l, 
    double root_u, int depth) {
  return (u-l) // width of current node
         - (config.eta * (root_u-root_l)) // epsilon
         * pow(config.discount, -depth);
}

inline void ValidateBounds(double& lb, double& ub) {
  if (ub >= lb)
    return;
  if (ub > lb - TINY || config.approximate_bounds)
    ub = lb;
  else {
		cerr << "UB: " << ub << "; LB: " << lb << endl;
    assert(false);
	}
}

} // namespace

#endif
