#ifndef POLICY_LOWER_BOUND_H
#define POLICY_LOWER_BOUND_H

#include "lower_bound/lower_bound.h"

/* This class computes the the lower bound by executing a policy (tree) on
 * a set of particles. After each action, the particles are grouped 
 * by the observation generated, and the simulation continues recursively 
 * at the next level until the particles reach a terminal state or
 * maximum depth. The best action to take at each step is variable, and can
 * be specified via a virtual method by refining this class.
 */
template<typename T>
class PolicyLowerBound : public ILowerBound<T> {
 public:
  PolicyLowerBound(const RandomStreams& streams) : ILowerBound<T>(streams) {}

  virtual ~PolicyLowerBound() {}

  pair<double, int> LowerBound(const vector<Particle<T>*>& particles,
                               const Model<T>& model,
                               int stream_position,
                               History& history) const;

 public:
  // Returns the best action for a set of particles. Implement this in
  // subclasses of this class.
  virtual int Action(const vector<Particle<T>*>& particles,
                     const Model<T>& model,
                     const History& history) const = 0;

 private:
  pair<double, int> LowerBoundImpl(vector<Particle<T>*>& particles,
                                   const Model<T>& model,
                                   int stream_position,
                                   History& history) const;
};

template<typename T>
pair<double, int> PolicyLowerBound<T>::LowerBound(
    const vector<Particle<T>*>& particles,
    const Model<T>& model,
    int stream_position,
    History& history) const {
  // Copy particles so that we can modify them
  auto copy = particles;
  for (auto& it: copy) {
    it = model.Copy(it);
	}
	/*
	cerr << "Simulating with the following particles at position " << stream_position << endl;
	for (auto& it: copy) {
    model.PrintState(it->state);
	}*/

  pair<double, int> lb = LowerBoundImpl(copy, model, stream_position, history);
  for (auto it: copy)
    model.Free(it);
  return lb;
}

template<typename T>
pair<double, int> PolicyLowerBound<T>::LowerBoundImpl(
    vector<Particle<T>*>& particles,
    const Model<T>& model,
    int stream_position,
    History& history) const {

  bool debug = false;
  if (debug) { cerr << "Lower bound depth = " << stream_position << endl; }

  // Terminal states have a unique observation, so if we took that branch it
  // means all the particles must be in a terminal state.
  if (model.IsTerminal(particles[0]->state)) {
		if(debug) {
			cerr << "Terminal: ";
			model.PrintState(particles[0]->state); 
		}
    return {0, -1};
	}

  if (stream_position >= Globals::config.search_depth)
    // The actual action should never be required when 
    // depth >= config.search_depth, so we return a dummy value.
    return { model.FringeLowerBound(particles), -1/*dummy*/ };

  int act = Action(particles, model, history);
  //if (debug) { cerr << "act = " << act << endl; }

  // Compute value based on one step lookahead
  double weight_sum = 0;
  MAP<uint64_t, pair<double, vector<Particle<T>*>>> partitioned_particles;
  double first_step_reward = 0;
//  if(debug)
	 // cerr<<"particle size lower bound "<<particles.size()<<endl;

	 //cout<<"particle address in lower bound "<<&particles<<endl;
  for (auto& p: particles) {
    uint64_t o; double r;
	//cout << "Before " << p->id << " " << stream_position << endl;
	//model.PrintState(p->state);
    model.Step(p->state, this->streams_.Entry(p->id, stream_position), 
               act, r, o);
	//cout << "After" << endl;
	//model.PrintState(p->state);
    auto& ref = partitioned_particles[o];
    ref.first += p->wt;
    ref.second.push_back(p);
    first_step_reward += r * p->wt;
    weight_sum += p->wt;
  }
  first_step_reward /= weight_sum;

	if(debug) { cerr << "reward = " << first_step_reward << endl; }

  double remaining_reward = 0;
  for (auto& it: partitioned_particles) {
    history.Add(act, it.first);
    remaining_reward += Globals::config.discount *
                        LowerBoundImpl(it.second.second, model, 
                                       stream_position + 1, history).first *
                        it.second.first / weight_sum;
    history.RemoveLast();
  }

  return { first_step_reward + remaining_reward, act };
}

#endif
