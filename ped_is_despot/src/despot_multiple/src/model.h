#ifndef MODEL_H
#define MODEL_H

#include "globals.h"

class History;
class RandomStreams;
template<typename T> class Particle;
template<typename T> class ILowerBound;
template<typename T> class IUpperBound;
template<typename T> class BeliefUpdate;

/* This is the template for a problem specification. To implement a new 
 * problem define a class X that represents a state in the problem and
 * specialize this template with parameter X. For example, the Tag problem
 * defines a type TagState and implements Model<TagState>.
 *
 * Some methods in the template are compulsory, whereas others depend on
 * which components are used, like the lower(upper) bound and belief update. 
 * These methods are marked appropriately.
 *
 * Problem-specific runtime parameters can be specified in a file, the path 
 * to which is passed as a command line argument to the program. There
 * are no restrictions on the format of the file, and it is the Model's
 * responsibility to parse it for any information it needs. See the Tag
 * ctor for an example of a simple parser that reads a file with lines 
 * of the form "key = value".
 *
 * Note: The modules UpperBoundStochastic and UpperBoundNonStochastic
 * require the state class (X) to have an implicit conversion to int
 * because they use state-indexed tables. For problems with |S| too
 * large to fit in memory, your model must subclass the IUpperBound
 * interface directly and provide an implementation.
 */
template<typename T>
class Model : public ILowerBound<T>, public IUpperBound<T> {
 public:
  // The ctor is not part of the interface, so it can have any signature.
  // Typically it will accept the path to a file from which it can read 
  // runtime parameters. The path can be specified as a cmdline argument
  // to the program and passed to this ctor.
  Model(string params_file); 

  void Statistics(const vector<Particle<T>*> particles) const;

  // Modifies a state by taking an action on it and uses a random 
  // number to determine the resulting state. Also computes the reward and
  // observation for the transition.
  void Step(T& state, double randomNum, int action, double& reward,
            uint64_t& obs) const;

  // Probability of receiving observation @z given an action and a *resulting*
  // state. Used in belief updates.
  double ObsProb(uint64_t z, const T& state, int action) const;

  // True starting state of the world.
  T GetStartState() const;

  // Starting belief of the agent, a mapping from state to probability.
  vector<pair<T, double>> InitialBelief() const;

  // |A|
  int NumActions() const;

  // Whether a state is a terminal state
  bool IsTerminal(const T& s) const;

  // A unique observation that must be emitted in and only in a terminal state.
  uint64_t TerminalObs() const;

  // Textual display
  void PrintState(const T& state, ostream& out = cout) const;
  void PrintObs(uint64_t obs, ostream& out = cout) const;

  // Methods to create and destroy particles. Copy this as is, or reimplement
  // it to manage memory manually.
  Particle<T>* Copy(const Particle<T>* particle) const {
    return new Particle<T>(*particle);
  }
  Particle<T>* Allocate() const {
    return new Particle<T>();
  }
  void Free(Particle<T>* particle) const {
    delete particle;
  }

  // The following methods don't all need to be implemented. Each method is
  // marked with the name of the component(s) that require it. Default
  // implementations are given for some of them, which can be copied as is
  // in the implementation of the new problem.

  // (=== PolicyLowerBound ===)
  // The lower bound at the fringe nodes (max depth) of the search tree.
  // Take care to ensure that this is a valid lower bound. For example in Tag,
  // there are negative rewards, so simply returning 0 when there are
  // non-terminal particles is incorrect.
  double FringeLowerBound(const vector<Particle<T>*>& particles) const;

  // (=== ModePolicyLowerBound ===)
  // Best action for a given state.
  int LowerBoundAction(const T& state) const;

  // (=== RandomPolicyLowerBound ===)
  // Given a state and the history so far, returns a vector of
  // preferred actions for the state, using only the history and the 
  // observable part of the state. 
  //
  // Implementation note: A pointer is returned to avoid unnecessary copying
  // in case we already have the answer at hand (e.g. precomputation). 
  // Further, it's a shared_ptr so that we control whether the memory for
  // the vector is freed after the client is done with it. For example, if
  // we return a precomputed vector each time, we can maintain a shared_ptr
  // to it to prevent it from freeing up when the copy of the pointer returned
  // to the client goes out of scope. On the other hand, if we construct a
  // vector on the fly every time this method is called, we can return a unique 
  // shared_ptr, ensuring that the memory is released when it goes out of
  // scope.
  shared_ptr<vector<int>> GeneratePreferred(const T& state, 
      const History& history) const;

  // (=== RandomPolicyLowerBound ===)
  // Returns a vector with all the legal actions for a given state, using only
  // the history and the observable part of a state.
  shared_ptr<vector<int>> GenerateLegal(const T& state, const History& history)
      const {
    static shared_ptr<vector<int>> actions = nullptr;
    if (!actions) {
      int num_actions = NumActions();
      actions = shared_ptr<vector<int>>(new vector<int>(num_actions));
      for (int a = 0; a < num_actions; a++)
        (*actions)[a] = a;
    }
    return actions;
  }

  // (=== UpperBoundStochastic ===)
  // This version of step() does not require the observation, as opposed to the
  // one above. In some cases this can give a significant speedup of the
  // upper bound precomputation (e.g. in LaserTag). If no gains are to be made,
  // simply call the above version with a dummy observation that is discarded.
  void Step(T& state, double random_num, int action, double& reward) const {
    uint64_t obs;
    Step(state, random_num, action, reward, obs);
  }
 
  //void RobStep(T& state, int action, Uniform unif) const;

  // (=== UpperBoundStochastic | UpperBoundNonStochastic ===)
  // The upper bound at the fringe nodes (at max depth) of the search tree.
  double FringeUpperBound(const T& state) const;

  // (=== ParticleFilterUpdate ===)
  // Returns a random state used in bootstrapping the particle filter when it
  // becomes empty. This method is called repeatedly, and the return value is
  // checked for consistency with the observation @obs, until a sufficient 
  // number of consistent particles is obtained. Therefore, although not 
  // necessary, returning states that are consistent with @obs will ensure the
  // filter refills quickly, especially in problems with large state spaces.
  //T RandomState(unsigned& seed, uint64_t obs) const; 
    T RandomState(unsigned& seed, T obs_state) const; 
  // (=== ExactBeliefUpdate ===)
  // Returns the transition matrix for the model, a mapping from
  // [s][a] to {s', P(s' | s, a)}
  vector<vector<UMAP<T, double>>> TransitionMatrix() const;
};

#endif
