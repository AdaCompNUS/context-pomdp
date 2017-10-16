#ifndef PARTICLE_H
#define PARTICLE_H

#include "memorypool.h"

template<typename T>
class Particle : public MemoryObject {
 public:
  Particle() : id(0), wt(0) {}

  Particle(T s, int id_, double weight) : state(s), id(id_), wt(weight) {}

  // Redefined to avoid copying the allocated-bit of the superclass
  Particle& operator=(const Particle& particle) {
    state = particle.state;
    id = particle.id;
    wt = particle.wt;
    return *this;
  }

  T state;
  int id;
  double wt;
};

#endif
