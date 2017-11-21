/* This code is for sampling from the dirchlet distribution. We have to provide the alpha vectors as input and then call the next() function though an object to get the sampled value.
 */

#ifndef DIRICHLET_H
#define DIRICHLET_H

#include "gamma.h"
#include <iostream>
#include <vector>

using namespace std;

class Dirichlet {
private:
  vector<double> alpha_; 
public:
  Dirichlet (vector<double> alpha) {
    alpha_ = alpha;
  }

  vector<double> alpha() {
    return alpha_;
  }

  vector<double> next(){
    return next(alpha_);
  }

  static vector<double> next(vector<double> alpha) {
    int dim = alpha.size(); 
    vector<double> x (dim, 0); 
    double sum = 0;
    for (int i = 0; i < dim; i++) {
      x[i] = Gamma::next(alpha[i], 1);
      sum = sum + x[i];
    }
    for (int i = 0; i < dim; i++) x[i] = x[i] / sum;
    return x;
  }
};

#endif

