#ifndef BERNOULLI_H
#define BERNOULLI_H

class Uniform {
public:
	unsigned seed;
	Uniform(double rNum) {
		seed = rNum * RAND_MAX;
	}

	double next() {
		return 1.0 * rand_r(&seed) / RAND_MAX;
	}
};

#endif

