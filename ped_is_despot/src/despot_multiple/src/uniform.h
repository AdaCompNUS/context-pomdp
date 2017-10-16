#ifndef UNIFORM_H
#define UNIFORM_H

class Uniform {
public:
	unsigned seed_;
	Uniform() : seed_(0) {}
	Uniform(unsigned seed) : seed_(seed) {}

	Uniform(double rNum) {
		seed_ = rNum * RAND_MAX;
	}

	double next() {
		return 1.0 * rand_r(&seed_) / RAND_MAX;
	}

	int nextInt(int n) {
		return int(n * next());
	}
};

#endif

