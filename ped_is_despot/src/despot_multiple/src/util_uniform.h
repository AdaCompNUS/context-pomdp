#ifndef UTILUNIFORM_H
#define UTILUNIFORM_H

class UtilUniform {
public:
	unsigned seed_;
	UtilUniform() : seed_(0) {}
	UtilUniform(unsigned seed) : seed_(seed) {}

	UtilUniform(double rNum) {
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

