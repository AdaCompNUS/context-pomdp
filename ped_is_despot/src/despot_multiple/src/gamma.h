#ifndef GAMMA_H
#define GAMMA_H

#include <cmath>
#include <cstdlib>
#include "util.h"

using namespace std;

class Gamma {
private:
	double k_;
	double theta_;
public:
	Gamma(double k, double theta) {
		k_ = k;
		theta_ = theta;
	}

	double next() {
		return next(k_, theta_);
	}

	static double next(double k, double theta) {
		if (k < 1) {
			double c = 1 / k;
			double d = (1 - k) * pow(k, k / (1 - k));
			double z, e, x;
			while(true) { //Weibull's algorithm
				double u = Util::RandomDouble();
				double v = Util::RandomDouble();
				z = -log(u);
				e = -log(v);
				x = pow(z, c);
				if (z + e >= d + x)
					return x * theta;
			}
		}
		else { // Cheng's algorithm
			double b = k - log(4);
			double c = k + sqrt(2 * k - 1);
			double d = sqrt(2 * k - 1);
			double e = 1 + log(4.5);
			double x, y, z, r;
			while(true) {
				double u = Util::RandomDouble();
				double v = Util::RandomDouble();
				y = log(v / (1 - v)) / d;
				x = k * exp(y);
				z = u * v * v;
				r = b + c * y - x;
				if (r >= 4.5 * z -e || r >= log(z))
					return x * theta;
			}
		}
	}
};

#endif
