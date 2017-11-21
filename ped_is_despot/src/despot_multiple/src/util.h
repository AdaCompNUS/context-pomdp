#ifndef UTIL_H
#define UTIL_H

namespace Util {
	inline void RandomSeed(int seed) {
		srand(seed);
	}

	inline double RandomDouble(double min, double max) {
    return (double) rand() / RAND_MAX * (max - min) + min;
	}

	inline double RandomDouble() {
    return (double) rand() / RAND_MAX;
	}

	inline int RandomCategory(vector<double> probs) {
		double r = RandomDouble();
		int c = 0;
		double sum = probs[0];
		while(sum < r) {
			c ++;
			sum += probs[c];
		}
		return c;
	}

	inline int GetCategory(vector<double> probs, double r) {
		int c = 0;
		double sum = probs[0];
		while(sum < r) {
			c ++;
			sum += probs[c];
		}
		return c;
	}
};

// Functions for hashing data structs
namespace std {
	template<class T>
	inline void hash_combine(size_t& seed, const T& v) {
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	template<typename S, typename T>
	struct hash<pair<S, T>> {
		inline size_t operator()(const pair<S, T>& v) const {
			size_t seed = 0;
			::hash_combine(seed, v.first);
			::hash_combine(seed, v.second);
			return seed;
		}
	};

	template<typename T>
	struct hash<vector<T>> {
		inline size_t operator()(const vector<T>& v) const {
			size_t seed = 0;
			for (const T& ele : v) {
				::hash_combine(seed, ele);
			}
			return seed;
		}
	};
}

#endif
