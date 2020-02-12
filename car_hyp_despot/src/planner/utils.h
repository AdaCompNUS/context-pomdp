#ifndef DEBUG_UTIL_H
#define DEBUG_UTIL_H

#pragma once

#include <cstdio>
#include <string>
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <math.h>
#include <chrono>
#include <locale>

using namespace std;
using namespace chrono;

template< typename... Args >
std::string string_sprintf( const char* format, Args... args ) {
  int length = std::snprintf( nullptr, 0, format, args... );
  assert( length >= 0 );

  char* buf = new char[length + 1];
  std::snprintf( buf, length + 1, format, args... );
  std::string str( buf );
  delete[] buf;

  return str;
}

#define ERR(msg) { std::string str = msg; \
				fprintf(stderr,"ERROR: %s, in %s, at file %s_line_%d \n", str.c_str(), __FUNCTION__, __FILE__, __LINE__); \
				raise(SIGABRT); }

#define DEBUG(msg) { std::string str = msg; \
				fprintf(stderr, "MSG: %s, in %s, at file %s_line_%d \n", str.c_str(), __FUNCTION__, __FILE__, __LINE__); }


namespace std {
	template<class T>
	inline void hypdespot_hash_combine(size_t& seed, const T& v) {
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	template<typename S, typename T>
	struct hash<pair<S, T>> {
		inline size_t operator()(const pair<S, T>& v) const {
			size_t seed = 0;
			::hypdespot_hash_combine(seed, v.first);
			::hypdespot_hash_combine(seed, v.second);
			return seed;
		}
	};

	template<typename T>
	struct hash<vector<T>> {
		inline size_t operator()(const vector<T>& v) const {
			size_t seed = 0;
			for (const T& ele : v) {
				::hypdespot_hash_combine(seed, ele);
			}
			return seed;
		}
	};
}

// NOTE: disabled C++11 feature
template<typename T>
void write(ostringstream& os, T t) {
	os << t;
}

template<typename T, typename ... Args>
void write(ostringstream& os, T t, Args ... args) {
	os << t;
	write(os, args...);
}

template<typename T, typename ... Args>
string concat(T t, Args ... args) {
	ostringstream os;
	write(os, t, args...);
	return os.str();
}

template<typename T>
string concat(vector<T> v) {
	ostringstream os;
	for (int i = 0; i < v.size(); i++) {
		os << v[i];
		os << " ";
	}
	return os.str();
}

#endif
