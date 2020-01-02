#ifndef DEBUG_UTIL_H
#define DEBUG_UTIL_H

#pragma once

#include <cstdio>
#include <string>
#include <cassert>


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

#endif
