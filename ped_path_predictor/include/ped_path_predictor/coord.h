#ifndef COORD_PED_PATH_PREDICTOR_H
#define COORD_PED_PATH_PREDICTOR_H

#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <math.h>

using namespace std;

struct COORD
{
  double x, y;
  
  COORD() {}

  COORD(double _x, double _y) : x(_x), y(_y) {}

  bool operator==(COORD rhs) const {
    return x == rhs.x && y == rhs.y;
  }

  bool operator!=(COORD rhs) const {
    return x != rhs.x || y != rhs.y;
  }

  COORD operator+(COORD rhs) const {
    return COORD(x + rhs.x, y + rhs.y);
  }

  COORD operator-(COORD rhs) const {
    return COORD(x - rhs.x, y - rhs.y);
  }

  COORD operator*(double mul) const
  {
    return COORD(x * mul, y * mul);
  }

  COORD operator/(double mul) const
  {
    assert(mul != 0);
    return COORD(x / mul, y / mul);
  }

  void operator+=(COORD offset) {
    x += offset.x;
    y += offset.y;
  }

  void operator-=(COORD offset) {
    x -= offset.x;
    y -= offset.y;
  }

  void operator*=(double mul)
  {
    x *= mul;
    y *= mul;
  }

  void operator/=(double mul)
  {
    assert(mul != 0);
    x /= mul;
    y /= mul;
  }

  friend ostream& operator<<(ostream& os, const COORD& val)  
  {  
    os<<"("<<val.x<<", "<<val.y<<")"; 
    return os;  
  }

  void Normalize(){
    double norm = sqrt(x * x + y * y);
    if(norm <= 0) return;
    else{
      x /= norm;
      y /= norm;
    }
  }

  
  static double EuclideanDistance(COORD lhs, COORD rhs);
  static double ManhattanDistance(COORD lhs, COORD rhs);
  static void Normalize(COORD &vec);
};

inline double COORD::EuclideanDistance(COORD lhs, COORD rhs) {
  return sqrt((lhs.x - rhs.x) * (lhs.x - rhs.x) + 
              (lhs.y - rhs.y) * (lhs.y - rhs.y));
}

inline double COORD::ManhattanDistance(COORD lhs, COORD rhs) {
  return fabs(lhs.x - rhs.x) + fabs(lhs.y - rhs.y);
}

inline void Normalize(COORD &vec) {
  double norm = sqrt(vec.x * vec.x + vec.y * vec.y);

  if(norm <= 0) return;
  else{
     vec.x /= norm;
     vec.y /= norm;
  }
}

inline std::ostream& operator<<(std::ostream& ostr, COORD& COORD) {
  ostr << "(" << COORD.x << ", " << COORD.y << ")";
  return ostr;
}

#endif
