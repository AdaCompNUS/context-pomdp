#ifndef COORD_H_DESPOT
#define COORD_H_DESPOT

#include <stdlib.h>
#include <assert.h>
#include <ostream>
#include <math.h>

struct COORD
{
  double x, y;
  
  COORD() {}

  COORD(double _x, double _y) : x(_x), y(_y) {}

  bool Valid() const {
    return x >= 0 && y >= 0;
  }

  bool operator==(COORD rhs) const {
    return x == rhs.x && y == rhs.y;
  }

  bool operator<(const COORD& other) const {
    return x < other.x || (x == other.x && y < other.y);
  }
  
  bool operator!=(COORD rhs) const {
    return x != rhs.x || y != rhs.y;
  }

  void operator+=(COORD offset) {
    x += offset.x;
    y += offset.y;
  }

  COORD operator+(COORD rhs) const {
    return COORD(x + rhs.x, y + rhs.y);
  }

  COORD operator-(const COORD& rhs) const {
      return COORD(x - rhs.x, y - rhs.y);
   }

  COORD operator*(float mul) const
  {
    return COORD(x * mul, y * mul);
  }

  
  static double EuclideanDistance(COORD lhs, COORD rhs);
  static double ManhattanDistance(COORD lhs, COORD rhs);
  
  static double SlopAngle(COORD start, COORD end);

  static bool get_dir(double a, double b);

  double Length(){
	  return sqrt(x*x + y*y);
  }

/*
  static int DirectionalDistance(COORD lhs, COORD rhs, int direction);
  enum {
    E_NORTH,
    E_EAST,
    E_SOUTH,
    E_WEST,
    E_NORTHEAST,
    E_SOUTHEAST,
    E_SOUTHWEST,
    E_NORTHWEST
  };
  static const COORD Null;
  static const COORD North, East, South, West;
  static const COORD NorthEast, SouthEast, SouthWest, NorthWest;
  static const COORD Compass[8];
  static const char* CompassString[8];
  static int Clockwise(int dir) { return (dir + 1) % 4; }
  static int Opposite(int dir) { return (dir + 2) % 4; }
  static int Anticlockwise(int dir) { return (dir + 3) % 4; }

  */
 // static void UnitTest();
};

inline double COORD::EuclideanDistance(COORD lhs, COORD rhs) {
  return sqrt((lhs.x - rhs.x) * (lhs.x - rhs.x) + 
              (lhs.y - rhs.y) * (lhs.y - rhs.y));
}

inline double COORD::ManhattanDistance(COORD lhs, COORD rhs) {
  return fabs(lhs.x - rhs.x) + fabs(lhs.y - rhs.y);
}

inline double COORD::SlopAngle(COORD start, COORD end){

	double angle = atan2(end.y - start.y, end.x - start.x);

	if (angle < 0)
		angle += 2 * M_PI;

	return angle;
}

inline bool COORD::get_dir(double a, double b){
	// a and b are two angles
	double diff = b - a;
	if (diff >= 0 and diff < M_PI)
		return true; // ccw
	else if (diff >= M_PI)
		return false; // cw
	else if (diff < 0 and diff >= -M_PI)
		return false; // cw
	else if (diff < -M_PI)
		return true; // ccw
	else
		return true;
}


/*
inline int COORD::DirectionalDistance(COORD lhs, COORD rhs, int direction) {
  switch (direction) {
    case E_NORTH: return rhs.y - lhs.y;
    case E_EAST: return rhs.x - lhs.x;
    case E_SOUTH: return lhs.y - rhs.y;
    case E_WEST: return lhs.x - rhs.x;
    default: assert(false);
  }
}
*/

inline std::ostream& operator<<(std::ostream& ostr, COORD& COORD) {
  ostr << "(" << COORD.x << ", " << COORD.y << ")";
  return ostr;
}

#endif // COORD_H
