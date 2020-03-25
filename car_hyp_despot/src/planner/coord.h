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

  COORD(double angle,double length,int dummy)
  {
  	x=length*cos(angle);
  	y=length*sin(angle);
  }

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

  double dot( const COORD& other){
    return x * other.x + y * other.y;
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
  static double DirectedDistance(COORD lhs, COORD rhs, double dir);
  
  static double SlopAngle(COORD start, COORD end);
  static double Angle(COORD A, COORD B, COORD C, double noise_level);
  static bool GetDir(double a, double b);

  double GetAngle() const  //[0,2*pi)
  {
    double angle = atan2(y,x);
    if (angle < 0)
      angle += 2*M_PI;
    if (angle >= 2*M_PI)
      angle -= 2*M_PI;
    return angle;
  }
  double Length() const{
	  return sqrt(x*x + y*y);
  }

  double LengthSq() const{
	  return x*x + y*y;
  }

  void AdjustLength(double length){
    if(Length()<0.001) return;   //vector length close to 0
    double rate=length/Length();
    x*=rate;
    y*=rate;
  }

  COORD Scale(double length) const {
      if(Length()<0.001)
    	  return COORD(0.0, 0.0);
      double rate=length/Length();
      return COORD(x * rate, y * rate);
  }
};

inline double COORD::EuclideanDistance(COORD lhs, COORD rhs) {
  return sqrt((lhs.x - rhs.x) * (lhs.x - rhs.x) + 
              (lhs.y - rhs.y) * (lhs.y - rhs.y));
}

inline double COORD::ManhattanDistance(COORD lhs, COORD rhs) {
  return fabs(lhs.x - rhs.x) + fabs(lhs.y - rhs.y);
}

inline double COORD::DirectedDistance(COORD lhs, COORD rhs, double dir){
  return fabs((lhs.x - rhs.x)*cos(dir) + (lhs.y - rhs.y)*sin(dir));
}

inline double COORD::SlopAngle(COORD start, COORD end){

	double angle = atan2(end.y - start.y, end.x - start.x);

	if (angle < 0)
		angle += 2 * M_PI;

	return angle;
}

inline double COORD::Angle(COORD A, COORD B, COORD C, double noise_level) {
	// angle between AB and AC in rad.
	double angle = 0;
	double norm_AB = (B-A).Length();
	double norm_AC = (C-A).Length();
	if (norm_AB > noise_level && norm_AC > noise_level){
		double cosa = (B-A).dot(C-A) / (norm_AB * norm_AC);
		cosa = std::max(-1.0, std::min(1.0, cosa));
		angle = acos(cosa);
	}
	return angle;
}

inline bool COORD::GetDir(double a, double b){
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

inline std::ostream& operator<<(std::ostream& ostr, COORD& COORD) {
  ostr << "(" << COORD.x << ", " << COORD.y << ")";
  return ostr;
}

#endif // COORD_H
