#ifndef MATHUTILS_H
#define MATHUTILS_H
#include <cmath>
#include "param.h"

class MyVector
{

public:
	MyVector();
	MyVector(double _dw,double _dh);
	MyVector(double angle,double length,int dummy);
	double GetAngle();   //[-pi,pi]
	double GetLength();
	void GetPolar(double &angle,double &length);
	void AdjustLength(double length);
	void SetAngle(double angle);
	MyVector  operator + (MyVector  vec);
	double dw,dh;
	double angle,length;
};

double DotProduct(double x1,double y1,double x2,double y2);
double CrossProduct(MyVector vec1, MyVector vec2);
double Norm(double x,double y);
void Uniform(double x,double y,double &ux,double &uy);
void AddVector(double in_angle,double in_length,double &out_angle,double &out_length);

#endif
