#include "math_utils.h"


MyVector::MyVector() {dw=0;dh=0;length=1;angle=0;}
MyVector::MyVector(double _dw,double _dh) {dw=_dw;dh=_dh;}
MyVector::MyVector(double angle,double length,int dummy)
{
	dw=length*cos(angle);
	dh=length*sin(angle);
}
double MyVector::GetAngle()   //[-pi,pi]
{
	return atan2(dh,dw);
}
double MyVector::GetLength()
{
	return sqrt(dh*dh+dw*dw);
}
void MyVector::GetPolar(double &angle,double &length)
{
	angle=GetAngle();
	length=GetLength();
}
void MyVector::AdjustLength(double length)
{
	if(GetLength()<0.1) return;   //vector length close to 0
	double rate=length/GetLength();
	dw*=rate;
	dh*=rate;
}
void MyVector::SetAngle(double angle)
{
	if(angle>M_PI) angle-=2*M_PI;
	if(angle<M_PI) angle+=2*M_PI;
	dw=length*cos(angle);
	dh=length*sin(angle);
}


MyVector  MyVector::operator + (MyVector  vec)
{
	return MyVector(dw+vec.dw,dh+vec.dh);
}

double DotProduct(double x1,double y1,double x2,double y2)
{
	return x1*x2+y1*y2;
}

double CrossProduct(MyVector vec1, MyVector vec2)
{
	return vec1.dw*vec2.dh-vec1.dh*vec2.dw;
}

double Norm(double x,double y)
{
	return sqrt(x*x+y*y);
}

void Uniform(double x,double y,double &ux,double &uy)
{
	double l=Norm(x,y);
	ux=x/l;
	uy=y/l;
}

void AddVector(double in_angle,double in_length,double &out_angle,double &out_length)
{
	double out_w=in_length*cos(in_angle)+out_length*cos(out_angle);
	double out_h=in_length*sin(in_angle)+out_length*sin(out_angle);
	out_angle=atan2(out_h,out_w);
	out_length=sqrt(out_w*out_w+out_h*out_h);
}

