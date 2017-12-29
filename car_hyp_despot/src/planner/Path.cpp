
#include "Path.h"
#include<iostream>
#include "math_utils.h"
#include <fstream>
using namespace std;



int Path::nearest(COORD pos) {
    auto& path = *this;
    double dmin = COORD::EuclideanDistance(pos, path[0]);
    int imin = 0;
    for(int i=0; i<path.size(); i++) {
        double d = COORD::EuclideanDistance(pos, path[i]);
        if(dmin > d) {
            dmin = d;
            imin = i;
        }
    }
    return imin;
}

double Path::mindist(COORD pos) {
    COORD mc = at(nearest(pos));
    double d = COORD::EuclideanDistance(mc, pos);
    return d;
}

int Path::forward(int i, double len) const {
    auto& path = *this;
    //while(len > 0 and i<path.size()-1) {
        //double d = COORD::EuclideanDistance(path[i], path[i+1]);
        //len -= d;
		//i++;
    //}
    float step=(len / ModelParams::PATH_STEP);

    if(step-(int)step>1.0-1e-5)
    {
        //cout<<"original step="<< step<<" new step"<<(int)(step+1)<<endl;
    	step++;
    }
    i += (int)(step);
//    i += int(len / ModelParams::PATH_STEP);

    if(i > path.size()-1) {
        i = path.size()-1;
    }
    return i;
}

double Path::getYaw(int i) const {
    auto& path = *this;
	//TODO review this code
	
	int j = forward(i, 1.0);
	if(i==j) { i = max(0, i-3);}

	const COORD& pos = path[i];
	const COORD& forward_pos = path[j];
	MyVector vec(forward_pos.x - pos.x, forward_pos.y - pos.y);
    double a = vec.GetAngle(); // is this the yaw angle?
	return a;
}

Path Path::interpolate() {
    auto& path = *this;
	Path p;

	const double step = ModelParams::PATH_STEP;
	double t=0, ti=0;
	for(int i=0; i<path.size()-1; i++) {
        double d = COORD::EuclideanDistance(path[i], path[i+1]);
		double dx = (path[i+1].x-path[i].x) / d;
		double dy = (path[i+1].y-path[i].y) / d;
		double sx = path[i].x;
		double sy = path[i].y;
		while(t < ti+d) {
			double u = t - ti;
			double nx = sx + dx*u;
			double ny = sy + dy*u;
			p.push_back(COORD(nx, ny));
			t += step;
		}

		ti += d;

		/*
		int n = int(d/ModelParams::PATH_STEP);
		double dx,dy;
		dx=(path[i+1].x-path[i].x)/n;
		dy=(path[i+1].y-path[i].y)/n;
		double nx,ny;
		nx=path[i].x;
		ny=path[i].y;
		for(int j=0;j<n;j++) {
			p.push_back(COORD(nx,ny));	
			nx+=dx;
			ny+=dy;
		}
		*/
	}
	p.push_back(path[path.size()-1]);
	return p;
}

void Path::cutjoin(const Path& p) {
	int i = max(0, nearest(p[0])-1);
	erase(begin()+i, end());
	insert(end(), p.begin()/*+1*/, p.end());
}
