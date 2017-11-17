
#include "GPU_Path.h"
#include<iostream>
#include "math_utils.h"
using namespace std;
/*
DEVICE void Dvc_Path::push_back(Dvc_COORD& src)
{
	way_points_[++pos_]=src;
}*/

DEVICE int Dvc_Path::nearest(Dvc_COORD pos) {
    auto& path = *this;
    float dmin = Dvc_COORD::EuclideanDistance(pos, path.way_points_[0]);
    int imin = 0;
    for(int i=0; i<path.size_; i++) {
        float d = Dvc_COORD::EuclideanDistance(pos, path.way_points_[i]);
        if(dmin > d) {
            dmin = d;
            imin = i;
        }
    }
    return imin;
}

/*DEVICE float Dvc_Path::mindist(Dvc_COORD pos) {
    Dvc_COORD mc = at(nearest(pos));
    float d = Dvc_COORD::EuclideanDistance(mc, pos);
    return d;
}*/

DEVICE int Dvc_Path::forward(int i, float len) const {
    auto& path = *this;
    //while(len > 0 and i<Dvc_Path.size()-1) {
        //float d = Dvc_COORD::EuclideanDistance(Dvc_Path[i], Dvc_Path[i+1]);
        //len -= d;
		//i++;
    //}
    len=len / ModelParams::PATH_STEP;
    if(len-int(len)>1.0-1e-5) i+=int(len+1);
    else
    	i += int(len);
    if(i > path.size_-1) {
        i = path.size_-1;
    }
    return i;
}

DEVICE float Dvc_Path::getYaw(int i) const {
    auto& path = *this;
	//TODO review this code
	
	int j = forward(i, 1.0);
	if(i==j) { i = max(0, i-3);}

	const Dvc_COORD& pos = path.way_points_[i];
	const Dvc_COORD& forward_pos = path.way_points_[j];
	Dvc_Vector vec(forward_pos.x - pos.x, forward_pos.y - pos.y);
    float a = vec.GetAngle(); // is this the yaw angle?
	return a;
}

/*DEVICE Dvc_Path Dvc_Path::interpolate() {
    auto& path = *this;
	Dvc_Path p(path.size_,0);

	const float step = ModelParams::PATH_STEP;
	float t=0, ti=0;
	for(int i=0; i<path.size_-1; i++) {
        float d = Dvc_COORD::EuclideanDistance(path.way_points_[i], path.way_points_[i+1]);
		float dx = (path.way_points_[i+1].x-path.way_points_[i].x) / d;
		float dy = (path.way_points_[i+1].y-path.way_points_[i].y) / d;
		float sx = path.way_points_[i].x;
		float sy = path.way_points_[i].y;
		while(t < ti+d) {
			float u = t - ti;
			float nx = sx + dx*u;
			float ny = sy + dy*u;
			p.push_back(Dvc_COORD(nx, ny));
			t += step;
		}

		ti += d;

	}
	p.push_back(path.way_points_[path.size_-1]);
	return p;
}

DEVICE void Dvc_Path::cutjoin(const Dvc_Path& p) {
	int i = max(0, nearest(p[0])-1);
	erase(begin()+i, end());
	insert(end(), p.begin()+1, p.end());
}*/
