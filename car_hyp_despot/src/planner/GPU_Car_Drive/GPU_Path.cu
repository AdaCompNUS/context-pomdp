
#include "GPU_Path.h"
#include<iostream>
#include "math_utils.h"
using namespace std;


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

DEVICE int Dvc_Path::forward(int i, float len) const {
    auto& path = *this;

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
	
	int j = forward(i, 1.0);
	if(i==j) { i = max(0, i-3);}

	const Dvc_COORD& pos = path.way_points_[i];
	const Dvc_COORD& forward_pos = path.way_points_[j];
	Dvc_Vector vec(forward_pos.x - pos.x, forward_pos.y - pos.y);
    float a = vec.GetAngle(); 
	return a;
}
