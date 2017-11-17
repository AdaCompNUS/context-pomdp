#pragma once
#include <vector>
#include <despot/GPUutil/GPUcoord.h>
#include "GPU_param.h"
using namespace despot;
struct Dvc_Path{

    DEVICE int nearest(Dvc_COORD pos);
    DEVICE float mindist(Dvc_COORD pos);
    DEVICE int forward(int i, float len) const;
    DEVICE float getYaw(int i) const;
    //DEVICE Dvc_Path interpolate();
    //DEVICE void cutjoin(const Path& p);


	Dvc_COORD* way_points_;
	int size_;
	int pos_;

	/*DEVICE Dvc_Path(int size, int pos){
		size_=size;
		way_points_=(Dvc_COORD*)malloc(size*sizeof(Dvc_COORD));
		pos_=pos;
	}*/

	//DEVICE void push_back(Dvc_COORD&);

};

