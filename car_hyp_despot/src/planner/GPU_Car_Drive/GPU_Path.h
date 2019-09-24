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

	Dvc_COORD* way_points_;
	int size_;
	int pos_;

};

