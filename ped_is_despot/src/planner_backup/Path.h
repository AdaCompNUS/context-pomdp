#pragma once
#include<vector>
#include"coord.h"
#include"param.h"

struct Path : std::vector<COORD> {
    int nearest(COORD pos);
    double mindist(COORD pos);
    int forward(int i, double len) const;
	double getYaw(int i) const;
	Path interpolate();
	void cutjoin(const Path& p);
};

