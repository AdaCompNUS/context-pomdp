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
	double getlength();
	double getCurDir();

	Path copy_without_travelled_points(double dist_to_remove);

	void copy_to(Path& des){
//		des.resize(size());
		des.assign(begin(),end());
	}
};

