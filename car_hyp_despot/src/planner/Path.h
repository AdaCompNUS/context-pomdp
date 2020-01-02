#pragma once
#include<vector>
#include"coord.h"
#include"param.h"

struct Path : std::vector<COORD> {
    int nearest(const COORD pos) const;
    double mindist(COORD pos);
    int forward(double i, double len) const;
	double getYaw(int i) const;
	Path interpolate(double max_len = 10000.0) const;
	void cutjoin(const Path& p);
	double getlength(int start=0);
	double getCurDir(int pos_along = 0);

	COORD GetCrossDir(int, bool);

	void text();

	void copy_to(Path& des){
//		des.resize(size());
		des.assign(begin(),end());
	}
};

double CapAngle(double x);
