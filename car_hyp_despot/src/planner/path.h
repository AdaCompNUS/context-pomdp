#pragma once
#include<vector>
#include"coord.h"
#include"param.h"

struct Path : std::vector<COORD> {
    int Nearest(const COORD pos) const;
    double MinDist(COORD pos);
    int Forward(double i, double len) const;
	double GetYaw(int i) const;
	Path Interpolate(double max_len = 10000.0) const;
	void CutJoin(const Path& p);

	double GetLength(int start=0);
	double GetCurDir(int pos_along = 0);
	COORD GetCrossDir(int, bool);

	void Text();

	void CopyTo(Path& des){
		des.assign(begin(),end());
	}
};

double CapAngle(double x);
