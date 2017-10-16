#ifndef WINDOW_H
#define WINDOW_H
#include <vector>
#include <iostream>
#include <utility>
#include "math_utils.h"
#include "map.h"
#include "param.h"
#include <cmath>

using namespace std;
class MyWindow  {
public:

	MyWindow(MyMap * pt) {map=pt;}
	MyWindow(MyMap* pt,int wO)
	{
		map=pt;	
		RollWindow(wO);
	}
	void RollWindow(int wO)
	{
		cout<<"window Origin "<<wO<<endl;
		wOrigin=wO;
		//int L=ModelParams::YSIZE*ModelParams::path_rln;
		int L=10*ModelParams::path_rln;
		int end=(wOrigin+L<map->pathLength)?(wOrigin+L):(map->pathLength-1);
		double dh=map->global_plan[end][1]-map->global_plan[wOrigin][1];
		double dw=map->global_plan[end][0]-map->global_plan[wOrigin][0];	
		double angle=atan2(dw,dh);	
		double yaw=sin(-angle/2);
		MapWindowSplit(map->global_plan[wOrigin][0],map->global_plan[wOrigin][1],yaw);
		UpdateRobMap(wOrigin);
		for(int i=0;i<ModelParams::XSIZE;i++)
			for(int j=0;j<ModelParams::YSIZE;j++)
				LocalToGlobalPreComp(i,j,LGMap[i][j][0],LGMap[i][j][1]);	
	}
	int GetIndex(int i,int j);
	void StepN(int w,int h,int & dest_w,int & dest_h,double angle,int step_size);
	bool LeftHanded(int x,int y,int vx,int vy); 
	bool InWindow(int w,int h);
	void MapWindowSplit(double w,double h,double yaw);
	void GlobalToLocal(int globalW,int globalH,int &localW,int&localH);
	void LocalToGlobal(int localW,int localH,int & globalW,int & globalH);
	void LocalToGlobalPreComp(int localW,int localH,int & globalW,int & globalH);
	void UpdateRobMap(int);
	double GetYaw();
	int w0,h0,w1,h1,w2,h2,w3,h3;
	MyMap*map;
	int wOrigin;
	vector<pair<int,int> > rob_map;
	int LGMap[ModelParams::XSIZE][ModelParams::YSIZE][2];
};



#endif
