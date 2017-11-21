#ifndef SFM_H
#define SFM_H
#include"math_utils.h"
#include "param.h"
#include "map.h"
#include "window.h"
#include "coord.h"
#include "util_uniform.h"
#include "pedestrian_state.h"
#include<cmath>
#include<vector>
#include<utility>
#include<cstring>

using namespace std;
class SFM;

class Pedestrian
{
public:
	Pedestrian() {}
	Pedestrian(int _w,int _h,int _goal,int _id) {w=_w;h=_h;goal=_goal;id=_id;ts=0;}
	Pedestrian(int _w,int _h) {w=_w;h=_h;}
	void Step()   //Pedestrian move according to SFM Model
	{
		MyVector vec;//=SocialForce(*this);
		vec.AdjustLength(ModelParams::rln);
		w+=vec.dw;
		h+=vec.dh;
	}
	int w,h,goal;
	int id;   //each pedestrian has a unique identity
	int ts;
	int last_update;
	SFM*sfm;
};

class Car
{
public:
	Car(MyMap*pt=0):map(pt) {carPos=w=h=0;}
	Car(MyMap*pt,int pos):map(pt)
	{
		carPos=pos;
		w=map->global_plan[carPos][0];
		h=map->global_plan[carPos][1];
	}
	Car(MyMap*pt,int _w,int _h)
	{
		map=pt;
		w=_w;
		h=_h;
	}

	MyMap *map;
	int carPos;
	int w,h;
};


class SFM 
{
public:
	bool debug;
	int pri_table[9][9][3];
	int next_grid[9][2];
	double phase_angles[8];
	int weight[10];
	MyVector goal_model[ModelParams::XSIZE][ModelParams::YSIZE][ModelParams::NGOAL];
	MyVector car_model[ModelParams::XSIZE][ModelParams::YSIZE][ModelParams::RMMax][ModelParams::NGOAL];
	int crash_model[ModelParams::XSIZE][ModelParams::YSIZE];
	int local_goals[ModelParams::NGOAL][2];
	MyMap *sfm_map;	
	MyWindow *sfm_window;
	double w_angle;
	SFM (MyMap*m_pt,MyWindow*w_pt)
	{
		
		debug=false;
		sfm_map=m_pt;
		sfm_window=w_pt;
		int temp1[9][2]={
			{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,0}
			//{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1},{0,0}
		};
		memcpy(next_grid,temp1,sizeof(temp1));
		int temp2[10]={30,10,4,2,1,8};
		memcpy(weight,temp2,sizeof(temp2));
		double start=0;
		for(int i=0;i<8;i++)
		{
			phase_angles[i]=start;
			start+=ModelParams::pi/4;
		}
		//BuildPriorTable();
	}
	void BuildPriorTable()
	{
		
		for(int i=0;i<8;i++)
		{
			pri_table[i][0][0]=next_grid[i][0];
			pri_table[i][0][1]=next_grid[i][1];
			pri_table[i][0][2]=weight[0];
			int next,delta;
			for(delta=1;delta<4;delta++)
			{
				next=(delta+i)%8;
				pri_table[i][delta*2-1][0]=next_grid[next][0];
				pri_table[i][delta*2-1][1]=next_grid[next][1];

				pri_table[i][delta*2-1][2]=weight[delta];   //probability weight

				next=(i-delta+8)%8;
				pri_table[i][delta*2][0]=next_grid[next][0];
				pri_table[i][delta*2][1]=next_grid[next][1];

				pri_table[i][delta*2][2]=weight[delta];

			}
			next=(delta+i)%8;
			pri_table[i][7][0]=next_grid[next][0];
			pri_table[i][7][1]=next_grid[next][1];
			pri_table[i][7][2]=weight[delta];
			pri_table[i][8][0]=next_grid[8][0];
			pri_table[i][8][1]=next_grid[8][1];
			pri_table[i][8][2]=weight[delta+1];
		}

		pri_table[8][0][0]=0;
		pri_table[8][0][1]=0;
		pri_table[8][0][2]=60;
		for(int i=1;i<9;i++)
		{
			pri_table[8][i][0]=next_grid[i-1][0];
			pri_table[8][i][1]=next_grid[i-1][1];
			pri_table[8][i][2]=4;
		}

	}

	double CalculateWeight(double angle)   //accept a positive angle [0,pi]
	{
		if(ModelParams::goodped==0)
		{
			if(angle<0.15) angle=0.15;
			double weight=ModelParams::pi/angle; 
			//weight=weight/2;
			//if(weight<1) weight=1;
			return sqrt(weight) + 5;
		}
		else
		{
			//if(angle<ModelParams::pi/4) return 100;
			//if(angle<ModelParams::pi/2) return 5;
			//else 	return 2;	
			if(angle<ModelParams::pi/4) return 10000;
			if(angle<ModelParams::pi/2) return 1000;
			else return 2;
		}
	}
	double GetAngularWeight(int phase,double angle,double length)
	{
		if(phase==8)  //not move
		{
			if(ModelParams::goodped==0)
			{
				if(length<1) return 5;
				//if(length<1) return 3+5;
				else return 1.5 + 5;
			}
			else
			{
				if(length<1)  return 10000;
				else 	return 10000;
			}
		}

		double diff;
		diff=angle-phase_angles[phase];  //[-2*pi,2*pi]
		if(diff<-ModelParams::pi) diff=diff+2*ModelParams::pi;
		else if(diff>ModelParams::pi) diff=(2*ModelParams::pi-diff);   //[-pi,pi]
		
		if(diff<0)    diff=-diff;   //[0,pi]
		double weight=CalculateWeight(diff);
	//	if(debug)
	//		cout<<weight<<" "<<diff<<" "<<angle<<" "<<phase<<" "<<phase_angles[phase]<<endl;
		return weight;

	}

	double GetLaneWeight(int lane)
	{
		return 1;
		int mid=ModelParams::XSIZE/2;
		double delta=0.5/(mid);
		return fabs(lane-mid)*delta+0.5;
	}

	int SelectPhase(double angle,double length)
	{
		if(length<1.0) return 8;
		if(fabs((angle))<(ModelParams::pi/8))    return 0; 
		else if((angle)>(ModelParams::pi/8)&&(angle)<(ModelParams::pi/8*3))   return 1;
		else if((angle)>(-ModelParams::pi/8*3)&&(angle)<(-ModelParams::pi/8))  return 7;
		else if((angle)>(ModelParams::pi/8*3)&&(angle)<(ModelParams::pi/8*5))  return 2;
		else if((angle)>(-ModelParams::pi/8*5)&&(angle)<(-ModelParams::pi/8*3))  return 6;
		else if((angle)>(ModelParams::pi/8*5)&&(angle)<(ModelParams::pi/8*7))  return 3;
		else if((angle)>(-ModelParams::pi/8*7)&&(angle)<(-ModelParams::pi/8*5))  return 5;
		else if(fabs((angle))>(ModelParams::pi/8*7))  return 4;

	}

	MyVector PedToGoal(Pedestrian &ped,int goal_w,int goal_h);
	MyVector PedToPed(Pedestrian &ped, vector<Pedestrian> &ped_list);
	//MyVector PedToCar(Pedestrian  &ped,Car & car);
	MyVector PedToCar(Pedestrian  &ped,int carPos);
	MyVector  SocialForce(Pedestrian & ped,vector<Pedestrian> & ped_list,Car& car);
	void ModelTrans(PedestrianState&state,UtilUniform &unif);
	void WorldTrans(vector<Pedestrian> &ped_list,Car&car,UtilUniform &unif);
	void ModelTransFast(PedestrianState&state,UtilUniform &unif);
	double ModelTransProb(PedestrianState state,PedestrianState state_new);
	void UpdateGoalModel();
	void UpdateCarModel();
	void UpdateCrashModel();
	void UpdateSFM()
	{

		w_angle=sfm_window->GetYaw();
		if(ModelParams::debug)   cout<<"print local goals"<<endl;
		for(int i=0;i<ModelParams::NGOAL;i++)
		{
			sfm_window->GlobalToLocal(sfm_map->goal_pos[i][0],sfm_map->goal_pos[i][1],local_goals[i][0],local_goals[i][1]);
			if(ModelParams::debug)   cout<<local_goals[i][0]<<" "<<local_goals[i][1]<<endl;
		}
		UpdateGoalModel();
		UpdateCarModel();
		UpdateCrashModel();

	}
	bool Danger(int x,int y,Car car)
	{
		int pedx,pedy;
		sfm_window->GlobalToLocal(x,y,pedx,pedy);
		int carx,cary;
		sfm_window->GlobalToLocal(car.w,car.h,carx,cary);
		if(pedx==carx&&cary==pedy) return true;
		else return false;
	}
};







#endif
