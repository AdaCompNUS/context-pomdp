#ifndef PED_STATE_H
#define PED_STATE_H
#include "coord.h"
#include "param.h"
#include "despot/interface/pomdp.h"
//#include "util/util.h"
#include "disabled_util.h"
#include <vector>
#include <utility>
using namespace std;

using namespace despot;
struct PedStruct {
	PedStruct(){
		id = -1;
    vel = ModelParams::PED_SPEED;
    mode = PED_DIS;
  }
	PedStruct(COORD a, int b, int c) {
		pos = a;
		goal = b;
		id = c;
    vel = ModelParams::PED_SPEED;
    mode = PED_DIS;
	}
	PedStruct(COORD a, int b, int c, float speed) {
		pos = a;
		goal = b;
		id = c;
		vel = speed;
    mode = PED_DIS;
	}
	COORD pos; //pos
	int goal;  //goal
	int id;   //id
  double vel;
  int mode;
  COORD dir; // heading dir, for cur_vel motion model
};

class Pedestrian
{
public:
	Pedestrian() {
		last_update = -1;
    }
	Pedestrian(double _w,double _h,int _id) {
        w=_w;h=_h;id=_id;last_update = -1;
    }
	Pedestrian(double _w,double _h) {w=_w;h=_h;
	last_update = -1;
    }

	double w,h;
	COORD vel;
	int id;   //each pedestrian has a unique identity
	double last_update;
};

struct CarStruct {
	COORD pos;
	double vel;
	double heading_dir;/*[0, 2*PI) heading direction with respect to the world X axis */
};

class PomdpState : public State {
public:
	CarStruct car;
	int num;
	PedStruct peds[ModelParams::N_PED_IN];

	float time_stamp;

	PomdpState() {time_stamp = -1;}

	string text() const {
		return concat(car.vel);
		//return "state.h text(): fdfdfdfdfdfdf";
	}
};

class PomdpStateWorld : public State {
public:
	CarStruct car;
	int num;
	PedStruct peds[ModelParams::N_PED_WORLD];

	float time_stamp;

//	int peds_mode[ModelParams::N_PED_WORLD];
	PomdpStateWorld() {time_stamp = -1;}

	string text() const {
		return concat(car.vel);
		//return "state.h text(): fdfdfdfdfdfdf";
	}
};

#endif
