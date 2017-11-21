#ifndef PED_STATE_H
#define PED_STATE_H
#include "coord.h"
#include "param.h"
#include "core/pomdp.h"
//#include "util/util.h"
#include "disabled_util.h"
#include <vector>
#include <utility>


using namespace std;

struct PedStruct {
	PedStruct(){
        vel = ModelParams::PED_SPEED;
    }
	PedStruct(COORD a, int b, int c) {
		pos = a;
		goal = b;
		id = c;
        vel = ModelParams::PED_SPEED;
	}
	COORD pos; //pos
	int goal;  //goal
	int id;   //id
    double vel;
};

class Pedestrian
{
public:
	Pedestrian() {
    }
	Pedestrian(double _w,double _h,int _id) {
        w=_w;h=_h;id=_id;
    }
	Pedestrian(double _w,double _h) {w=_w;h=_h;
    }

	double w,h;
	int id;   //each pedestrian has a unique identity
	double last_update;
};

struct CarStruct {
	int pos;
	double vel;
	double dist_travelled;
};

class PomdpState : public State {
public:
	CarStruct car;
	int num;
	PedStruct peds[ModelParams::N_PED_IN];
	PomdpState() {}

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
	PomdpStateWorld() {}

	string text() const {
		return concat(car.vel);
		//return "state.h text(): fdfdfdfdfdfdf";
	}
};

#endif
