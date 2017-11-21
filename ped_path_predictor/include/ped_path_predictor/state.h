#ifndef STATE_PATH_PREDICTOR_H
#define STATE_PATH_PREDICTOR_H
#include <ped_path_predictor/coord.h>
#include <ped_path_predictor/param.h>
#include <vector>
#include <utility>
using namespace std;

struct PedStruct {
	COORD pos;
    COORD vel_cur; //current velocity
    COORD vel_pref; //preferred velocity
    int id; //pedestrian id
    int goal;  //pedesrtian goal (optional interface)

    double last_update; //last update time;

	PedStruct(){
        vel_cur.x = ModelParams::PED_SPEED;
        vel_cur.y = 0;

        vel_pref = vel_cur;
    }
	PedStruct(COORD pos, COORD vel_cur, int id = -1, int goal = -1) {
		this->pos = pos;
		this->vel_cur = vel_cur;
		this->vel_pref = vel_cur;

		this->id = id;
		this->goal = goal;
	}

	PedStruct(COORD pos, int id = -1, int goal = -1) {
		this->pos = pos;

		this->id = id;
		this->goal = goal;

		vel_cur.x = ModelParams::PED_SPEED;
        vel_cur.y = 0;

        vel_pref = vel_cur;
	}
	
};

struct CarStruct {
	COORD pos;
	COORD vel;
	double dist_travelled;
};


#endif
