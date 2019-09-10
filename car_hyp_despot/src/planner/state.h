#ifndef PED_STATE_H
#define PED_STATE_H
#include "coord.h"
#include "param.h"
#include "Path.h"
#include "despot/interface/pomdp.h"
//#include "util/util.h"
#include "disabled_util.h"
#include <vector>
#include <utility>
using namespace std;

using namespace despot;

enum AgentType { car=0, ped=1, num_values=2};

struct AgentStruct {
	
	AgentStruct(){
		id = -1;
    	speed = -1;
    	mode = 1;//PED_DIS;
  	}
	
	AgentStruct(COORD a, int b, int c) {
		pos = a;
		intention = b;
		id = c;
    	speed = -1;
    	mode = 1;//PED_DIS;
	}
	
	AgentStruct(COORD a, int b, int c, float speed) {
		pos = a;
		intention = b;
		id = c;
		speed = speed;
    	mode = 1;//PED_DIS;
	}

	COORD pos; //pos
    int mode;
	int intention;  //intended path
	int pos_along_path; // traveled distance along the path
	int cross_dir;
	int id;   //id
	AgentType type;
    double speed;
    COORD vel; // heading dir, for cur_vel motion model
    // std::vector<COORD> bb;
    double heading_dir;
    double bb_extent_x, bb_extent_y;
};


class Agent
{
public:
	Agent() {
		time_stamp = -1; vel.x = 0; vel.y = 0;
    }
	Agent(double _w,double _h,int _id) {
        w=_w;h=_h;id=_id;time_stamp = -1; vel.x = 0; vel.y = 0;
    }
	Agent(double _w,double _h) {w=_w;h=_h;
		time_stamp = -1; vel.x = 0; vel.y = 0;
    }

    virtual AgentType type() const = 0;

	double w,h;
	COORD vel;
	int id;   //each pedestrian has a unique identity
	double time_stamp;
	// double last_update;

	bool reset_intention;
	std::vector<Path> paths;	
};

class Pedestrian: public Agent
{
public:
	using Agent::Agent;

	bool cross_dir;

	AgentType type() const{
		return AgentType::ped;
	}
};

class Vehicle: public Agent
{
public:
	using Agent::Agent;

	std::vector<COORD> bb;
	double heading_dir;
	AgentType type() const {
		return AgentType::car;
	}
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
	AgentStruct agents[ModelParams::N_PED_IN];

	float time_stamp;

	PomdpState() {time_stamp = -1;}

	string text() const {
		return concat(car.vel);
	}
};

class PomdpStateWorld : public State {
public:
	CarStruct car;
	int num;
	AgentStruct agents[ModelParams::N_PED_WORLD];

	float time_stamp;

//	int peds_mode[ModelParams::N_PED_WORLD];
	PomdpStateWorld() {time_stamp = -1;}

	string text() const {
		return concat(car.vel);
	}

	void assign(PomdpStateWorld& src){
		car.pos = src.car.pos;
		car.vel = src.car.vel;
		car.heading_dir = src.car.heading_dir;
		num = src.num;
		for (int i = 0; i < num; i++){
			agents[i].pos = src.agents[i].pos;
			agents[i].mode = src.agents[i].mode;
			agents[i].intention = src.agents[i].intention;
			agents[i].pos_along_path = src.agents[i].pos_along_path;
			agents[i].cross_dir = src.agents[i].cross_dir;
			agents[i].id = src.agents[i].id;
			agents[i].type = src.agents[i].type;
			agents[i].speed = src.agents[i].speed;
			agents[i].vel = src.agents[i].vel;
			agents[i].bb_extent_x = src.agents[i].bb_extent_x;
			agents[i].bb_extent_y = src.agents[i].bb_extent_y;
			agents[i].heading_dir = src.agents[i].heading_dir;
		}
		time_stamp = src.time_stamp;
	}
};

#endif
