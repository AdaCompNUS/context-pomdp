#ifndef AGENT_STATE_H
#define AGENT_STATE_H
#include <vector>
#include <utility>

#include "coord.h"
#include "param.h"
#include "path.h"
#include "disabled_util.h"
#include "despot/interface/pomdp.h"

using namespace std;

using namespace despot;

enum AgentType { car=0, ped=1, num_values=2};

struct AgentStruct {
	
	AgentStruct(){
		set_default_values();
  	}
	
	AgentStruct(COORD a, int b, int c) {
		set_default_values();
		pos = a;
		intention = b;
		id = c;
	}
	
	AgentStruct(COORD a, int b, int c, float _speed) {
		set_default_values();
		pos = a;
		intention = b;
		id = c;
		speed = _speed;
	}

	void set_default_values(){
		intention = -1;
		id = -1;
		speed = 0.0;
		mode = 1; //PED_DIS
		type = AgentType::car;
		pos_along_path = 0;
		bb_extent_x = 0;
		bb_extent_y = 0;
		heading_dir = 0;
		cross_dir = 0;
	}

	COORD pos;
    int mode;
	int intention; // intended path
	int pos_along_path; // traveled distance along the path
	int cross_dir;
	int id;
	AgentType type;
    double speed;
    COORD vel;
    double heading_dir;
    double bb_extent_x, bb_extent_y;

    void text(std::ostream& out) const {
    	out << "agent: id / pos / speed / vel / intention / dist2car / infront =  "
			<< id << " / "
			<< "(" << pos.x << ", " << pos.y << ") / "
			<< speed << " / "
			<< "(" << vel.x << ", " << vel.y << ") / "
			<< intention << " / "
			<< " (mode) " << mode
			<< " (type) " << type
			<< " (bb) " << bb_extent_x
			<< " " << bb_extent_y
			<< " (cross) " << cross_dir
			<< " (heading) " << heading_dir << endl;
    }
};


class Agent
{
public:
	Agent() {
		set_default();
    }

	Agent(double _w,double _h,int _id) {
		set_default();
        w=_w;h=_h;id=_id;
    }

	Agent(double _w,double _h) {
		set_default();
		w=_w;h=_h;
    }

	virtual ~Agent() {}

	void set_default() {
		time_stamp = -1; vel.x = 0; vel.y = 0;
		reset_intention = false; ros_time_stamp = 0;
		h = 0; w = 0;
		id = -1;
	}
    virtual AgentType type() const = 0;

	double w,h;
	COORD vel;
	int id;   //each pedestrian has a unique identity
	double time_stamp;
	double ros_time_stamp;
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

	PomdpState() {time_stamp = -1; num = 0;}

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
	PomdpStateWorld() {time_stamp = -1; num = 0;}

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
