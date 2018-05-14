#pragma once
#include"state.h"
#include"Path.h"
#include <RVO.h>
using namespace despot;

struct PedBelief {
	int id;
    COORD pos;
    double vel;
    std::vector<double> prob_goals;
    int sample_goal() const;
    int maxlikely_goal() const;
};

class WorldModel {
public:

    WorldModel();
    ~WorldModel();

	bool isMovingAway(const PomdpState& state, int ped);
	void getClosestPed(const PomdpState& state, int& closest_front_ped, double& closest_front_dist,
			int& closest_side_ped, double& closest_side_dist);
	double getMinCarPedDist(const PomdpState& state);
	double getMinCarPedDistAllDirs(const PomdpState& state);
	int defaultPolicy(const std::vector<State*>& particles);
    int defaultGraphEdge(const State* particle, bool print_debug);

    bool isLocalGoal(const PomdpState& state);
    bool isLocalGoal(const PomdpStateWorld& state);

    bool isGlobalGoal(const CarStruct& car);
	bool inFront(COORD ped_pos, int car) const;
    
    bool inCollision(const PomdpState& state);
    bool inRealCollision(const PomdpState& state);
    bool inCollision(const PomdpStateWorld& state);
    
    bool inCollision(const PomdpState& state, int &id);
    bool inCollision(const PomdpStateWorld& state, int &id);
    
    int minStepToGoal(const PomdpState& state);

	void PedStep(PedStruct &ped, Random& random);
	void PedStep(PedStruct &ped, double& random);

    double ISPedStep(CarStruct &car, PedStruct &ped, Random& random);//importance sampling PedStep
    void RVO2PedStep(PedStruct peds[], Random& random, int num_ped); //no interaction between car and pedestrian
    void RVO2PedStep(PedStruct peds[], Random& random, int num_ped, CarStruct car); //pedestrian also need to consider car when moving
    void RVO2PedStep(PedStruct peds[], double& random, int num_ped); //no interaction between car and pedestrian
    void RVO2PedStep(PedStruct peds[], double& random, int num_ped, CarStruct car); //pedestrian also need to consider car when moving
    void PedStepDeterministic(PedStruct& ped, int step);
    void FixGPUVel(CarStruct &car);
	void RobStep(CarStruct &car, Random& random);
	void RobStep(CarStruct &car, double& random, double acc);
    void RobStep(CarStruct &car, Random& random, double acc);
    void RobVelStep(CarStruct &car, double acc, Random& random);
    void RobVelStep(CarStruct &car, double acc, double& random);
    double ISRobVelStep(CarStruct &car, double acc, Random& random);//importance sampling RobvelStep

    double pedMoveProb(COORD p0, COORD p1, int goal_id);
    void setPath(Path path);
    void updatePedBelief(PedBelief& b, const PedStruct& curr_ped);
    PedBelief initPedBelief(const PedStruct& ped);

    inline int GetThreadID(){return MapThread(this_thread::get_id());}
    void InitRVO();


	Path path;
    std::vector<COORD> goals;
    double freq;
    const double in_front_angle_cos;
    std::vector<RVO::RVOSimulator*> ped_sim_;

public:
    std::string goal_file_name_;
    void InitPedGoals();
};

class WorldStateTracker {
public:
    typedef pair<float, Pedestrian> PedDistPair;

    WorldStateTracker(WorldModel& _model): model(_model) {}

    void updatePed(const Pedestrian& ped);
    void updateCar(const COORD& car, const double dist=0);
    void updateVel(double vel);
    void cleanPed();

    bool emergency();

    std::vector<PedDistPair> getSortedPeds();

    PomdpState getPomdpState();

    COORD carpos;
    double carvel;
    double car_dist_trav;

    std::vector<Pedestrian> ped_list;

    WorldModel& model;
};

class WorldBeliefTracker {
public:
    WorldBeliefTracker(WorldModel& _model, WorldStateTracker& _stateTracker): model(_model), stateTracker(_stateTracker) {}

    void update();
    PomdpState sample();
    vector<PomdpState> sample(int num);
    vector<PedStruct> predictPeds();

    WorldModel& model;
    WorldStateTracker& stateTracker;
    CarStruct car;
    std::map<int, PedBelief> peds;
	std::vector<PedBelief> sorted_beliefs;

    void PrintState(const State& s, ostream& out = cout) const;
    void printBelief() const;
};


