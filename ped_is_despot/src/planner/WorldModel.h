#pragma once
#include"state.h"
#include"Path.h"
#include <RVO.h>

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

	bool isMovingAway(const PomdpState& state, int ped);
	void getClosestPed(const PomdpState& state, int& closest_front_ped, double& closest_front_dist,
			int& closest_side_ped, double& closest_side_dist);
	double getMinCarPedDist(const PomdpState& state);
	double getMinCarPedDistAllDirs(const PomdpState& state);
	int defaultPolicy(const vector<State*>& particles);
    
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
    double ISPedStep(CarStruct &car, PedStruct &ped, Random& random);//importance sampling PedStep

    void RVO2PedStep(PedStruct peds[], Random& random, int num_ped); //no interaction between car and pedestrian
    void RVO2PedStep(PedStruct peds[], Random& random, int num_ped, CarStruct car); //pedestrian also need to consider car when moving
    //double RVO2ISPedStep(CarStruct &car, PedStruct &ped, Random& random);//importance sampling PedStep

    void PedStepDeterministic(PedStruct& ped, int step);
	void RobStep(CarStruct &car, Random& random);
    void RobStep(CarStruct &car, Random& random, double acc); ///use ave vel to compute dist
    void RobVelStep(CarStruct &car, double acc, Random& random);
    double ISRobVelStep(CarStruct &car, double acc, Random& random);//importance sampling RobvelStep

    double pedMoveProb(COORD p0, COORD p1, int goal_id);
    void setPath(Path path);
    void updatePedBelief(PedBelief& b, const PedStruct& curr_ped);
    PedBelief initPedBelief(const PedStruct& ped);


	Path path;
    std::vector<COORD> goals;
    double freq;
    const double in_front_angle_cos;

    RVO::RVOSimulator* ped_sim_;
};

class WorldStateTracker {
public:
    typedef pair<float, Pedestrian> PedDistPair;

    WorldStateTracker(WorldModel& _model): model(_model) {}

    void updatePed(const Pedestrian& ped);
    void updateCar(const COORD& car);
    void updateVel(double vel);
    void cleanPed();

    bool emergency();

    std::vector<PedDistPair> getSortedPeds();

    PomdpState getPomdpState();

    COORD carpos;
    double carvel;

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

