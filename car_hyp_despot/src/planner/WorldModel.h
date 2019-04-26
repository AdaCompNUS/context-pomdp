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

    std::vector<std::vector<double>> prob_modes_goals;
    int sample_goal() const;
    int maxlikely_goal() const;

    void sample_goal_mode(int& goal, int& mode) const;
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
	ACT_TYPE defaultStatePolicy(const State* _state) const;
    int defaultGraphEdge(const State* particle, bool print_debug);

//    bool isLocalGoal(const PomdpState& state);
//    bool isLocalGoal(const PomdpStateWorld& state);

    bool isGlobalGoal(const CarStruct& car);
	bool inFront(const COORD ped_pos, const CarStruct car) const;
    
    bool inCollision(const PomdpState& state);
    bool inCollision(const PomdpStateWorld& state);
    bool inRealCollision(const PomdpStateWorld& state);
    
    bool inCollision(const PomdpState& state, int &id);
    bool inCollision(const PomdpStateWorld& state, int &id);
    bool inRealCollision(const PomdpStateWorld& state, int &id);
    
    int minStepToGoal(const PomdpState& state);
    int minStepToGoalWithSteer(const PomdpState& state);

    int hasMinSteerPath(const PomdpState& state);

	void PedStep(PedStruct &ped, Random& random);
	void PedStep(PedStruct &ped, double& random);

    double ISPedStep(CarStruct &car, PedStruct &ped, Random& random);//importance sampling PedStep
    void RVO2PedStep(PedStruct peds[], Random& random, int num_ped); //no interaction between car and pedestrian
    void RVO2PedStep(PedStruct peds[], Random& random, int num_ped, CarStruct car); //pedestrian also need to consider car when moving
    void RVO2PedStep(PedStruct peds[], double& random, int num_ped); //no interaction between car and pedestrian
    void RVO2PedStep(PedStruct peds[], double& random, int num_ped, CarStruct car); //pedestrian also need to consider car when moving
    void RVO2PedStep(PomdpStateWorld& state, Random& random);
    void PedStepDeterministic(PedStruct& ped, int step);
    void PedStepCurVel(PedStruct& ped, COORD vel);

    void FixGPUVel(CarStruct &car);
	void RobStep(CarStruct &car, double steering, Random& random);
	void RobStep(CarStruct &car, double steering, double& random);
	void RobStep(CarStruct &car, double& random, double acc, double steering);
    void RobStep(CarStruct &car, Random& random, double acc, double steering);
    void RobVelStep(CarStruct &car, double acc, Random& random);
    void RobVelStep(CarStruct &car, double acc, double& random);
    double ISRobVelStep(CarStruct &car, double acc, Random& random);//importance sampling RobvelStep

    void RobStepCurVel(CarStruct &car);
    void RobStepCurAction(CarStruct &car, double acc, double steering);

    double pedMoveProb(COORD p0, COORD p1, int goal_id);
    double pedMoveProb(COORD prev, COORD curr, int ped_id, int goal_id, int ped_mode);

    void setPath(Path path);
    void updatePedBelief(PedBelief& b, const PedStruct& curr_ped);
    PedBelief initPedBelief(const PedStruct& ped);

    inline int GetThreadID(){return Globals::MapThread(this_thread::get_id());}
    void InitRVO();


	Path path;
    COORD car_goal;
    std::vector<COORD> goals;
    double freq;
    const double in_front_angle_cos;
    std::vector<RVO::RVOSimulator*> ped_sim_;

    std::vector<COORD> obstacles;

public:
    std::string goal_file_name_;
    void InitPedGoals();
    void AddObstacle(std::vector<RVO::Vector2> obstacles);
    bool CheckCarWithObstacles(const CarStruct& car, int flag);// 0 in search, 1 real check
    bool CheckCarWithObsLine(const CarStruct& car, COORD& obs_last_point, COORD& obs_first_point, int flag);// 0 in search, 1 real check

    /// lets drive
    vector<vector<COORD>> ped_mean_dirs;
    COORD DistractedPedMeanDir(COORD& ped, int goal_id);
    COORD AttentivePedMeanDir(int ped_id, int goal_id);

    void add_car_agents(int num_peds, CarStruct& car);

    void PrepareAttentivePedMeanDirs(std::map<int, PedBelief> peds, CarStruct& car);

    void PrintMeanDirs(std::map<int, PedBelief> old_peds, map<int, PedStruct>& curr_peds);
};

class WorldStateTracker {
public:
    typedef pair<float, Pedestrian> PedDistPair;

    WorldStateTracker(WorldModel& _model): model(_model) {
    	car_heading_dir = 0;
    	carvel = 0;
    }

    void updatePed(const Pedestrian& ped, bool doPrint = false);
    void updateCar(const CarStruct car);
    void updateVel(double vel);
    void cleanPed();
    void removePeds();

    bool emergency();

    std::vector<PedDistPair> getSortedPeds(bool doPrint = false);

    PomdpState getPomdpState();

    COORD getPedVel(int ped_id);

    // Car state
    COORD carpos;
    double carvel;
    double car_heading_dir;

    //Ped states
    std::vector<Pedestrian> ped_list;
    WorldModel& model;
};

class WorldBeliefTracker {
public:
    WorldBeliefTracker(WorldModel& _model, WorldStateTracker& _stateTracker): model(_model), stateTracker(_stateTracker) {}

    void update();
    PomdpState sample(bool predict = false);
    vector<PomdpState> sample(int num, bool predict = false);
    vector<PedStruct> predictPeds();
    PomdpState predictPedsCurVel(PomdpState*, double acc, double steering);

    WorldModel& model;
    WorldStateTracker& stateTracker;
    CarStruct car;
    std::map<int, PedBelief> peds;
	std::vector<PedBelief> sorted_beliefs;

    void PrintState(const State& s, ostream& out = cout) const;
    void printBelief() const;

    PomdpState text() const;

public:
    double cur_time_stamp;

    double cur_acc;
    double cur_steering;
};

enum PED_MODES{
	PED_ATT=0,
	PED_DIS=1
};

double cap_angle(double angle);
int ClosestInt(double v);
int FloorIntRobust(double v);
