#ifndef CARWORLDSIMULATOR_H
#define CARWORLDSIMULATOR_H
#include <despot/planner.h>

#include "coord.h"
#include "Path.h"
#include "state.h"
#include "WorldModel.h"
#include <despot/interface/pomdp.h>
#include <despot/core/pomdp_world.h>
using namespace despot;

class Simulator:public POMDPWorld {
public:
    typedef pair<float, Pedestrian> PedDistPair;

    Simulator(DSPOMDP* model, unsigned seed=0);
    ~Simulator();

    int numPedInArea(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world);

    int numPedInCircle(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world, double car_x, double car_y);
    int run(int argc, char *argv[]);

    PedStruct randomPed();
    PedStruct randomFarPed(double car_x, double car_y);
    PedStruct randomPedAtCircleEdge(double car_x, double car_y);

    void generateFixedPed(PomdpState &s);

    //virtual DSPOMDP* InitializeModel(option::Option* options) ;
	//virtual void InitializeDefaultParameters();

	void ImportPeds(std::string filename, PomdpStateWorld& world_state);
	void ExportPeds(std::string filename, PomdpStateWorld& world_state);

	void PrintWorldState(PomdpStateWorld state, ostream& out = cout);

	void UpdateWorld();

    COORD start, goal;

    Path path;
    static WorldModel worldModel;

    WorldStateTracker* stateTracker;
	PomdpStateWorld world_state;
	int num_of_peds_world;

	Random* rand_;

public:

	virtual bool Connect(){ return true;}

	virtual State* Initialize();

	virtual State* GetCurrentState() const;

	virtual bool ExecuteAction(ACT_TYPE action, OBS_TYPE& obs);

private:
	int random_ped_mode(PedStruct& ped);
};

#endif
