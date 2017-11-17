#ifndef CARWORLDSIMULATOR_H
#define CARWORLDSIMULATOR_H
#include <despot/simple_tui.h>

#include "coord.h"
#include "Path.h"
#include "state.h"
#include "WorldModel.h"
using namespace despot;

class Simulator:public SimpleTUI {
public:
    typedef pair<float, Pedestrian> PedDistPair;

    Simulator();
    ~Simulator();

    int numPedInArea(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world);

    int numPedInCircle(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world, double car_x, double car_y);
    int run(int argc, char *argv[]);

    PedStruct randomPed();
    PedStruct randomFarPed(double car_x, double car_y);
    PedStruct randomPedAtCircleEdge(double car_x, double car_y);

    void generateFixedPed(PomdpState &s);


    virtual DSPOMDP* InitializeModel(option::Option* options) ;
	virtual void InitializeDefaultParameters();

	virtual void InitializedGPUModel(std::string rollout_type, DSPOMDP* Hst_model=NULL);
	virtual void InitializeGPUGlobals();
	virtual void InitializeGPUPolicyGraph(PolicyGraph* hostGraph);
	virtual void DeleteGPUModel();
	virtual void DeleteGPUGlobals();
	virtual void DeleteGPUPolicyGraph();

    virtual void UpdateGPUGoals(DSPOMDP* Hst_model);
    virtual void UpdateGPUPath(DSPOMDP* Hst_model);

	void ImportPeds(std::string filename, PomdpStateWorld& world_state);
	void ExportPeds(std::string filename, PomdpStateWorld& world_state);

    COORD start, goal;

    Path path;
    WorldModel worldModel;

};

#endif
