
#include <despot/planner.h>

using namespace despot;

class SolverPrior;
class DrivingController: public Planner {
public:
	DrivingController(){;}

	DSPOMDP* InitializeModel(option::Option* options) ;
	World* InitializeWorld(std::string& world_type, DSPOMDP* model, option::Option* options);

	void InitializeDefaultParameters() ;
	std::string ChooseSolver();

	bool RunStep(despot::Solver*, despot::World*, despot::Logger*);

	void PlanningLoop(Solver*& solver, World* world, Logger* logger);

	Simulator* driving_simulator_;

	SolverPrior* prior_;

private:
	void CreateNNPriors(DSPOMDP* model);
};
