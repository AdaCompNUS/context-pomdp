#ifndef Pedestrian_H
#define Pedestrian_H

#include "Model.h"
#include "coord.h"
#include "memorypool.h"

class Pedestrian_State : public MEMORY_OBJECT
{
public:
    COORD RobPos;
    COORD PedPos;
    int   Vel;
    int   Goal;
};

class Pedestrian: public Model
{
public:
    //Pedestrian(int size);
    Pedestrian(string filename);
		int getStartState();
    void FreeState(State* state) const;
    State step(State state, double rNum, int action, double& reward, uint64_t& obs);
		State step(State s, double rNum, int action, double& reward)  {
			uint64_t obs;
			return step(s, rNum, action, reward, obs);
		}
		double obsProb(uint64_t z, State s, int action);
		double fringeUpperBound(State s);
		double fringeLowerBound(vector<Particle>& particles);
		UMAP<State, double> initBelief();

		void InitModel();

		void printState(State state);
    int numStates() const;
		int numActions() const { return 3; }
		bool isTerminal(State s) const {
			return s == numStates() - 1;
		}

		uint64_t terminalObs() const {
			return 150;
		}

		int bestDefaultAction(State s, int rnsPos) {
			return 0;
		}

    int StateToN(Pedestrian_State* state) const;
    void NToState(int n,Pedestrian_State*) const;

protected:
    //ros::ServiceClient * client_pt;
    int Size;
    int trans[5][20][5][20];
    int walk_dirs[5][20][2];
    double qValue[20000][3];
    double Value[20000];
    enum
    {
        ACT_CUR,
        ACT_ACC,
        ACT_DEC
    };
private:
		State startState;
		double Discount;
		Pedestrian_State* dummyState;
    mutable MEMORY_POOL<Pedestrian_State> MemoryPool;
};

#endif
