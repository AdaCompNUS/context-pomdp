#ifndef ROS_SimulationEngine_H
#define ROS_SimulationEngine_H

#include <vector>
#include <string>
#include "MOMDP.h"
using namespace std;
using namespace momdp;

namespace momdp
{
    class AlphaVectorPolicy;
    class SolverParams;

    class ROS_SimulationEngine
    {
        private:
            SharedPointer<MOMDP> problem;
            SharedPointer<AlphaVectorPolicy> policy;
            SolverParams * solverParams;


        public:
            ROS_SimulationEngine();
            void setup(SharedPointer<MOMDP> problem, SharedPointer<AlphaVectorPolicy> policy, SolverParams * solverParams);
            
            void performActionObs(belief_vector& outBelObs, int action, const BeliefWithState& belSt) const;
            void performActionUnobs(belief_vector& outBelUnobs, int action, const BeliefWithState& belSt, int currObsState) const;
            void getPossibleObservations(belief_vector& possObs, int action, const BeliefWithState& belSt) const;
            string toString();

            double getReward(const BeliefWithState& belst, int action);

            void checkTerminal(string o, string s, vector<int> &bhout, vector<int> &fhout);
            int getGreedyAction(vector<int> &, vector<int> &);



            void display(belief_vector& b, ostream& s);
            //int runFor(int iters, ofstream* streamOut, double& reward, double& expReward);
            void runStep(SharedPointer<BeliefWithState>& currBelSt, int currAction, int currObservation, int nextSVal,   SharedPointer<BeliefWithState>& nextBelSt );

			


            virtual ~ROS_SimulationEngine(void);
    };



}
#endif

