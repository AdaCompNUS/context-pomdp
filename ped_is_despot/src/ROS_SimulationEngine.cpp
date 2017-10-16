#include <sstream>
#include <fstream>
#include "ROS_SimulationEngine.h"
#include "AlphaVectorPolicy.h"
#include "CPTimer.h"
#include "solverUtils.h"

using namespace std;
using namespace momdp;

namespace momdp
{
    void printTuple(map<string, string> tuple, ofstream* streamOut){
	*streamOut << "(";
	for(map<string, string>::iterator iter = tuple.begin() ; iter != tuple.end() ; )
	{
	    *streamOut << iter->second;
	    if(++iter!=tuple.end())
		*streamOut << ",";
	}
	*streamOut<<")" << endl;
    }

    ROS_SimulationEngine::ROS_SimulationEngine()
    {
    }

    ROS_SimulationEngine::~ROS_SimulationEngine(void)
    {
    }

    void ROS_SimulationEngine::checkTerminal(string p, string s, vector<int> &bhout, vector<int> &fhout) {
        if (s.substr(0,3) == "bt2") {
            if (s.substr(9,2) == "FH") {
                int ind = atoi(p.substr(4,1).c_str());
                fhout[ind]++;
            } else if (s.substr(9,2) == "BH") {
                int ind = atoi(p.substr(4,1).c_str());
                bhout[ind]++;
            }
        }
    }

    int ROS_SimulationEngine::getGreedyAction(vector<int> &bhout, vector<int> &fhout) {
        int greedyAction = 2; //start with BHL
        int currBest = bhout[0];

        vector<int> temp;
        for (int i=0;i<(int)bhout.size();i++){
            temp.push_back(bhout[i]);
        }
        for (int i=0;i<(int)fhout.size();i++){
            temp.push_back(fhout[i]);
        }
        
        for (int i=1; i<(int)temp.size();i++){
            if (temp[i]>currBest) {
                currBest = temp[i];
                greedyAction = 2+i;
            }
        }

        return greedyAction;
    }

    void ROS_SimulationEngine::setup(SharedPointer<MOMDP> problem, SharedPointer<AlphaVectorPolicy> policy, SolverParams * solverParams)
    {
        this->policy = policy;
        this->problem = problem;
        this->solverParams = solverParams;
    }

    void ROS_SimulationEngine::performActionObs(belief_vector& outBelObs, int action, const BeliefWithState& belSt) const 
    {
        // DEBUG_SIMSPEED_270409 skip calculating outprobs for x when there is only one possible x value
        if (problem->XStates->size() == 1) 
        {
            // clear out the entries
            outBelObs.resize(1);
            outBelObs.push_back(0,1.0); 
        } 
        else 
        {
            //problem->getTransitionMatrixX(action, belSt.sval);
	    const SharedPointer<SparseMatrix>  transMatX = problem->XTrans->getMatrix(action, belSt.sval); 
            mult(outBelObs, *belSt.bvec, *transMatX);
	}
    }

  void ROS_SimulationEngine::performActionUnobs(belief_vector& outBelUnobs, int action, const BeliefWithState& belSt, int currObsState) const
  {
        const SharedPointer<SparseMatrix>  transMatY = problem->YTrans->getMatrix(action, belSt.sval, currObsState);
        mult(outBelUnobs, *belSt.bvec, *transMatY);
  }

    void ROS_SimulationEngine::getPossibleObservations(belief_vector& possObs, int action, 	const BeliefWithState& belSt) const
    {
        //const SparseMatrix obsMat = problem->getObservationMatrix(action, belSt.sval);
		const SharedPointer<SparseMatrix>  obsMat = problem->obsProb->getMatrix(action, belSt.sval);
        mult(possObs,  *belSt.bvec, *obsMat);
    }


    double ROS_SimulationEngine::getReward(const BeliefWithState& belst, int action)
    {
        //const SparseMatrix rewMat = problem->getRewardMatrix(belst.sval);
		const SharedPointer<SparseMatrix>  rewMat = problem->rewards->getMatrix(belst.sval);
        return inner_prod_column(*rewMat, action, *belst.bvec);
    }

    string ROS_SimulationEngine::toString()
    {
        std::ostringstream mystrm; 
        mystrm << "action selector: (replaced by Policy) ";
        return mystrm.str();
    }

    void ROS_SimulationEngine::display(belief_vector& b, ostream& s)
    {
        for(unsigned int i = 0; i < b.filled(); i++)
        {
            s << b.data[i].index << " -> " << b.data[i].value << endl;
        }
    }


	//int ROS_SimulationEngine::getPSGObservation()
	//{
	//}
	
	//int ROS_SimulationEngine::getPSGState()
	//{
	//}

	double getDist(string XState, int& rx)
	{
		int px, py,  ry;
		double dist=-1;
		//cout << "XState " << XState << endl;
		
		string temp("strRob");		
		char junk[20];
		if( string::npos !=  XState.find(temp))
		{
			/// XState has str .. skip this state
			dist = -10; /// reached goal			
		}
		else if( string::npos !=  XState.find("strs"))
		{
			/// XState has str .. skip this state
			dist = -1; /// ped reached goal			
		}
		else
		{			
			sscanf(XState.c_str(),"sx%dy%dsR%d%s", &px, &py, &ry, junk);

			dist = fabs(px-rx) + fabs(py-ry);
			
			if(0==dist)
			{
				cout << "XState " << XState << " px:" << px << " rx:" << rx  << endl;
			}
			
			//if( (px==rx) && (py==ry))
				//dist=0;
			//else 
				//dist = py - ry;

		}
		
		return dist;
			
	}

	void ROS_SimulationEngine::runStep(SharedPointer<BeliefWithState>& currBelSt, int currAction, int currObservation, int nextSVal,  SharedPointer<BeliefWithState>& nextBelSt )
	{
		cout << " runStep <<------------------------ " << endl;
		
		cout << "Curr bel state " << endl;
		
		currBelSt->bvec->write(cout); cout << endl;
		cout << "Curr action  " << currAction << endl;
		cout << "Curr Obs  " << currObservation << endl;
		cout << "Next SVal  " << nextSVal << endl;

		//int belSize = currBelSt->bvec->size();
		
		//SparseVector jspv;
		//SharedPointer<MOMDP> momdpProblem = dynamic_pointer_cast<MOMDP> (problem);
		
		//momdpProblem->getJointUnobsStateProbVector(jspv, currBelSt , currAction, nextSVal);
		//problem->getJointUnobsStateProbVector(jspv, currBelSt , currAction, nextSVal);
		//jspv.resize(belSize);
		//cout << "jspv : "; jspv.write(cout); cout << endl;
		
		//nextBelSt = problem->beliefTransition->nextBelief2(currBelSt, currAction, currObservation, nextSVal, jspv);
		//nextBelSt = problem->beliefTransition->nextBelief2(NULL, currAction, currObservation, nextSVal, jspv);
		 
		nextBelSt = problem->beliefTransition->nextBelief(currBelSt, currAction, currObservation, nextSVal);
		
		//nextBelSt->bvec->resize(belSize);
		cout << "Next bel state " << endl;
		nextBelSt->bvec->write(cout); cout << endl;
		
		cout << " runStep ------------------------>> " << endl;
		
		//return 0;
		return;
	}

    
};
