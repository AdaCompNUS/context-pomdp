#include "Pedestrian.h"
#include <string>
#include <math.h>
#include <fstream>
#include "Bernoulli.h"

using namespace std;

Pedestrian_State*teststate1;
Pedestrian_State*teststate2;
const int CRUSH=-3000;

/*
Pedestrian::Pedestrian(int size):Size(size)
{
	Discount=0.95;
	teststate1=MemoryPool.Allocate();
	teststate2=MemoryPool.Allocate();
	dummyState = MemoryPool.Allocate();
	InitModel();
}
*/

Pedestrian::Pedestrian(string filename) 
{
	Discount=0.95;
	
	dummyState = MemoryPool.Allocate();
	teststate1 = MemoryPool.Allocate();
	teststate2 = MemoryPool.Allocate();

	Pedestrian_State* teststate = MemoryPool.Allocate();
	teststate->RobPos.X=2;
	teststate->RobPos.Y=0;
	teststate->PedPos.X=0;
	teststate->PedPos.Y=4;
	teststate->Vel=0;

	ifstream fin(filename, ios::in);
	fin >> Size;
	fin >> teststate->Goal;

	InitModel();

	startState = StateToN(teststate);
	cout << "Initialized " << Size << " " << teststate->Goal << " " << startState << endl; cout.flush();
}

void Pedestrian::printState(State state)
{
	NToState(state, dummyState);
	cout <<"Rob Pos : "<<dummyState->RobPos.X<<" "<<dummyState->RobPos.Y<<endl;
	cout <<"Ped Pos : "<<dummyState->PedPos.X<<" "<<dummyState->PedPos.Y<<endl;
	cout <<"Vel: " << dummyState->Vel << endl;
}

void Pedestrian::InitModel()
{
	int i1,i2,j1,j2;
	for(i1=0;i1<5;i1++)
	{
		for(j1=0;j1<Size;j1++)
		{
			walk_dirs[i1][j1][0]=walk_dirs[i1][j1][1]=4;
		}
	}
}

State Pedestrian::step(State state, double rNum, int action, double& reward, uint64_t& obs) {
	NToState(state, dummyState);
	Pedestrian_State teststate  = *dummyState;
	reward=0;
	obs = 150;
	State nextState;
	
	if (action==ACT_ACC)	//accelerate
	{	
		if(teststate.Vel!=2)	teststate.Vel++;	//speed limit
	}
	else if(action==ACT_DEC)
	{
		if(teststate.Vel!=0)	teststate.Vel--;
		//if(teststate.Vel>0)	teststate.Vel=0;
	}

	//crush detect	
	if(abs(teststate.RobPos.X-teststate.PedPos.X)<=1&&teststate.RobPos.Y==teststate.PedPos.Y)	
	{
		reward=CRUSH;
		nextState = numStates()-1;
		return nextState;

	}
	if(abs(teststate.RobPos.X-teststate.PedPos.X)<=1&&teststate.RobPos.Y==teststate.PedPos.Y-1&&teststate.Vel>0)
	{
		reward=CRUSH;
		nextState = numStates()-1;
		return nextState;
	}
	
	//pos transition
	if(teststate.RobPos.Y>=teststate.PedPos.Y)	//overtake the pedestrian,end
	{
		reward=1000;
		nextState = numStates()-1;
		return nextState;
	}
	else if(teststate.RobPos.Y>=Size||teststate.PedPos.Y>=Size)
	{
		reward=0;
		nextState = numStates()-1;
		return nextState;
	}
	else
	{
		teststate.RobPos.Y=teststate.RobPos.Y+teststate.Vel;
	}
	
	int &pedX=teststate.PedPos.X;
	int &pedY=teststate.PedPos.Y;

	//cout << "Before: " << pedX << " " << pedY << " " << teststate.Goal << endl;
	bernoulli brv(rNum);

	if(brv.below(0.1)==true)	{
		//cout << "No Move" << endl;
		//no move with prob 1/3
	} else {	
		int walk_dir;
		double rate=0.7;
		double noise=0.06;
		if(teststate.Goal==0)
		{
			walk_dir=trans[teststate.PedPos.X][teststate.PedPos.Y][0][0];
			if(walk_dir==3)		//two directions the same
			{
				if(brv.below(rate) && pedX>0)
				{
					pedX--;	
				}
				else
				{
					pedY--;
				}
			}
			else if(walk_dir==1&&pedX>0)	//left-right
			{
				pedX--;
			}			
			else	if(walk_dir==2&&pedY>0)		//up down
			{
				pedY--;
			}
			else			//exception
			{
				
			}
		}
		else if(teststate.Goal==1)
		{
			walk_dir=walk_dirs[teststate.PedPos.X][teststate.PedPos.Y][0];
			//cout << "walk_dir = " << walk_dir << endl;
			//if(pedY==7)
			//ROS_INFO("noise!!! %d %d %d",action,pedX,pedY);
			if(walk_dir==4)		//two directions the same
			{
				double val1 = brv.next(), val2 = brv.next();
				//cout << val1 << " " << rate << endl;
				//cout << val2 << " " << noise << endl;

				if(val1 < rate && pedX>0)
				{
					pedX--;	
				}
				else if(val2 < noise && pedX<4)
				{
					//ROS_INFO("noise!!!!!!!!!");
					pedX++;
				}
				else if(pedY<Size-1)
				{
					pedY++;
				}
			}
			else if(walk_dir==1&&pedX>0)	//left
			{
				pedX--;
			}			
			else	if(walk_dir==2&&pedX<4)		//right
			{
				pedX++;
			}
			else 	if(walk_dir==3&&pedY<Size-1)
			{
				pedY++;
			}
			else			//exception
			{
				
			}
		}
		else if(teststate.Goal==2)
		{
			walk_dir=trans[teststate.PedPos.X][teststate.PedPos.Y][4][0];
			if(walk_dir==3)		//two directions the same
			{
				if(brv.below(rate)==true)
				{
					pedX++;	
				}
				else
				{
					pedY--;
				}
			}
			else if(walk_dir==1)	//left-right
			{
				pedX++;
			}			
			else	if(walk_dir==2)		//up down
			{
				pedY--;
			}
			else			//exception
			{
				
			}
		}
		else
		{
			walk_dir=walk_dirs[teststate.PedPos.X][teststate.PedPos.Y][1];
			if(walk_dir==4)		//two directions the same
			{
				if(brv.below(rate)==true &&pedX<4)
				{
					pedX++;	
				}
				else if(brv.below(noise)==true&&pedX>0)
				{
					pedX--;
				}
				else if(pedY<Size-1)
				{
					pedY++;
				}
			}
			else if(walk_dir==1&&pedX>0)	//left
			{
				pedX--;
			}		
	
			else	if(walk_dir==2&&pedX<4)		//right
			{
				pedX++;
			}
			else 	if(walk_dir==3&&pedY<Size-1)		//up
			{
				pedY++;
			}
			else			//exception
			{

			}
		}
	}

	//cout << "After: " << pedX << " " << pedY << endl;
	obs =teststate.PedPos.X*30+teststate.PedPos.Y;
	nextState = StateToN(&teststate);
	reward=0;
	return nextState;
}

double Pedestrian::obsProb(uint64_t obs, State s, int action) {
	NToState(s, dummyState);
	return (s != numStates()-1 && obs == dummyState->PedPos.X*30 + dummyState->PedPos.Y) || (s == numStates()-1 && obs == terminalObs());
}

double Pedestrian::fringeUpperBound(State s)  {
	NToState(s, dummyState);
	int d = (int)(fabs(dummyState->PedPos.Y - dummyState->RobPos.Y)) / 2;
	return 1000 * pow(Discount, d);
}

double Pedestrian::fringeLowerBound(vector<Particle>& particles) {
	return CRUSH;
}
/*
double Pedestrian::fringeLowerBound(State s) {
	NToState(s, dummyState);
	int d = (int)(fabs(dummyState->PedPos.Y - dummyState->RobPos.Y)) / 2;
	return CRUSH * pow(Discount, d);
}
*/

State Pedestrian::getStartState() {
	return startState;
}

UMAP<State, double> Pedestrian::initBelief() {
	UMAP<State, double> belief;
	for(int s=0; s<numStates(); s++)
		belief[s] = 0;
	NToState(startState, dummyState);
	dummyState->Goal = 1;
	belief[StateToN(dummyState)] = 0.5;
	dummyState->Goal = 3;
	belief[StateToN(dummyState)] = 0.5;
	return belief;
}

int Pedestrian::numStates() const
{
	return 5*Size*Size*3*2 + 1;	
}
int Pedestrian::StateToN(Pedestrian_State* teststate) const
{
	int add_goal=(5*Size*Size*3)*(teststate->Goal==3);
	int add_vel=(5*Size*Size)*teststate->Vel;
	int add_Y1=(5*Size)*teststate->RobPos.Y;
	int add_Y2=5*teststate->PedPos.Y;
	int add_X=teststate->PedPos.X;
	int N=add_X+add_Y2+add_Y1+add_vel+add_goal;
	return N;
}

void Pedestrian::NToState(int n,Pedestrian_State*teststate) const
{
	//5 lanes
	int N=n;
	//Pedestrian_State*teststate=MemoryPool.Allocate();
	teststate->Goal=(N/(5*Size*Size*3)==0?1:3);
	N=N%(5*Size*Size*3);
	teststate->Vel=N/(5*Size*Size);
	teststate->RobPos.X = 2;
	N=N%(5*Size*Size);
	teststate->RobPos.Y=N/(5*Size);
	N=N%(5*Size);
	teststate->PedPos.Y=N/5;
	N=N%5;
	teststate->PedPos.X=N;
	//return teststate;
}
