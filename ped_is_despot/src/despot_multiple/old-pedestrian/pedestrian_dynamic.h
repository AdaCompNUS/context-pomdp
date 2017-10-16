#ifndef PEDESTRIAN_DYNAMIC_H
#define PEDESTRIAN_DYNAMIC_H

#include "simulator.h"
#include "coord.h"
#include "grid.h"

class PEDESTRIAN_DYNAMIC_STATE:public STATE
{
public:
    COORD RobPos;
    COORD PedPos;
    int   Vel;
    int   Goal;
};

class PEDESTRIAN_DYNAMIC:public SIMULATOR
{
public:
    PEDESTRIAN_DYNAMIC(int size);
    //virtual void UpdateModel(int);
    virtual STATE* Copy(const STATE& state) const;
    virtual void Validate(const STATE& state) const;
    virtual STATE* CreateStartState() const;
    virtual void FreeState(STATE* state) const;
    virtual bool Step(STATE& state, int action,
        int& observation, double& reward) const;
    int GetNumOfStates();
    int StateToN(STATE* state);
    void NToState(int n,PEDESTRIAN_DYNAMIC_STATE*);
    double TransFn(int s1,int a,int s2);
    double GetReward(int s);
    double MaxQ(int s);
	int QMDP_SelectAction(STATE*state);
    double QMDP(STATE*state);
    double QMDP(int state);
    void Init_QMDP();
    void display_QMDP();
    void writeQMDP();
    void loadQMDP();
    virtual void InitModel();
	virtual void CopyModel(int **);
    static void UpdateModel(int );
    double Heuristic(STATE& state);
    void DisplayBeliefs(const BELIEF_STATE&, std::ostream&)const;
    void DisplayState(const STATE&, std::ostream&) const;
	VNODE* MapBelief(VNODE*vn,VNODE*r,const HISTORY &h,int);
	VNODE* Find(VNODE*v1,const HISTORY &h);
	VNODE* Find_old(VNODE*v1,const HISTORY &h);

	bool Same(const BELIEF_STATE& beliefState1,const BELIEF_STATE& beliefState2,double & distance);

	std::vector<VNODE*> vnode_old;
	std::vector<VNODE*> vnode_list;
	std::vector<HISTORY> history_list;
	int FindIndex(VNODE*v)
	{
		for(int i=0;i<vnode_list.size();i++)
		{
			if(vnode_list[i]==v) 	return i;
		}
		return -1;
	}

	static int map[4][10];
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
    mutable MEMORY_POOL<PEDESTRIAN_DYNAMIC_STATE> MemoryPool;
};

class PEDESTRIAN_DYNAMIC_REAL_STATE:public STATE
{
public:
    COORD RobPos;
    COORD PedPos;
    int   Vel;
    int   Goal;
};

class PEDESTRIAN_DYNAMIC_REAL:public SIMULATOR
{
public:
    PEDESTRIAN_DYNAMIC_REAL(int size);
    //virtual void UpdateModel(int);
    virtual STATE* Copy(const STATE& state) const;
    virtual void Validate(const STATE& state) const;
    virtual STATE* CreateStartState() const;
    virtual void FreeState(STATE* state) const;
    virtual bool Step(STATE& state, int action,
        int& observation, double& reward) const;
    int GetNumOfStates();
    int StateToN(STATE* state);
    void NToState(int n,PEDESTRIAN_DYNAMIC_REAL_STATE*);
    double TransFn(int s1,int a,int s2);
    double GetReward(int s);
    double MaxQ(int s);
	int QMDP_SelectAction(STATE*state);
    double QMDP(STATE*state);
    double QMDP(int state);
    void Init_QMDP();
    void display_QMDP();
    void writeQMDP();
    void loadQMDP();
    virtual void InitModel() const;
    static void UpdateModel(int );
    double Heuristic(STATE& state);
    void DisplayBeliefs(const BELIEF_STATE&, std::ostream&)const;
    void DisplayState(const STATE&, std::ostream&) const;
	VNODE* MapBelief(VNODE*vn,VNODE*r,const HISTORY &h,int);
	VNODE* Find(VNODE*v1,const HISTORY &h);
	VNODE* Find_old(VNODE*v1,const HISTORY &h);
	bool Same(const BELIEF_STATE& beliefState1,const BELIEF_STATE& beliefState2,double & distance);

	std::vector<VNODE*> vnode_old;
	std::vector<VNODE*> vnode_list;
	std::vector<HISTORY> history_list;
	int FindIndex(VNODE*v)
	{
		for(int i=0;i<vnode_list.size();i++)
		{
			if(vnode_list[i]==v) 	return i;
		}
		return -1;
	}
	virtual void DisplayModel() const
	{
		for(int i=0;i<4;i++)
		{
			for(int j=0;j<10;j++)
				std::cout<<map[i][j];
			std::cout<<std::endl;
		}
	}
	static int map[4][10];
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
    mutable MEMORY_POOL<PEDESTRIAN_DYNAMIC_REAL_STATE> MemoryPool;
};

#endif
