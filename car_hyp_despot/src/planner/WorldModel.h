#pragma once
#include"state.h"
#include"Path.h"
#include <RVO.h>
#include "debug_util.h"
#include <unordered_map>

using namespace despot;

struct AgentBelief {
	int id;
    COORD pos;
    double speed;
    COORD vel;
    double heading_dir;
    AgentType type;
    bool reset;
    // std::vector<double> prob_goals;
    bool cross_dir;

    std::vector<Path> goal_paths;
    // this can be shared for all types of intention
    std::vector<std::vector<double>> prob_modes_goals;

    std::vector<COORD> bb;

    int sample_goal() const;
    int maxlikely_intention() const;

    void sample_goal_mode(int& goal, int& mode) const;

    void reset_belief();
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

	void PedStep(AgentStruct &ped, Random& random);
	void PedStep(AgentStruct &ped, double& random);

    double ISPedStep(CarStruct &car, AgentStruct &ped, Random& random);//importance sampling PedStep
    void RVO2AgentStep(AgentStruct peds[], Random& random, int num_ped); //no interaction between car and pedestrian
    void RVO2AgentStep(AgentStruct peds[], Random& random, int num_ped, CarStruct car); //pedestrian also need to consider car when moving
    void RVO2AgentStep(AgentStruct peds[], double& random, int num_ped); //no interaction between car and pedestrian
    void RVO2AgentStep(AgentStruct peds[], double& random, int num_ped, CarStruct car); //pedestrian also need to consider car when moving
    void RVO2AgentStep(PomdpStateWorld& state, Random& random);
    void PedStepGoal(AgentStruct& ped, int step=1);
    void PedStepCurVel(AgentStruct& ped, int step=1);
    void PedStepPath(AgentStruct& agent, int step=1);

    COORD GetGoalPos(const AgentStruct& ped, int intention_id=-1);
    COORD GetGoalPos(const AgentBelief& ped, int intention_id);
    COORD GetGoalPos(const Agent& ped, int intention_id);

    COORD GetGoalPosFromPaths(int agent_id, int intention_id, int pos_along_path, 
        const COORD& agent_pos, AgentType type, bool agent_cross_dir);

    int GetNumIntentions(const AgentStruct& ped);
    int GetNumIntentions(const Agent& ped);
    int GetNumIntentions(int ped_id);

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

    double agentMoveProb(COORD p0, const Agent& p1, int goal_id);
    double agentMoveProb(COORD prev, const Agent& curr, int goal_id, int ped_mode);

    void setPath(Path path);
    void updatePedBelief(AgentBelief& b, const Agent& curr_ped);
    AgentBelief initPedBelief(const Agent& ped);

    inline int GetThreadID(){return Globals::MapThread(this_thread::get_id());}
    void InitRVO();

    void cal_bb_extents(AgentBelief& bb, AgentStruct& agent);
    void cal_bb_extents(AgentStruct& agent, std::vector<COORD>& bb, double heading_dir);

	Path path;
    COORD car_goal;
    std::vector<COORD> goals;
    double freq;
    const double in_front_angle_cos;
    std::vector<RVO::RVOSimulator*> traffic_agent_sim_;

    const std::string goal_mode = "cur_vel"; // "path", "goal"

    void rasie_unsupported_goal_mode(std::string function){
        std::cout << function << ": unsupported goal mode " 
            << goal_mode << std::endl;
        raise(SIGABRT);
    }

    std::vector<COORD> obstacles;

    std::unordered_map<int, int> id_map_num_paths;
    std::unordered_map<int, std::vector<Path>> id_map_paths;

    int NumPaths(int agent_id){ 
        auto it = id_map_num_paths.find(agent_id);
        if (it == id_map_num_paths.end()){
            std::cout << "agent id " << agent_id <<
             " not found in id_map_num_paths" << std::endl;
            raise(SIGABRT);
        }
        return it->second;
    }

    std::vector<Path>& PathCandidates(int agent_id){
        auto it = id_map_paths.find(agent_id);
        if (it == id_map_paths.end()){
            std::cout << __FILE__ << ": agent id " << agent_id <<
             " not found in id_map_paths" << std::endl;

            print_path_map();

            std::vector<Path> ps;

            id_map_paths[agent_id] = ps;
            it = id_map_paths.find(agent_id);
            // raise(SIGABRT);
        }
        return it->second;
    }

    void print_path_map(){
        return;

        cout << "id_map_paths: ";
        for(auto it = id_map_paths.begin(); it != id_map_paths.end(); ++it) {
            cout << " (" << it->first << ", l_" << it->second.size() << ")";
        }
        cout << endl;       
    }


public:
    std::string goal_file_name_;
    void InitPedGoals();
    void AddObstacle(std::vector<RVO::Vector2> obstacles);
    bool CheckCarWithObstacles(const CarStruct& car, int flag);// 0 in search, 1 real check
    bool CheckCarWithObsLine(const CarStruct& car, COORD obs_last_point, COORD obs_first_point, int flag);// 0 in search, 1 real check

    bool CheckCarWithVehicle(const CarStruct& car, const AgentStruct& veh, int flag);
    /// lets drive
    std::map<int, vector<COORD>> ped_mean_dirs;
    COORD DistractedPedMeanDir(AgentStruct& ped, int goal_id);
    COORD AttentivePedMeanDir(int ped_id, int goal_id);

    void add_car_agent(int num_peds, CarStruct& car);
    void add_veh_agent(AgentBelief& veh);

    void PrepareAttentiveAgentMeanDirs(std::map<int, AgentBelief> peds, CarStruct& car);

    void PrintMeanDirs(std::map<int, AgentBelief> old_peds, map<int, const Agent*>& curr_peds);

    void EnsureMeanDirExist(int ped_id);

public:
  template<typename T>
  double GetSteerToPath(const T& state) const {
    COORD car_goal= path[path.forward(path.nearest(state.car.pos), 5.0)]; // target at 3 meters ahead along the path
    COORD dir = car_goal - state.car.pos;
    double theta = atan2(dir.y, dir.x);

    if(theta<0)
      theta+=2*M_PI;

    theta = theta - state.car.heading_dir;

    while (theta > 2*M_PI)
      theta -= 2*M_PI;
    while (theta < 0)
      theta += 2*M_PI;

    float guide_steer = 0;
    if(theta < M_PI)
      guide_steer = min(theta, ModelParams::MaxSteerAngle); // Dvc_ModelParams::MaxSteerAngle//CCW
    else if(theta>M_PI)
      guide_steer = max((theta - 2*M_PI), -ModelParams::MaxSteerAngle); //-Dvc_ModelParams::MaxSteerAngle;//CW
    else
      guide_steer = 0;//ahead

    return guide_steer;
  }

};

class WorldStateTracker {
public:
    typedef pair<float, Agent*> AgentDistPair;

    WorldStateTracker(WorldModel& _model): model(_model) {
    	car_heading_dir = 0;
    	carvel = 0;
    }

    void updatePed(const Pedestrian& ped, bool doPrint = false);
    void updateVeh(const Vehicle& veh, bool doPrint = false); // this is for exo-vehicles
    
    void trackVel(Agent& des, const Agent& src, bool&, bool);
    void tracPos(Agent& des, const Agent& src, bool);
    void tracIntention(Agent& des, const Agent& src, bool);
    void tracBoundingBox(Vehicle& des, const Vehicle& src, bool);
    void tracCrossDirs(Pedestrian& des, const Pedestrian& src, bool);

    void updateCar(const CarStruct car);
    void updateVel(double vel);

    void cleanPed();
    void cleanVeh();
    void cleanAgents();

    void removeAgents();

    double timestamp() {
        static double starttime=get_time_second();
        return get_time_second()-starttime;
    }

    template<typename T>
    bool AgentIsAlive(T& agent, vector<T>& agent_list_new) {
        bool insert=true;
        double w1,h1;
        w1 = agent.w;
        h1 = agent.h;
        for(const auto& p: agent_list_new) {
            double w2,h2;
            w2=p.w;
            h2=p.h;
            if (abs(w1-w2)<=0.1&&abs(h1-h2)<=0.1) {
                insert=false;
                break;
            }
        }
        if (latest_time_stamp - agent.time_stamp > 0.2){ // agent disappeared for 1 second
            cout << "agent "<< agent.id << " disappeared for too long (>0.2s)." << endl;
            insert=false;
        }
        else
            ; // fprintf(stderr, "agent %d alive, latest_time_stamp = %f, agent time_stamp = %f \n", 
                // agent.id, latest_time_stamp, agent.time_stamp);
            
        return insert;
    }

    void setPomdpCar(CarStruct& car);
    void setPomdpPed(AgentStruct& agent, const Agent& src);

    bool emergency();

    std::vector<AgentDistPair> getSortedAgents(bool doPrint = false);

    PomdpState getPomdpState();
    PomdpStateWorld getPomdpWorldState();

    void text(const vector<AgentDistPair>& sorted_agents) const;  
    void text(const vector<Pedestrian>& tracked_peds) const;
    void text(const vector<Vehicle>& tracked_vehs) const;
    
    // Car state
    COORD carpos;
    double carvel;
    double car_heading_dir;

    //Ped states
    std::vector<Pedestrian> ped_list;
    std::vector<Vehicle> veh_list;

    double latest_time_stamp;

    WorldModel& model;
};

class WorldBeliefTracker {
public:
    WorldBeliefTracker(WorldModel& _model, WorldStateTracker& _stateTracker): model(_model), stateTracker(_stateTracker) {}

    void update();
    PomdpState sample(bool predict = false);
    vector<PomdpState> sample(int num, bool predict = false);
    vector<AgentStruct> predictAgents();
    PomdpState predictPedsCurVel(PomdpState*, double acc, double steering);

    WorldModel& model;
    WorldStateTracker& stateTracker;
    CarStruct car;
    std::map<int, AgentBelief> agent_beliefs;
	std::vector<AgentBelief*> sorted_beliefs;

    void PrintState(const State& s, ostream& out = cout) const;
    void printBelief() const;

    PomdpState text() const;

    void text(const std::map<int, AgentBelief>&) const;

public:
    double cur_time_stamp;

    double cur_acc;
    double cur_steering;
};

enum PED_MODES {
	AGENT_ATT = 0,
	AGENT_DIS = 1,
    NUM_AGENT_TYPES = 2,
};

double cap_angle(double angle);
int ClosestInt(double v);
int FloorIntRobust(double v);
