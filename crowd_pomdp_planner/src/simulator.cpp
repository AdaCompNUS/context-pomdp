/*int main()
{return 0;}*/

#include "ped_pomdp.h"
//#include "despotstar.h"
#include "WorldModel.h"
#include "state.h"
#include "Path.h"
#include "solver/despot.h"

using namespace std;

int n_sim = 1;

const double PED_X0 = 35;/// used to generate peds' locations, where x is in (PED_X0, PED_X1), and y is in (PED_Y0, PED_Y1)
const double PED_Y0 = 35;
const double PED_X1 = 42;
const double PED_Y1 = 52;
const int n_peds = 6; // should be smaller than ModelParams::N_PED_IN

class Simulator {
public:
    typedef WorldStateTracker::AgentDistPair AgentDistPair;

    Simulator(): start(40, 40), goal(40, 52.5) /// set the path to be a straight line
    {
        Path p;
        p.push_back(start);
        p.push_back(goal);
        path = p.interpolate();
        worldModel.setPath(path);
    }

    int numPedInArea(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world)
    {
        int num_inside = 0;

        for (int i=0; i<num_of_peds_world; i++)
        {
            if(peds[i].pos.x >= PED_X0 && peds[i].pos.x <= PED_X1 && peds[i].pos.y >= PED_Y0 && peds[i].pos.y <= PED_Y1) num_inside++;
        }

        return num_inside;
    }

    int numPedInCircle(PedStruct peds[ModelParams::N_PED_WORLD], int num_of_peds_world, double car_x, double car_y)
    {
        int num_inside = 0;

        for (int i=0; i<num_of_peds_world; i++)
        {
            if((peds[i].pos.x - car_x)*(peds[i].pos.x - car_x) + (peds[i].pos.y - car_y)*(peds[i].pos.y - car_y) <= ModelParams::LASER_RANGE * ModelParams::LASER_RANGE) num_inside++;
        }

        return num_inside;
    }

    void run() {
        //cout << "====================" << endl;
        WorldStateTracker stateTracker(worldModel);
        WorldBeliefTracker beliefTracker(worldModel, stateTracker);
        PedPomdp * pomdp = new PedPomdp(worldModel);

        //RandomStreams streams(Seeds::Next(Globals::config.num_scenarios), Globals::config.search_depth);
        // pomdp->InitializeParticleLowerBound("smart");
        // pomdp->CreateScenarioLowerBound("SMART");
        // pomdp->CreateParticleUpperBound("SMART");
        // pomdp->CreateScenarioUpperBound("SMART", "SMART");

        ScenarioLowerBound *lower_bound = pomdp->CreateScenarioLowerBound("SMART");
        ScenarioUpperBound *upper_bound = pomdp->CreateScenarioUpperBound("SMART", "SMART");

        Solver * solver = new DESPOT(pomdp, lower_bound, upper_bound);

        //for pomdp planning and print world info
        PomdpState s;
        // for tracking world state
        PomdpStateWorld world_state;

        world_state.car.pos = 0;
        world_state.car.vel = 0;
        world_state.car.dist_travelled = 0;
        world_state.num = n_peds;
        //generate initial n_peds peds
        for(int i=0; i<n_peds; i++) {
            world_state.peds[i] = randomPed();
            world_state.peds[i].id = i;
        }

        int num_of_peds_world = n_peds;

        double total_reward = 0;
        int step = 0;

        for(int step=0; step < 180; step++) {
            cout << "====================" << "step= " << step << endl;
            double reward;
            uint64_t obs;
            stateTracker.updateCar(path[world_state.car.pos]);
            stateTracker.updateVel(world_state.car.vel);
            
            //update the peds in stateTracker
            for(int i=0; i<num_of_peds_world; i++) {
                Pedestrian p(world_state.peds[i].pos.x, world_state.peds[i].pos.y, world_state.peds[i].id);
                stateTracker.updatePed(p);
            }

            /*//add pedestrian in the starting area if fewer than 6 pedestrians inside the area
            while(numPedInArea(world_state.peds, num_of_peds_world)<n_peds && num_of_peds_world < ModelParams::N_PED_WORLD)
            {
                PedStruct new_ped= randomPed();
                new_ped.id = num_of_peds_world;
                world_state.peds[num_of_peds_world]=new_ped;
                
                num_of_peds_world++;

                Pedestrian p(new_ped.pos.x, new_ped.pos.y, new_ped.id);
                stateTracker.updatePed(p); //add the new generated ped into stateTracker.ped_list
            }*/

            //suppose the area that laser can sense is A, then A is a circle consists of the end points of the laser rays. 
            //if fewer than 6 pedestrians inside A, add pedestrian at the edge of A uniformly at random, and set the pedestrian's goal across the circle
            //i.e., if pedestrian wants to go to his goal, he will have to go across the circle.
            while(numPedInCircle(world_state.peds, num_of_peds_world,path[world_state.car.pos].x, path[world_state.car.pos].y)<n_peds && num_of_peds_world < ModelParams::N_PED_WORLD)
            {
                PedStruct new_ped= randomPedAtCircleEdge(path[world_state.car.pos].x, path[world_state.car.pos].y);
                new_ped.id = num_of_peds_world;
                world_state.peds[num_of_peds_world]=new_ped;
                
                num_of_peds_world++;
                world_state.num++;

                Pedestrian p(new_ped.pos.x, new_ped.pos.y, new_ped.id);
                stateTracker.updatePed(p); //add the new generated ped into stateTracker.ped_list
            }



            if(worldModel.isGlobalGoal(world_state.car)) {
                cout << "goal_reached=1" << endl;
                break;
            }

            

            s.car.pos = world_state.car.pos;
            s.car.vel = world_state.car.vel;
            s.car.dist_travelled = world_state.car.dist_travelled;
            s.num = n_peds;
            std::vector<PedDistPair> sorted_peds = stateTracker.getSortedPeds();
            //update s.peds to the nearest n_peds peds
            for(int i=0; i<n_peds; i++) {
                s.peds[i] = world_state.peds[sorted_peds[i].second.id];
            }

            cout << "state=[[" << endl;
            pomdp->PrintState(s);
            cout << "]]" << endl;
            
            int collision_peds_id=-1;
            if( world_state.car.vel > 0.001 && worldModel.inCollision(world_state,collision_peds_id) ) {
                cout << "collision=1: " << collision_peds_id<<endl;
            }
            else if(worldModel.inCollision(s,collision_peds_id)) {
                //cout << "close=1: " << collision_peds_id<<endl;
            }

            beliefTracker.update();
            ////vector<PomdpState> samples = beliefTracker.sample(Globals::config.num_scenarios);
            vector<PomdpState> samples = beliefTracker.sample(2000);//samples are used to construct particle belief. num_scenarios is the number of scenarios sampled from particles belief to construct despot
            vector<State*> particles = pomdp->ConstructParticles(samples);
            ParticleBelief* pb = new ParticleBelief(particles, pomdp);
            solver->belief(pb);
            Globals::config.silence = false;
            int act = solver->Search().action;
            cout << "act= " << act << endl;
            bool terminate = pomdp->Step(world_state,
                    Random::RANDOM.NextDouble(),
                    act, reward, obs);
            cout << "obs= " << endl;
            cout << "reward= " << reward << endl;
            total_reward += reward * Globals::Discount(step);
            if(terminate) {
                cout << "terminate=1" << endl;
                break;
            }
        }
        cout << "final_state=[[" << endl;
        pomdp->PrintState(s);
        cout << "]]" << endl;
        cout << "total_reward= " << total_reward << endl;
       // cout << "====================" << endl;
    }

    PedStruct randomPed() {
        int n_goals = worldModel.goals.size();
        int goal = Random::RANDOM.NextInt(n_goals);
        double x = Random::RANDOM.NextDouble(PED_X0, PED_X1);
        double y = Random::RANDOM.NextDouble(PED_Y0, PED_Y1);
        if(goal == n_goals-1) {
            // stop intention
            while(path.mindist(COORD(x, y)) < 1.0) {
                // dont spawn on the path
                x = Random::RANDOM.NextDouble(PED_X0, PED_X1);
                y = Random::RANDOM.NextDouble(PED_Y0, PED_Y1);
            }
        }
        int id = 0;
        return PedStruct(COORD(x, y), goal, id);
    }

    PedStruct randomPedAtCircleEdge(double car_x, double car_y) {
        int n_goals = worldModel.goals.size();
        int goal = Random::RANDOM.NextInt(n_goals);
        double x, y;
        double angle;

        angle = Random::RANDOM.NextDouble(0, M_PI/2);

        if(goal==3) {
            x = car_x - ModelParams::LASER_RANGE * cos(angle);
            y = car_y - ModelParams::LASER_RANGE * sin(angle);
        } else if(goal == 4){
            x = car_x + ModelParams::LASER_RANGE * cos(angle);
            y = car_y - ModelParams::LASER_RANGE * sin(angle);
        } else if(goal == 2 || goal == 1){
            x = car_x + ModelParams::LASER_RANGE * cos(angle);
            y = car_y + ModelParams::LASER_RANGE * sin(angle);
        } else if(goal == 0 || goal == 5){
            x = car_x - ModelParams::LASER_RANGE * cos(angle);
            y = car_y + ModelParams::LASER_RANGE * sin(angle);
        } else{
            angle = Random::RANDOM.NextDouble(-M_PI, M_PI);
            x = car_x + ModelParams::LASER_RANGE * cos(angle);
            y = car_y + ModelParams::LASER_RANGE * sin(angle);
            if(goal == n_goals-1) {
                while(path.mindist(COORD(x, y)) < 1.0) {
                    // dont spawn on the path
                    angle = Random::RANDOM.NextDouble(-M_PI, M_PI);
                    x = car_x + ModelParams::LASER_RANGE * cos(angle);
                    y = car_y + ModelParams::LASER_RANGE * sin(angle);
                }
            }
        }
        int id = 0;
        return PedStruct(COORD(x, y), goal, id);
    }

    void generateFixedPed(PomdpState &s) {
        
        s.peds[0] = PedStruct(COORD(38.1984, 50.6322), 5, 0);

        s.peds[1] = PedStruct(COORD(35.5695, 46.2163), 4, 1);

        s.peds[2] = PedStruct(COORD(41.1636, 49.6807), 4, 2);

        s.peds[3] = PedStruct(COORD(35.1755, 41.4558), 4, 3);

        s.peds[4] = PedStruct(COORD(37.9329, 35.6085), 3, 4);

        s.peds[5] = PedStruct(COORD(41.0874, 49.6448), 5, 5);
    }

    COORD start, goal;

    Path path;
    WorldModel worldModel;

};

int main(int argc, char** argv) {
    if (argc >= 2) ModelParams::CRASH_PENALTY = -atof(argv[1]);
    else  ModelParams::CRASH_PENALTY = -1000;
    
    Globals::config.num_scenarios=300;
    Globals::config.time_per_move = (1.0/ModelParams::control_freq) * 0.9;
    // TODO the seed should be initialized properly so that
    // different process as well as process on different machines
    // all get different seeds
    Seeds::root_seed(get_time_second());
    // Global random generator
    double seed = Seeds::Next();
    Random::RANDOM = Random(seed);
    cerr << "Initialized global random generator with seed " << seed << endl;


    if (argc >= 3) n_sim = atoi(argv[2]);
    Simulator sim;
    for(int i=0; i<n_sim; i++){
        cout<<"++++++++++++++++++++++ ROUND "<<i<<" ++++++++++++++++++++"<<endl;
        sim.run();
    }
}


/*#include "ped_pomdp.h"
#include "despotstar.h"
#include "WorldModel.h"
#include "state.h"
#include "Path.h"

using namespace std;

const int n_sim = 5000;

const double PED_X0 = 35;
const double PED_Y0 = 35;
const double PED_X1 = 42;
const double PED_Y1 = 47;
const int n_peds = 6; // should be smaller than ModelParams::N_PED_IN

class Simulator {
public:
    Simulator(): start(40, 40), goal(40, 45)
    {
        Path p;
        p.push_back(start);
        p.push_back(goal);
        path = p.interpolate();
        worldModel.setPath(path);
    }

    void run() {
        cout << "====================" << endl;
        WorldStateTracker stateTracker(worldModel);
        WorldBeliefTracker beliefTracker(worldModel, stateTracker);
        PedPomdp pomdp(worldModel);

        RandomStreams streams(Seeds::Next(Globals::config.n_particles), Globals::config.search_depth);
        pomdp.InitializeParticleLowerBound("smart");
        pomdp.InitializeScenarioLowerBound("smart", streams);
        pomdp.InitializeParticleUpperBound("smart", streams);
        pomdp.InitializeScenarioUpperBound("smart", streams);
        DESPOTSTAR solver(&pomdp, NULL, streams);

        // init state
        PomdpState s;
        s.car.pos = 0;
        s.car.vel = 0;
        s.car.dist_travelled = 0;
        s.num = n_peds;
        for(int i=0; i<n_peds; i++) {
            s.peds[i] = randomPed();
            s.peds[i].id = i;
        }

        double total_reward = 0;
        int step = 0;

        for(int step=0; step < 50; step++) {
            cout << "step=" << step << endl;
            double reward;
            uint64_t obs;
            stateTracker.updateCar(path[s.car.pos]);
            stateTracker.updateVel(s.car.vel);
            for(int i=0; i<n_peds; i++) {
                Pedestrian p(s.peds[i].pos.x, s.peds[i].pos.y, s.peds[i].id);
                stateTracker.updatePed(p);
            }

            if(worldModel.isGlobalGoal(s.car)) {
                cout << "goal_reached=1" << endl;
                break;
            }

            cout << "state=[[" << endl;
            pomdp.PrintState(s);
            cout << "]]" << endl;

            if(worldModel.inCollision(s)) {
                cout << "collision=1" << endl;
            }

            beliefTracker.update();
            vector<PomdpState> samples = beliefTracker.sample(Globals::config.n_particles);
		    vector<State*> particles = pomdp.ConstructParticles(samples);
            ParticleBelief* pb = new ParticleBelief(particles, &pomdp);
            solver.belief(pb);
            int act = solver.Search();
            cout << "act=" << act << endl;
            bool terminate = pomdp.Step(s,
                    Random::RANDOM.NextDouble(),
                    act, reward, obs);
            cout << "obs=" << obs << endl;
            cout << "reward=" << reward << endl;
            total_reward += reward * Globals::Discount(step);
            if(terminate) {
                cout << "terminate=1" << endl;
                break;
            }
        }
        cout << "final_state=[[" << endl;
        pomdp.PrintState(s);
        cout << "]]" << endl;
        cout << "total_reward=" << total_reward << endl;
        cout << "====================" << endl;
    }

    PedStruct randomPed() {
        int n_goals = worldModel.goals.size();
        int goal = Random::RANDOM.NextInt(n_goals);
        double x = Random::RANDOM.NextDouble(PED_X0, PED_X1);
        double y = Random::RANDOM.NextDouble(PED_Y0, PED_Y1);
        if(goal == n_goals-1) {
            // stop intention
            while(path.mindist(COORD(x, y)) < 1.0) {
                // dont spawn on the path
                x = Random::RANDOM.NextDouble(PED_X0, PED_X1);
                y = Random::RANDOM.NextDouble(PED_Y0, PED_Y1);
            }
        }
        int id = 0;
        return PedStruct(COORD(x, y), goal, id);
    }

    COORD start, goal;

    Path path;
    WorldModel worldModel;

};

int main(int argc, char** argv) {
    ModelParams::CRASH_PENALTY = -atof(argv[1]);
    
    Globals::config.n_particles=300;
    Globals::config.time_per_move = (1.0/ModelParams::control_freq) * 0.9;
    // TODO the seed should be initialized properly so that
    // different process as well as process on different machines
    // all get different seeds
    Seeds::root_seed(get_time_second());
    // Global random generator
    double seed = Seeds::Next();
    Random::RANDOM = Random(seed);
    cerr << "Initialized global random generator with seed " << seed << endl;


    Simulator sim;
    for(int i=0; i<n_sim; i++)
        sim.run();
}
*/