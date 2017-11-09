#include <ped_path_predictor/ped_path_predictor.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

/*
Pedestrian Path Predictor Class
*/
PedPathPredictor::PedPathPredictor(){
	world_his = new WorldStateHistroy(nh);

	sim = new RVO::RVOSimulator();
	// Specify global time step of the simulation.
    sim->setTimeStep(0.33f);    
    // Specify default parameters for agents that are subsequently added.
    //sim->setAgentDefaults(5.0f, 8, 10.0f, 5.0f, 0.5f, 2.0f);
    sim->setAgentDefaults(1.5f, 1, 3.0f, 6.0f, 0.12f, 3.0f);

    //global_frame_id = ModelParams::rosns + "/map";
    global_frame_id = "/map";

    
    ped_prediction_pub = nh.advertise<sensor_msgs::PointCloud>("ped_prediction", 1);

    AddObstacle();

    timer = nh.createTimer(ros::Duration(1.0/ModelParams::control_freq), &PedPathPredictor::Predict, this);
}

 void PedPathPredictor::AddObstacle(){

/*    ifstream file;
    file.open("obstacles.txt", std::ifstream::in);

    if(file.fail()){
        cout<<"open obstacle.txt failed"<<endl;
        return;
    }

    std::vector<RVO::Vector2> obstacles[100];

    std::string line;
    int obst_num = 0;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        
        double x;
        double y;
        while (iss >> x >>y){
            cout << x <<" "<< y<<endl;
            obstacles[obst_num].push_back(RVO::Vector2(x, y));
        }

        sim->addObstacle(obstacles[obst_num]);
        obst_num++;
        if(obst_num > 99) break;
    }
    sim->processObstacles();

    file.close();*/

        std::vector<RVO::Vector2> obstacle[12];

        obstacle[0].push_back(RVO::Vector2(-222.55,-137.84));
        obstacle[0].push_back(RVO::Vector2(-203.23,-138.35));
        obstacle[0].push_back(RVO::Vector2(-202.49,-127));
        obstacle[0].push_back(RVO::Vector2(-222.33,-127));

        obstacle[1].push_back(RVO::Vector2(-194.3,-137.87));
        obstacle[1].push_back(RVO::Vector2(-181.8,-138));
        obstacle[1].push_back(RVO::Vector2(-181.5,-127));
        obstacle[1].push_back(RVO::Vector2(-194.3,-127));

        obstacle[2].push_back(RVO::Vector2(-178.5,-137.66));
        obstacle[2].push_back(RVO::Vector2(-164.95,-137.66));
        obstacle[2].push_back(RVO::Vector2(-164.95,-127));
        obstacle[2].push_back(RVO::Vector2(-178.5,-127));

        obstacle[3].push_back(RVO::Vector2(-166.65,-148.05));
        obstacle[3].push_back(RVO::Vector2(-164,-148.05));
        obstacle[3].push_back(RVO::Vector2(-164,-138));
        obstacle[3].push_back(RVO::Vector2(-166.65,-138));

        obstacle[4].push_back(RVO::Vector2(-172.06,-156));
        obstacle[4].push_back(RVO::Vector2(-166,-156));
        obstacle[4].push_back(RVO::Vector2(-166,-148.25));
        obstacle[4].push_back(RVO::Vector2(-172.06,-148.25));

        obstacle[5].push_back(RVO::Vector2(-197.13,-156));
        obstacle[5].push_back(RVO::Vector2(-181.14,-156));
        obstacle[5].push_back(RVO::Vector2(-181.14,-148.65));
        obstacle[5].push_back(RVO::Vector2(-197.13,-148.65));

        obstacle[6].push_back(RVO::Vector2(-222.33,-156));
        obstacle[6].push_back(RVO::Vector2(-204.66,-156));
        obstacle[6].push_back(RVO::Vector2(-204.66,-148.28));
        obstacle[6].push_back(RVO::Vector2(-222.33,-148.28));

        obstacle[7].push_back(RVO::Vector2(-214.4,-143.25));
        obstacle[7].push_back(RVO::Vector2(-213.5,-143.25));
        obstacle[7].push_back(RVO::Vector2(-213.5,-142.4));
        obstacle[7].push_back(RVO::Vector2(-214.4,-142.4));

        obstacle[8].push_back(RVO::Vector2(-209.66,-144.35));
        obstacle[8].push_back(RVO::Vector2(-208.11,-144.35));
        obstacle[8].push_back(RVO::Vector2(-208.11,-142.8));
        obstacle[8].push_back(RVO::Vector2(-209.66,-142.8));

        obstacle[9].push_back(RVO::Vector2(-198.58,-144.2));
        obstacle[9].push_back(RVO::Vector2(-197.2,-144.2));
        obstacle[9].push_back(RVO::Vector2(-197.2,-142.92));
        obstacle[9].push_back(RVO::Vector2(-198.58,-142.92));

        obstacle[10].push_back(RVO::Vector2(-184.19,-143.88));
        obstacle[10].push_back(RVO::Vector2(-183.01,-143.87));
        obstacle[10].push_back(RVO::Vector2(-181.5,-141.9));
        obstacle[10].push_back(RVO::Vector2(-184.19,-142.53));

        obstacle[11].push_back(RVO::Vector2(-176,-143.69));
        obstacle[11].push_back(RVO::Vector2(-174.43,-143.69));
        obstacle[11].push_back(RVO::Vector2(-174.43,-142));
        obstacle[11].push_back(RVO::Vector2(-176,-142));


        for (int i=0; i<12; i++){
           sim->addObstacle(obstacle[i]);
        }

        sim->processObstacles();
}

PedPathPredictor::~PedPathPredictor(){
	if(world_his != NULL) {
		delete world_his;
		world_his = NULL;
	}
	if(sim != NULL) {
		delete sim;
		sim = NULL;
	}
}

void PedPathPredictor::UpdatePed(PedStruct &ped, double cur_pos_x, double cur_pos_y){
    ped.vel_cur.x = (cur_pos_x - ped.pos.x) / ModelParams::RVO2_TIME_PER_STEP;
    ped.vel_cur.y = (cur_pos_y - ped.pos.y) / ModelParams::RVO2_TIME_PER_STEP;
    ped.vel_pref = (ped.vel_cur * (1 - ModelParams::RVO2_PREVEL_WEIGHT)) + (ped.vel_pref * ModelParams::RVO2_PREVEL_WEIGHT);
    ped.pos.x = cur_pos_x;
    ped.pos.y = cur_pos_y;
}

void PedPathPredictor::Predict(const ros::TimerEvent &e){
	
	// Set up the scenario.
	// setupScenario(sim);

	state_cur = world_his->get_current_state();
	sim->clearAllAgents();

	sensor_msgs::PointCloud pc;
	pc.header.frame_id=global_frame_id;
    pc.header.stamp=ros::Time::now();

    //adding pedestrians
    for(int i=0; i<state_cur.peds.size(); i++){
        sim->addAgent(RVO::Vector2(state_cur.peds[i].pos.x, state_cur.peds[i].pos.y));
    }

	// Perform (and manipulate) the simulation.
	for(int step = 0; step < ModelParams::N_SIM_STEP; step++) {
    	for (size_t i = 0; i < state_cur.peds.size(); ++i) {
	        sim->setAgentPrefVelocity(i, RVO::Vector2(state_cur.peds[i].vel_pref.x, state_cur.peds[i].vel_pref.y));
	    }
	    sim->doStep();
	    for (size_t i = 0; i < state_cur.peds.size(); ++i) {
	    	geometry_msgs::Point32 p;
	        p.x = sim->getAgentPosition(i).x();
	        p.y = sim->getAgentPosition(i).y();
	        p.z = 1.0;
	        if(step % 2 == 0) pc.points.push_back(p); // publish ped positions every two time step

	        UpdatePed(state_cur.peds[i], p.x, p.y);
	    }
	}

	ped_prediction_pub.publish(pc);
}


