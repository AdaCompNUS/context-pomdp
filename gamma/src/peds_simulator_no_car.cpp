#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <peds_unity_system/car_info.h>
#include <peds_unity_system/ped_info.h>
#include <peds_unity_system/peds_info.h>
#include <peds_unity_system/peds_car_info.h>
#include <iostream>
#include <vector>
#include <RVO.h>
#include "coord.h"
#include <nav_msgs/Path.h>

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>


/**
 * H: center of the head of the car
 * N: a point right in front of the car
 * M: an arbitrary point
 *
 * Check whether M is in the safety zone
 */
bool inRealCollision(double Mx, double My, double Hx, double Hy, double Nx, double Ny);

struct Car {
    Car(){
        vel = 1.0;
        max_tracking_err = 0.0; //1.5;
        max_tracking_angle =45; //45.0;
    }
    COORD pos;
    double yaw;
    double vel;
    double max_tracking_err;
    double max_tracking_angle;
};

struct Ped {
    Ped(){
        vel = 1.2;
    }
    Ped(COORD a, int b, int c) {
        pos = a;
        goal = b;
        id = c;
        vel = 1.2;
    }
    COORD pos; //pos
    int goal;  //goal
    int id;   //id
    double vel;
    double vel_x;
    double vel_y;
};

class PedsSystem {
public:
    Car car;
    std::vector<Ped> peds;

    RVO::RVOSimulator* ped_sim_;

    ros::Subscriber peds_car_info_sub;
    ros::Subscriber vel_sub;
    ros::Publisher peds_info_pub;
    ros::Publisher path_pub;
    ros::Publisher speed_pub;
    std::vector<COORD> goals;
    COORD veh_goal;
    bool initialized;

    float freq;

    double pref_speed;
    float max_speed;
    int num_neighbor;
    float neighbor_dist;
    float car_radius;
    float goal_threshold;

    int cur_step;

    std::string obstacle_file_name_;
    std::string goal_file_name_;

    PedsSystem(){
        initialized = false;
        car.vel = 0.4;
        freq = 10.0;

        pref_speed = 1.5;
        max_speed = 1.5;
        num_neighbor = 6;
        neighbor_dist = 2.0;
        car_radius = 1.6;
        goal_threshold = 2.0;

        cur_step = 0;

        printCommonSetting();

        ros::NodeHandle n("~");
        n.param<std::string>("goal_file_name", goal_file_name_, "null");

        if (goal_file_name_=="null"){
            goals = { // indian cross 2 larger map
                COORD(3.5, 20.0),
                COORD(-3.5, 20.0), 
                COORD(3.5, -20.0), 
                COORD(-3.5, -20.0),
                COORD(20.0  , 3.5),
                COORD( 20.0 , -3.5), 
                COORD(-20.0 , 3.5), 
                COORD( -20.0, -3.5),
                COORD(-1, -1) // stop
            };
        }
        else{
            goals.resize(0);

            std::ifstream file;
            file.open(goal_file_name_, std::ifstream::in);

            if(file.fail()){
                std::cout<<"open goal file failed !!!!!!"<<std::endl;
                return;
            }

            std::string line;
            int goal_num = 0;
            while (std::getline(file, line))
            {
                std::istringstream iss(line);
                
                double x;
                double y;
                while (iss >> x >>y){
                    std::cout << x <<" "<< y<<std::endl;
                    goals.push_back(COORD(x, y));
                }

                goal_num++;
                if(goal_num > 99) break;
            }

            file.close();
        }

        n.param("goalx", veh_goal.x, 0.0);
        n.param("goaly", veh_goal.y, 19.9);

        std::cout << "car goal: " << veh_goal.x << " " << veh_goal.y << std::endl;

        ped_sim_ = new RVO::RVOSimulator();
    
        // Specify global time step of the simulation.
        ped_sim_->setTimeStep(1.0f/freq);    

        // setAgentDefaults (float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2 &velocity=Vector2())
        ped_sim_->setAgentDefaults(5.0f, 5, 2.5f, 2.5f, 0.25f, 2.5f); // densesense

        addObstacle();
    }

    void spin() {
        ros::NodeHandle nh;
        peds_car_info_sub = nh.subscribe("peds_car_info", 1, &PedsSystem::pedsCarCallBack, this);
        ros::Timer timer = nh.createTimer(ros::Duration(1 / freq), &PedsSystem::actionCallBack, this);
        vel_sub = nh.subscribe("cmd_vel_pomdp", 1, &PedsSystem::velCallBack, this);
        peds_info_pub = nh.advertise<peds_unity_system::peds_info>("peds_info",1);
        path_pub = nh.advertise<nav_msgs::Path>("plan",1, true);
        //speed_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel",1, true); // directly send cmd vel to unity
        speed_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel_pomdp",1, true); // send cmd vel to pure pursuit controller
        
        ros::spin();
    }

    int findPedwithID(int id){
        int i=0;
        for(i; i<peds.size(); i++){
            if(id == peds[i].id) return i;
        }
        return i;
    }

    void pedsCarCallBack(peds_unity_system::peds_car_infoConstPtr peds_car_ptr) {

        car.pos.x = peds_car_ptr->car.car_pos.x;
        car.pos.y = peds_car_ptr->car.car_pos.y;
        car.yaw = peds_car_ptr->car.car_yaw;

        car.vel = peds_car_ptr->car.car_speed;

        Ped tmp_ped;
        for (int i = 0; i < peds_car_ptr->peds.size(); i++)
        {
            tmp_ped.pos.x = peds_car_ptr->peds[i].ped_pos.x;
            tmp_ped.pos.y = peds_car_ptr->peds[i].ped_pos.y;
            tmp_ped.goal = peds_car_ptr->peds[i].ped_goal_id;
            tmp_ped.id = peds_car_ptr->peds[i].ped_id;
            //tmp_ped.vel = peds_car_ptr->peds[i].ped_speed;

            int index = findPedwithID(tmp_ped.id);
            if(index >= peds.size()) peds.push_back(tmp_ped);
            else peds[index] = tmp_ped;
        }

        initialized = true;
    }

    void actionCallBack(const ros::TimerEvent& event) {
        if (initialized == false) return;

        std::cout<<car.vel<<std::endl;
        
        RVO2PedStep();

        peds_unity_system::peds_info peds_info_msg;
        getPedsInfoMsg(peds_info_msg);
        //peds_info_pub.publish(peds_info_msg);
        peds.clear();
    }

    void velCallBack(geometry_msgs::TwistConstPtr action) {
/*        if (initialized == false) return;
        //if(action->linear.x==-1) car.vel = 0;
        //// car.vel = ( (action->linear.x <= 0.4) ? 0.4 : action->linear.x );
        car.vel = action->linear.x;
        if(action->linear.x==-1) car.vel = 0;
        //car.vel = 1.5;*/
    }

    void getPedsInfoMsg(peds_unity_system::peds_info & peds_info_msg){

        peds_unity_system::ped_info tmp_single_ped;

        for(int i=0; i< peds.size(); i++){
            tmp_single_ped.ped_pos.x = peds[i].pos.x;
            tmp_single_ped.ped_pos.y = peds[i].pos.y;
            tmp_single_ped.ped_pos.z = 0;
            tmp_single_ped.ped_goal_id = peds[i].goal;
            tmp_single_ped.ped_id = peds[i].id;
            tmp_single_ped.ped_speed = peds[i].vel;

            tmp_single_ped.ped_vel.x = peds[i].vel_x;
            tmp_single_ped.ped_vel.y = peds[i].vel_y;

            peds_info_msg.peds.push_back(tmp_single_ped);
        }

    }

    void printCommonSetting(){
        std::cout<<"common setting:"<<std::endl;
        std::cout<<"freq / veh_goal / car_radius / pref_speed / max_speed / num_neighbor / neighbor_dist : = "<<freq<<" "<<veh_goal<<" "<<car_radius<<" "<<pref_speed<<" "<<max_speed<<" "<<num_neighbor<<" "<<neighbor_dist<<" "<<std::endl;
    }
    void printState(){
        std::cout<<"planning state:"<<std::endl;
        std::cout<<"car state:"<<std::endl;
        std::cout<<"pos / yaw / speed / max_tracking_err / max_tracking_angle : = "<<car.pos<<" "<<car.yaw<<" "<<car.vel<<" "<<car.max_tracking_err<<" "<<car.max_tracking_angle<<std::endl;

        std::cout << "Distance to goal: " << COORD::EuclideanDistance(veh_goal, car.pos) << std::endl;

        std::cout<<"pedestrian state:"<<std::endl;
        for(int i=0; i<peds.size(); i++){

            int goal_id = peds[i].goal;
            RVO::Vector2 pos = RVO::Vector2(peds[i].pos.x, peds[i].pos.y);
            RVO::Vector2 pref_vel;

            COORD ped_goal = COORD(-1, -1); // stop intention
            if (goal_id >= goals.size()-1) { /// stop intention
                pref_vel = RVO::Vector2(0.0f, 0.0f);
            }
            else{
                ped_goal = goals[goal_id];
                RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
                if ( absSq(goal - pos) < 0.0625 ) {
                    pref_vel = normalize(goal - pos)*0.6;
                } else {
                    pref_vel = normalize(goal - pos)*1.2;
                }
            }
            std::cout<<"id / pos / pref_vel / ped_goal : = "<<peds[i].id<<" "<<peds[i].pos<<" "<<pref_vel<<" "<<ped_goal<<std::endl;
        }
    }

    // check if a and b are withing radius distance
    bool inCircle(COORD a, COORD b, float radius) {
        if((a.x - b.x)*(a.x - b.x)+(a.y - b.y)*(a.y - b.y) < radius * radius) return true;
        return false;
    }

    bool isGoalReached(){
        if(inCircle(veh_goal, car.pos, goal_threshold)) return true;
        return false;
    }

    bool inRectCollision(){ // in collision with the car represented by rectangle; this is the real collision

        RVO::Vector2 heading(cos(3.1415 / 180.0 * car.yaw), sin(3.1415 / 180.0 * car.yaw));

        for(int i=0; i<peds.size(); i++) {
            if(::inRealCollision(peds[i].pos.x, peds[i].pos.y, car.pos.x, car.pos.y, car.pos.x + heading.x(), car.pos.y + heading.y())) {
                return true;
            }
        }
        return false;
    }


    bool inCircleCollision() { // in collision with the car represented by circle; this is the collision the planner thinks of.

        for(int i=0; i<peds.size(); i++) {
            if(inCircle(peds[i].pos, car.pos, car_radius)) {
                return true;
            }
        }
        return false;
    }

    void RVO2PedStep(){

        std::cout << "====== executing step "<< cur_step << "======="<< std::endl;

        
        printState();
        if(isGoalReached()){
            std::cout<<"goal reached"<<std::endl;
            ros::shutdown();
        }
        if(car.vel >= 0.1 && inRectCollision()){
            std::cout<<"vehicle in real collision"<<std::endl;
        }
        if(car.vel >= 0.1 && inCircleCollision()){
            std::cout<<"vehicle in planner collision"<<std::endl;
        }


        ped_sim_->setNotUpdated();

        //adding pedestrians
        for(int i=0; i<peds.size(); i++){

            int goal_id = peds[i].goal;
            RVO::Vector2 pos = RVO::Vector2(peds[i].pos.x, peds[i].pos.y);
            RVO::Vector2 pref_vel;

            if (goal_id >= goals.size()-1) { /// stop intention
                pref_vel = RVO::Vector2(0.0f, 0.0f);
            }
            else{
                RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
                if ( absSq(goal - pos) < 0.0625 ) {
                    pref_vel = normalize(goal - pos)*0.6;
                } else {
                    pref_vel = normalize(goal - pos)*1.2;
                }
            }

            ped_sim_->updateAgent(peds[i].id, pos, pref_vel);
        }

        ped_sim_->deleteOldAgents();


        RVO::Vector2 prefered_vel( veh_goal.x - car.pos.x, veh_goal.y - car.pos.y );
        prefered_vel = pref_speed * RVO::normalize(prefered_vel);
        RVO::Vector2 heading(cos(3.1415 / 180.0 * car.yaw), sin(3.1415 / 180.0 * car.yaw));

        ped_sim_->addAgent(RVO::Vector2(car.pos.x + 0.56*5 * cos(3.1415 / 180.0 * car.yaw), car.pos.y + 2.8* sin(3.1415 / 180.0 * car.yaw)), neighbor_dist, num_neighbor, 1.0f, 2.0f, car_radius + car.max_tracking_err, max_speed, car.max_tracking_angle, heading, RVO::Vector2(), "vehicle");

        ped_sim_->setAgentPrefVelocity(peds.size(), prefered_vel); // the num_ped-th pedestrian is the car. set its prefered velocity 
        ped_sim_->setAgentPedID(peds.size(),-1);


        ped_sim_->doStep();


        double ref_x = car.pos.x + 0.56*5 * cos(3.1415 / 180.0 * car.yaw);
        double ref_y = car.pos.y + 2.8* sin(3.1415 / 180.0 * car.yaw);

        double target_x = ped_sim_->getAgentPosition(peds.size()).x();
        double target_y = ped_sim_->getAgentPosition(peds.size()).y();

        for(int i=0; i<peds.size(); i++){
            peds[i].vel = freq*sqrt((ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)*(ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)
                +(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y)*(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y));

            peds[i].vel_x = freq*(ped_sim_->getAgentPosition(i).x()-peds[i].pos.x);
            peds[i].vel_y = freq*(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y);

            peds[i].pos.x=ped_sim_->getAgentPosition(i).x();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
            peds[i].pos.y=ped_sim_->getAgentPosition(i).y();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

        }


        double speed = freq*sqrt((target_x - ref_x)*(target_x - ref_x)
            + (target_y - ref_y)*(target_y - ref_y));


/*        std::cout<<"##################"<<std::endl;
        std::cout<<"speed: "<<speed<<std::endl;
        std::cout<<"path: "<< car.pos.x << " "<<car.pos.y << " "<< ped_sim_->getAgentPosition(peds.size()).x()<< " "<< ped_sim_->getAgentPosition(peds.size()).y()<<std::endl;
        std::cout<<"heading: "<< heading.x() << " "<<heading.y()<<std::endl;*/
        
        double new_vel_orientation = atan2(target_y - ref_y, target_x- ref_x);
        new_vel_orientation = new_vel_orientation/3.14*180;
        while(new_vel_orientation>180) new_vel_orientation-=180;
        while(new_vel_orientation<-180) new_vel_orientation+=180;

//        std::cout<<"heading yaw: "<<car.yaw<<"  vel ori: "<<new_vel_orientation<< "  diff: "<<car.yaw - new_vel_orientation<<std::endl;

        static int num_vel_in_angle_bound = 0;
        static int num_plan = 0;
        num_plan ++;
        if(abs(car.yaw - new_vel_orientation) < car.max_tracking_angle) {
            // publishPath(car.pos.x, car.pos.y, ped_sim_->getAgentPosition(peds.size()).x(), ped_sim_->getAgentPosition(peds.size()).y());
            //publishPath(ref_x, ref_y, target_x, target_y);
            publishPath(ref_x, ref_y, target_x, target_y, car.pos.x, car.pos.y);
            publishSpeed(speed);

            num_vel_in_angle_bound++;
        } else{
            publishPath(ref_x, ref_y, ref_x + cos(3.1415 / 180.0 * car.yaw), ref_y + sin(3.1415 / 180.0 * car.yaw), car.pos.x, car.pos.y);
            //publishPath(ref_x, ref_y, ref_x + cos(3.1415 / 180.0 * car.yaw), ref_y + sin(3.1415 / 180.0 * car.yaw));
            publishSpeed(0.0f);
        }

//        std::cout<<"num plan: "<<num_plan<<"  in bound: "<<num_vel_in_angle_bound<< "  ratio: "<<num_vel_in_angle_bound/float(num_plan)<<std::endl;
        cur_step++;

    }


    double GenGaussNum(double mean, double std){

        double u1 = 1.0-(std::rand()%10000)/10000.0; //uniform(0,1] random float
        double u2 = 1.0-(std::rand()%10000)/10000.0;
        double rand_std_normal = std::sqrt(-2.0 * std::log(u1)) *
            sin(2.0 * 3.14159 * u2); //random normal(0,1)
        double rand_normal =
            mean + std * rand_std_normal;

        return rand_normal;
    }

    double AddGaussNoise(double value, double mean, double std){
        return value + GenGaussNum (mean, std);
    }


    void addObstacle(){

        ros::NodeHandle n("~");
        n.param<std::string>("obstacle_file_name", obstacle_file_name_, "null");

        if(obstacle_file_name_=="null") {
            std::vector<RVO::Vector2> obstacle[4];

            obstacle[0].push_back(RVO::Vector2(-4,-4));
            obstacle[0].push_back(RVO::Vector2(-30,-4));
            obstacle[0].push_back(RVO::Vector2(-30,-30));
            obstacle[0].push_back(RVO::Vector2(-4,-30));

            obstacle[1].push_back(RVO::Vector2(4,-4));
            obstacle[1].push_back(RVO::Vector2(4,-30));
            obstacle[1].push_back(RVO::Vector2(30,-30));
            obstacle[1].push_back(RVO::Vector2(30,-4));

    		obstacle[2].push_back(RVO::Vector2(4,4));
            obstacle[2].push_back(RVO::Vector2(30,4));
            obstacle[2].push_back(RVO::Vector2(30,30));
            obstacle[2].push_back(RVO::Vector2(4,30));

    		obstacle[3].push_back(RVO::Vector2(-4,4));
            obstacle[3].push_back(RVO::Vector2(-4,30));
            obstacle[3].push_back(RVO::Vector2(-30,30));
            obstacle[3].push_back(RVO::Vector2(-30,4));

            for (int i=0; i<4; i++){
               ped_sim_->addObstacle(obstacle[i]);
            }

            ped_sim_->processObstacles();

        } else {

            std::ifstream file;
            file.open(obstacle_file_name_, std::ifstream::in);

            if(file.fail()){
                std::cout<<"open "<<obstacle_file_name_<<" failed !!!!!!"<<std::endl;
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
                    std::cout << x <<" "<< y<<std::endl;
                    obstacles[obst_num].push_back(RVO::Vector2(x, y));
                }

                ped_sim_->addObstacle(obstacles[obst_num]);         
                

                obst_num++;
                if(obst_num > 99) break;
            }

            ped_sim_->processObstacles();               
                
            file.close();
        }
    } 


    void publishPath(double x_start, double y_start, double x_target, double y_target, double rear_center_x, double rear_center_y) {
        nav_msgs::Path navpath;
        ros::Time plan_time = ros::Time::now();

        RVO::Vector2 normalize_direction = RVO::normalize(RVO::Vector2(x_target - x_start, y_target - y_start));
        RVO::Vector2 new_target = 3.0*normalize_direction;

        navpath.header.frame_id = "map";
        navpath.header.stamp = plan_time;
        

        geometry_msgs::PoseStamped pose;
        pose.header.stamp = plan_time;
        pose.header.frame_id = navpath.header.frame_id;
        pose.pose.position.x = rear_center_x;
        pose.pose.position.y = rear_center_y;
        pose.pose.position.z = 0.0;
        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = 0.0;
        pose.pose.orientation.w = 1.0;
        navpath.poses.push_back(pose);


        pose.pose.position.x = rear_center_x + new_target.x();
        pose.pose.position.y = rear_center_y + new_target.y();
        navpath.poses.push_back(pose);
        

        path_pub.publish(navpath);
    }

    void publishPath(double x_start, double y_start, double x_target, double y_target) {
        nav_msgs::Path navpath;
        ros::Time plan_time = ros::Time::now();

        RVO::Vector2 normalize_direction = RVO::normalize(RVO::Vector2(x_target - x_start, y_target - y_start));
        RVO::Vector2 new_target = 3.0*normalize_direction;

        navpath.header.frame_id = "map";
        navpath.header.stamp = plan_time;
        

        geometry_msgs::PoseStamped pose;
        pose.header.stamp = plan_time;
        pose.header.frame_id = navpath.header.frame_id;
        pose.pose.position.x = x_start;
        pose.pose.position.y = y_start;
        pose.pose.position.z = 0.0;
        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = 0.0;
        pose.pose.orientation.w = 1.0;
        navpath.poses.push_back(pose);


        pose.pose.position.x = x_start + new_target.x();
        pose.pose.position.y = y_start + new_target.y();
        navpath.poses.push_back(pose);
        

        path_pub.publish(navpath);
    }

    void publishSpeed(double speed)
    {
        geometry_msgs::Twist cmd;
        cmd.angular.z = 0;
        cmd.linear.x = speed;
        cmd.linear.y = car.vel;
        speed_pub.publish(cmd);

        std::cout<<"current speed: "<<cmd.linear.y<<"  target speed: "<<cmd.linear.x<<std::endl;
    }


};

int main(int argc,char**argv)
{
	ros::init(argc,argv,"peds_unity_system");
    std::srand(std::time(0));
    PedsSystem peds_system;
    peds_system.spin();
	return 0;
}
