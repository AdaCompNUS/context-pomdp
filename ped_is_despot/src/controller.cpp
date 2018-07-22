#include "controller.h"
#include "core/node.h"
#include "core/solver.h"
#include "core/globals.h"
#include <csignal>
using namespace std;
static ped_is_despot::imitation_data p_IL_data; //for imitation learning



double marker_colors[20][3] = {
    	{0.0,1.0,0.0},  //green
		{1.0,0.0,0.0},  //red
		{0.0,0.0,1.0},  //blue
		{1.0,1.0,0.0},  //sky blue
		{0.0,1.0,1.0},  //yellow
		{0.5,0.25,0.0}, //brown
		{1.0,0.0,1.0}   //pink
};

int action_map[3]={2,0,1};

struct my_sig_action {
    typedef void (* handler_type)(int, siginfo_t*, void*);

    explicit my_sig_action(handler_type handler)
    {
        memset(&_sa, 0, sizeof(struct sigaction));
        _sa.sa_sigaction = handler;
        _sa.sa_flags = SA_SIGINFO;
    }

    operator struct sigaction const*() const
    {
        return &_sa;
    }
protected:
    struct sigaction _sa;
};

struct div_0_exception {};

void handle_div_0(int sig, siginfo_t* info, void*)
{
	switch(info->si_code)
	{
		case FPE_INTDIV:
	        cout<< "Integer divide by zero."<<endl;
			break;
		case FPE_INTOVF:
			cout<< "Integer overflow. "<<endl;
			break;
		case FPE_FLTUND:
			cout<< "Floating-point underflow. "<<endl;	
			break;
		case FPE_FLTRES:
			cout<< "Floating-point inexact result. "<<endl;
			break;
		case FPE_FLTINV:
			cout<< "Floating-point invalid operation. "<<endl;
			break;
		case FPE_FLTSUB:
			cout<< "Subscript out of range. "<<endl;
			break;
		case FPE_FLTDIV:
			cout<< "Floating-point divide by zero. "<<endl;
			break;
		case FPE_FLTOVF:
			cout<< "Floating-point overflow. "<<endl;
			break;
	};
	exit(-1);
}

Controller::Controller(ros::NodeHandle& nh, bool fixed_path, double pruning_constant, double pathplan_ahead, string obstacle_file_name):  worldStateTracker(worldModel), worldBeliefTracker(worldModel, worldStateTracker), fixed_path_(fixed_path), pathplan_ahead_(pathplan_ahead), obstacle_file_name_(obstacle_file_name)
{
	my_sig_action sa(handle_div_0);
    if (0 != sigaction(SIGFPE, sa, NULL)) {
        std::cerr << "!!!!!!!! fail to setup handler !!!!!!!!" << std::endl;
        //return 1;
    }

/// for golfcart
 /*   Path p;
    COORD start = COORD(-205, -142.5);
    COORD goal = COORD(-189, -142.5);
    p.push_back(start);
    p.push_back(goal);
    worldModel.setPath(p.interpolate());
    fixed_path_ = true;
*/

/// for audi r8
    fixed_path_=false;

    

	cout << "fixed_path = " << fixed_path_ << endl;
	cout << "pathplan_ahead = " << pathplan_ahead_ << endl;
	Globals::config.pruning_constant = pruning_constant;

	global_frame_id = ModelParams::rosns + "/map";
	cerr <<"DEBUG: Entering Controller()"<<endl;
	control_freq=ModelParams::control_freq;

	cerr << "DEBUG: Initializing publishers..." << endl;
    believesPub_ = nh.advertise<ped_is_despot::peds_believes>("peds_believes",1);
    cmdPub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel_pomdp",1);
	plannerPedsPub_=nh.advertise<sensor_msgs::PointCloud>("planner_peds",1);
    actionPub_ = nh.advertise<visualization_msgs::Marker>("pomdp_action",1);
    actionPubPlot_= nh.advertise<geometry_msgs::Twist>("pomdp_action_plot",1);
	pathPub_= nh.advertise<nav_msgs::Path>("pomdp_path_repub",1, true);
	pathSub_= nh.subscribe("plan", 1, &Controller::RetrievePathCallBack, this);
    navGoalSub_ = nh.subscribe("navgoal", 1, &Controller::setGoal, this);
	goal_pub=nh.advertise<visualization_msgs::MarkerArray> ("pomdp_goals",1);
	start_goal_pub=nh.advertise<ped_pathplan::StartGoal> ("ped_path_planner/planner/start_goal", 1);

	markers_pub=nh.advertise<visualization_msgs::MarkerArray>("pomdp_belief",1);
    pedStatePub_=nh.advertise<sensor_msgs::PointCloud>("ped_state", 1);
    pedPredictionPub_ = nh.advertise<sensor_msgs::PointCloud>("ped_prediction", 1);

   	// for imitation learning
   	carSub_ = nh.subscribe("IL_car_info", 1, &Controller::update_il_car, this);
   	steerSub_ = nh.subscribe("IL_steer_cmd", 1, &Controller::update_il_steering, this);
    IL_pub = nh.advertise<ped_is_despot::imitation_data>("il_data", 1);

    ros::NodeHandle n("~");
    n.param<std::string>("goal_file_name", worldModel.goal_file_name_, "null");

	safeAction=2;
	target_speed_=0.0;
	goal_reached=false;
	worldModel.InitPedGoals();
	cerr << "DEBUG: Init simulator" << endl;
	initSimulator();
    //RetrievePaths();
	worldModel.InitRVO();
    addObstacle();

	cerr <<"DEBUG: before entering controlloop"<<endl;
    timer_ = nh.createTimer(ros::Duration(1.0/control_freq), &Controller::controlLoop, this);

	last_acc_=0;
	b_update_il = true;
	b_use_drive_net_ = false;
	if(!b_use_drive_net_)
		timer_speed=nh.createTimer(ros::Duration(0.05), &Controller::publishSpeed, this);
	else
		cout << "Node not publishing cmd_vel_pomdp: drive_net is used instead."<< endl;

}



bool Controller::getObjectPose(string target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const
{
    out_pose.setIdentity();

    try {
        tf_.transformPose(target_frame, in_pose, out_pose);
    }
    catch(tf::LookupException& ex) {
        ROS_ERROR("No Transform available Error: %s\n", ex.what());
        return false;
    }
    catch(tf::ConnectivityException& ex) {
        ROS_ERROR("Connectivity Error: %s\n", ex.what());
        return false;
    }
    catch(tf::ExtrapolationException& ex) {
        ROS_ERROR("Extrapolation Error: %s\n", ex.what());
        return false;
    }
    return true;
}

void Controller::addObstacle(){

	if(obstacle_file_name_ == "null"){
	/*    std::vector<RVO::Vector2> obstacle[12];

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

		int NumThreads=Globals::config.NUM_THREADS;

	    for(int tid=0; tid<NumThreads;tid++){
		    for (int i=0; i<12; i++){
		 	   worldModel.ped_sim_[tid]->addObstacle(obstacle[i]);
			}

		    //Process the obstacles so that they are accounted for in the simulation.
		    worldModel.ped_sim_[tid]->processObstacles();
		}*/


		/// for indian_cross
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


	    int NumThreads=Globals::config.NUM_THREADS;

	    for(int tid=0; tid<NumThreads;tid++){
		    for (int i=0; i<4; i++){
		 	   worldModel.ped_sim_[tid]->addObstacle(obstacle[i]);
			}

		    /* Process the obstacles so that they are accounted for in the simulation. */
		    worldModel.ped_sim_[tid]->processObstacles();
		}
	} else{

		int NumThreads=Globals::config.NUM_THREADS;

		ifstream file;
	    file.open(obstacle_file_name_, std::ifstream::in);

	    if(file.fail()){
	        cout<<"open obstacle file failed !!!!!!"<<endl;
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

	        for(int tid=0; tid<NumThreads;tid++){
			 	worldModel.ped_sim_[tid]->addObstacle(obstacles[obst_num]);			
			}

	        obst_num++;
	        if(obst_num > 99) break;
	    }

	    for(int tid=0; tid<NumThreads;tid++){
			worldModel.ped_sim_[tid]->processObstacles();			    
		}
	    
	    file.close();
	}

}

/*for despot*/
void Controller::initSimulator()
{
  Globals::config.root_seed=1024;
  //Globals::config.n_belief_particles=2000;
  Globals::config.num_scenarios=50;
  Globals::config.time_per_move = (1.0/ModelParams::control_freq) * 0.9;
  Globals::config.max_policy_sim_len=25;
  Seeds::root_seed(Globals::config.root_seed);
  cerr << "Random root seed set to " << Globals::config.root_seed << endl;

  // Global random generator
  double seed = Seeds::Next();
  Random::RANDOM = Random(seed);
  cerr << "Initialized global random generator with seed " << seed << endl;

  gpu_handler=new Simulator(/*NULL, NULL*/);

  gpu_handler->InitializeDefaultParameters() ;

  gpu_handler->PrepareGPU();


  despot=new PedPomdp(worldModel); 
  //despot->num_active_particles = 0;

/*  RandomStreams* streams = NULL;
  streams = new RandomStreams(Seeds::Next(Globals::config.num_scenarios), Globals::config.search_depth);
  despot->InitializeParticleLowerBound("smart");
  despot->InitializeScenarioLowerBound("smart", *streams);
  despot->InitializeParticleUpperBound("smart", *streams);
  despot->InitializeScenarioUpperBound("smart", *streams);*/

  ScenarioLowerBound *lower_bound = despot->CreateScenarioLowerBound("SMART");
  ScenarioUpperBound *upper_bound = despot->CreateScenarioUpperBound("SMART", "SMART");

  //solver = new DESPOT(despot, NULL, *streams);
  solver = new DESPOT(despot, lower_bound, upper_bound, NULL, Globals::config.useGPU);

  gpu_handler->PrepareGPUData(despot, solver);

  if(worldModel.path.size()>0)
  	gpu_handler->UpdateGPUPath(despot);
}



Controller::~Controller()
{
    geometry_msgs::Twist cmd;
    cmd.angular.z = 0;
    cmd.linear.x = 0;
    cmdPub_.publish(cmd);
    gpu_handler->ReleaseGPUData(despot, solver);
    delete gpu_handler;
}



void Controller::updateSteerAnglePublishSpeed(geometry_msgs::Twist speed)
{
    //cmdPub_.publish(speed);
}

void Controller::publishSpeed(const ros::TimerEvent &e)
{
	geometry_msgs::Twist cmd;
	cmd.angular.z = 0;
	cmd.linear.x = target_speed_;
	cmd.linear.y = real_speed_;
	//cout<<"publishing cmd speed "<<target_speed_<<endl;
	cmdPub_.publish(cmd);
}



void Controller::publishROSState()
{
	geometry_msgs::Point32 pnt;
	
	geometry_msgs::Pose pose;
	
	geometry_msgs::PoseStamped pose_stamped;
	pose_stamped.header.stamp=ros::Time::now();

	pose_stamped.header.frame_id=global_frame_id;

	pose_stamped.pose.position.x=(worldStateTracker.carpos.x+0.0);
	pose_stamped.pose.position.y=(worldStateTracker.carpos.y+0.0);
	pose_stamped.pose.orientation.w=1.0;
	car_pub.publish(pose_stamped);
	
	geometry_msgs::PoseArray pA;
	pA.header.stamp=ros::Time::now();

	pA.header.frame_id=global_frame_id;
	for(int i=0;i<worldStateTracker.ped_list.size();i++)
	{
		//GetCurrentState(ped_list[i]);
		pose.position.x=(worldStateTracker.ped_list[i].w+0.0);
		pose.position.y=(worldStateTracker.ped_list[i].h+0.0);
		pose.orientation.w=1.0;
		pA.poses.push_back(pose);
		//ped_pubs[i].publish(pose);
	}

	pa_pub.publish(pA);


	uint32_t shape = visualization_msgs::Marker::CYLINDER;

	for(int i=0;i<worldModel.goals.size();i++)
	{
		visualization_msgs::Marker marker;

		marker.header.frame_id=ModelParams::rosns+"/map";
		marker.header.stamp=ros::Time::now();
		marker.ns="basic_shapes";
		marker.id=i;
		marker.type=shape;
		marker.action = visualization_msgs::Marker::ADD;

		marker.pose.position.x = worldModel.goals[i].x;
		marker.pose.position.y = worldModel.goals[i].y;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;

		marker.scale.x = 1;
		marker.scale.y = 1;
		marker.scale.z = 1;
		marker.color.r = marker_colors[i][0];
		marker.color.g = marker_colors[i][1];
		marker.color.b = marker_colors[i][2];
		marker.color.a = 1.0;
		
		markers.markers.push_back(marker);
	}
	goal_pub.publish(markers);
	markers.markers.clear();
}



void Controller::publishPlannerPeds(const State &s)
{
	const PomdpState & state=static_cast<const PomdpState&> (s);
	sensor_msgs::PointCloud pc;
	pc.header.frame_id=ModelParams::rosns+"/map";
	pc.header.stamp=ros::Time::now();
	for(int i=0;i<state.num;i++) {
		geometry_msgs::Point32 p;
		p.x=state.peds[i].pos.x;
		p.y=state.peds[i].pos.y;
		p.z=1.5;
		pc.points.push_back(p);
	}
	plannerPedsPub_.publish(pc);	
}
void Controller::publishAction(int action, double reward)
{
		uint32_t shape = visualization_msgs::Marker::CUBE;
		visualization_msgs::Marker marker;
        
		
		marker.header.frame_id=global_frame_id;
		marker.header.stamp=ros::Time::now();
		marker.ns="basic_shapes";
		marker.id=0;
		marker.type=shape;
		marker.action = visualization_msgs::Marker::ADD;

		double px,py;
		px=worldStateTracker.carpos.x;
		py=worldStateTracker.carpos.y;
		marker.pose.position.x = px+1;
		marker.pose.position.y = py+1;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;

		// Set the scale of the marker in meters
		marker.scale.x = 0.6;
		marker.scale.y = 3;
		marker.scale.z = 0.6;

		marker.color.r = marker_colors[action_map[action]][0];
        marker.color.g = marker_colors[action_map[action]][1];
		marker.color.b = marker_colors[action_map[action]][2];
		marker.color.a = 1.0;

		ros::Duration d(1/control_freq);
		marker.lifetime=d;
		actionPub_.publish(marker);
        geometry_msgs::Twist action_cmd;
        action_cmd.linear.x=safeAction;
        action_cmd.linear.y=reward;
        actionPubPlot_.publish(action_cmd);
}

COORD poseToCoord(const tf::Stamped<tf::Pose>& pose) {
	COORD coord;
	coord.x=pose.getOrigin().getX();
	coord.y=pose.getOrigin().getY();
	return coord;
}


geometry_msgs::PoseStamped Controller::getPoseAhead(const tf::Stamped<tf::Pose>& carpose) {
    static int last_i = -1;
    static double last_yaw = 0;
    static COORD last_ahead;
	auto& path = worldModel.path;
	COORD coord = poseToCoord(carpose);

	int i = path.nearest(coord);
    COORD ahead;
    double yaw;

    if (i == last_i) {
        yaw = last_yaw;
        ahead = last_ahead;
    } else {
        int j = path.forward(i, pathplan_ahead_);
        //yaw = (path.getYaw(j) + path.getYaw(j-1)+ path.getYaw(j-2)) / 3;
        yaw = path.getYaw(j);
        ahead = path[j];
        last_yaw = yaw;
		last_ahead = ahead;
        last_i = i;
    }

	auto q = tf::createQuaternionFromYaw(yaw);
	tf::Pose p(q, tf::Vector3(ahead.x, ahead.y, 0));
	tf::Stamped<tf::Pose> pose_ahead(p, carpose.stamp_, carpose.frame_id_);
	geometry_msgs::PoseStamped posemsg;
	tf::poseStampedTFToMsg(pose_ahead, posemsg);
	return posemsg;
}

void Controller::sendPathPlanStart(const tf::Stamped<tf::Pose>& carpose) {
	if(fixed_path_ && worldModel.path.size()>0)  return;

	ped_pathplan::StartGoal startGoal;
	geometry_msgs::PoseStamped pose;
	tf::poseStampedTFToMsg(carpose, pose);

	//cout.precision(8);

	// set start
	if(pathplan_ahead_ > 0 && worldModel.path.size()>0) {
		startGoal.start = getPoseAhead(carpose);
	} else {
		startGoal.start=pose;
	}

	pose.pose.position.x=goalx_;
	pose.pose.position.y=goaly_;

	// set goal
	// create door
	//pose.pose.position.x=17;
	//pose.pose.position.y=52;

	// after CREATE door
	//pose.pose.position.x=19.5;
	//pose.pose.position.y=55.5;
	
	// before create door
	//pose.pose.position.x=18.8;
	//pose.pose.position.y=44.5;
	//
	// cheers
	//pose.pose.position.x=4.0;
	//pose.pose.position.y=9.0;
	
	
	// large map goal
	//pose.pose.position.x=108;
	//pose.pose.position.y=143;
	
	startGoal.goal=pose;
	start_goal_pub.publish(startGoal);	
}

void Controller::setGoal(const geometry_msgs::PoseStamped::ConstPtr goal) {
    goalx_ = goal->pose.position.x;
    goaly_ = goal->pose.position.y;
    gpu_handler->UpdateGPUGoals(despot);

}
void Controller::RetrievePathCallBack(const nav_msgs::Path::ConstPtr path)  {
//	cout<<"receive path from navfn "<<path->poses.size()<<endl;
	if(fixed_path_ && worldModel.path.size()>0) return;

	if(path->poses.size()==0) return;

	if (b_update_il == true)
		p_IL_data.plan = *path; // record to be further published for imitation learning

	Path p;
	for(int i=0;i<path->poses.size();i++) {
        COORD coord;
		coord.x=path->poses[i].pose.position.x;
		coord.y=path->poses[i].pose.position.y;
        p.push_back(coord);
	}

	if(pathplan_ahead_>0 && worldModel.path.size()>0) {
        /*double pd = worldModel.path.mindist(p[0]);
        if(pd < 2 * ModelParams::PATH_STEP) {
            // only accept new path if the starting point is close to current path
            worldModel.path.cutjoin(p);
            auto pi = worldModel.path.interpolate();
            worldModel.setPath(pi);
        }*/
        worldModel.path.cutjoin(p);
        auto pi = worldModel.path.interpolate();
        worldModel.setPath(pi);
	} else {
		worldModel.setPath(p.interpolate());
	}
	gpu_handler->UpdateGPUPath(despot);

	publishPath(path->header.frame_id, worldModel.path);
}

void Controller::update_il_car(const peds_unity_system::car_info::ConstPtr car) {
    if (b_update_il == true){
    	p_IL_data.past_car = p_IL_data.cur_car;
    	p_IL_data.cur_car = *car;
    }
}

void Controller::update_il_steering(const std_msgs::Float32::ConstPtr steer){
	if (b_update_il == true){
    	p_IL_data.action_reward.angular.x = float(steer->data);
    	cout<< "receive steer "<<steer->data <<endl;
    }
}

void Controller::publishPath(const string& frame_id, const Path& path) {
	nav_msgs::Path navpath;
	ros::Time plan_time = ros::Time::now();

	navpath.header.frame_id = frame_id;
	navpath.header.stamp = plan_time;
	
	for(const auto& s: path) {
		geometry_msgs::PoseStamped pose;
		pose.header.stamp = plan_time;
		pose.header.frame_id = frame_id;
		pose.pose.position.x = s.x;
		pose.pose.position.y = s.y;
		pose.pose.position.z = 0.0;
		pose.pose.orientation.x = 0.0;
		pose.pose.orientation.y = 0.0;
		pose.pose.orientation.z = 0.0;
		pose.pose.orientation.w = 1.0;
		navpath.poses.push_back(pose);
	}

	pathPub_.publish(navpath);
}


void Controller::controlLoop(const ros::TimerEvent &e)
{		
		static int run_step = 0;

		cout.precision(8);
        /*static*/ double starttime=get_time_second();
        cout<<"*********************"<<endl;
	 //   cout<<"entering control loop"<<endl;
     //   cout<<"current time "<<get_time_second()-starttime<<endl;
        tf::Stamped<tf::Pose> in_pose, out_pose;

  
        /****** update world state ******/
        ros::Rate err_retry_rate(10);

        //Planning for next step
       //pathplan_ahead_=target_speed_/ModelParams::control_freq;
       pathplan_ahead_ = 0;

      //  cout<<"*** path ahead: "<<pathplan_ahead_<<endl;

		// transpose to base link for path planing
		in_pose.setIdentity();
		in_pose.frame_id_ = ModelParams::rosns + "/base_link";
		if(!getObjectPose(global_frame_id, in_pose, out_pose)) {
			cerr<<"transform error within control loop"<<endl;
			cout<<"laser frame "<<in_pose.frame_id_<<endl;
            err_retry_rate.sleep();
            return;
		}

		sendPathPlanStart(out_pose);
		if(worldModel.path.size()==0) {
			cout<<"Empty path."<<endl;
			return;
		}

		// transpose to laser frame for ped avoidance
		in_pose.setIdentity();
		in_pose.frame_id_ = ModelParams::rosns + ModelParams::laser_frame;
		if(!getObjectPose(global_frame_id, in_pose, out_pose)) {
			cerr<<"transform error within control loop"<<endl;
			cout<<"laser frame "<<in_pose.frame_id_<<endl;
            err_retry_rate.sleep();
            return;
		}


		COORD coord;

		ped_pathplan::StartGoal startGoal;
		if(pathplan_ahead_ > 0 && worldModel.path.size()>0) {
			startGoal.start = getPoseAhead(out_pose);
			coord.x = startGoal.start.pose.position.x;
			coord.y = startGoal.start.pose.position.y;
		}else{
			coord = poseToCoord(out_pose);
		}

		

		//cout << "after get topics / update world state:" << endl;
        //cout<<"current time "<<get_time_second()-starttime<<endl;
        worldStateTracker.updateVel(real_speed_/*+last_acc_*/);
        //cout<< "real speed: "<<real_speed_<<endl;

	//cout << "transformed pose = " << coord.x << " " << coord.y << endl;
		cout << "======transformed pose = "<< coord.x << " " <<coord.y << endl;

		worldStateTracker.updateCar(coord);

        worldStateTracker.cleanPed();

		PomdpState curr_state = worldStateTracker.getPomdpState();

		b_update_il = false ; // imitation learning: pause update of car info and path info for imitation data
		
		if(real_speed_ >= 0.05 && worldModel.inRealCollision(curr_state)){
			if(goal_reached==true)
				cout << "after reach goal: in collision"<<endl;
			else
				cout << "INININ in collision at step "<<run_step<<endl;
		}
		if(real_speed_ >= 0.05 && worldModel.inCollision(curr_state)){
			if(goal_reached==true)
				cout << "after reach goal: near collision"<<endl;
			else
				cout << "near collision at step "<< run_step<<endl;
		}
		publishROSState();


       // cout << "root state:" << endl;
		//despot->PrintState(curr_state, cout);

        /****** check world state *****/

		if(worldModel.isGlobalGoal(curr_state.car)) {
			goal_reached=true;
		}
		if(goal_reached==true) {
			safeAction=2;

			cout<<"goal reached at step "<<run_step<<endl;

			ros::shutdown();

			target_speed_=real_speed_;
		    target_speed_ -= 0.5;
			if(target_speed_<=0.0) target_speed_ = 0.0;

            // shutdown the node after reaching goal
            // TODO consider do this in simulaiton only for safety
            if(real_speed_ <= 0.01) {
            	cout<<"real_spead rrrr"<<endl;
                ros::shutdown();
            }
			return;
		}

		if(worldStateTracker.emergency())
		{
			target_speed_=-1;
			cout<<"emergency eeee"<<endl;
			return;
		}


        /****** update belief ******/

		//cout<<"before belief update"<<endl;
		worldBeliefTracker.update();
		//cout<<"after belief update"<<endl;

        publishPedsPrediciton();

        //cout<<"1"<<endl;

        vector<PomdpState> samples = worldBeliefTracker.sample(2000);
        //cout<<"2"<<endl;
		vector<State*> particles = despot->ConstructParticles(samples);
		//cout<<"3"<<endl;

        /*
		double sum=0;
		for(int i=0; i<particles.size(); i++)
			sum+=particles[i]->weight;
		cout<<"particle weight sum "<<sum<<endl;
        */

		cout<<"run_step: "<<run_step<<endl;
		run_step ++;
		ParticleBelief *pb=new ParticleBelief(particles, despot);
		//cout<<"4"<<endl;
     //   despot->PrintState(*(pb->particles()[0]));
		publishPlannerPeds(*(pb->particles()[0]));
		//cout<<"5"<<endl;
		
		solver->belief(pb);
		//cout<<"6"<<endl;

        /****** random simulation for verification purpose ******/
        /*
        Random rand((unsigned)1012312);
        cout << "Starting simulation" << endl;
        for (int i = 0; i < 100; i ++) {
            uint64_t obs;
            double reward;

            PomdpState state = worldBeliefTracker.sample();
            if (state.num <= 1)
                continue;

            cout << "Initial state: " << endl;
            despot->PrintState(state);

	        sensor_msgs::PointCloud pc;
        	pc.header.frame_id="/map";
            pc.header.stamp=ros::Time::now();
            ros::Rate loop_rate(ModelParams::control_freq);
            for (int j = 0; j < 20; j ++) {

                cout << "Step " << j << endl;
                int action = rand.NextInt(despot->NumActions());
                double r = rand.NextDouble();
                despot->Step(state, r, action, reward, obs);
                cout << "Action = " << action << endl;
                despot->PrintState(state);
                cout << "Reward = " << reward << endl;
                cout << "Obs = " << obs << endl;
                worldBeliefTracker.printBelief();
                for(int k=0;k<state.num;k++) {
       		        geometry_msgs::Point32 p;
                    p.x=state.peds[k].pos.x;
                    p.y=state.peds[k].pos.y;
                    p.z=1.0;
                    pc.points.push_back(p);
                }

                pedStatePub_.publish(pc); 
  //              publishBelief();
                loop_rate.sleep();
            }

        }
        cout << "End of simulation" << endl;
        */


        /****** solve for safe action ******/

		safeAction=solver->Search().action;
		//cout<<"7"<<endl;
		//safeAction = 1;

		//actionPub_.publish(action);
		double step_reward=StepReward(curr_state,safeAction);
		//printf("%.5f", step_reward);
		//cout<< "step reward **="<< step_reward<<endl;

		//publishAction(safeAction, step_reward); // replaced by publishImitationData
		//cout<<"8"<<endl;

		cout<<"action **= "<<safeAction<<endl;

		cout<<"reward **= "<<step_reward<<endl;


		//publishBelief(); // replaced by publishImitationData

		
		//cout<<"9"<<endl;


        /****** update target speed ******/

		target_speed_=real_speed_+last_acc_;
		double last_speed=real_speed_+last_acc_;
		cout<<"real speed: "<<real_speed_<<endl;
		cout<<"predicted real speed: "<<real_speed_+last_acc_<<endl;
        if(safeAction==0) {
        	last_acc_=0;
        }
		else if(safeAction==1) {
			last_acc_= ModelParams::AccSpeed/ModelParams::control_freq;
			target_speed_+=last_acc_;
		}
		else if(safeAction==2) {
			last_acc_=- ModelParams::AccSpeed/ModelParams::control_freq;
			target_speed_+=last_acc_;
		}
		if(target_speed_<=0.0) {
			target_speed_ = 0.0;
			last_acc_=target_speed_-last_speed;
		}
		if(target_speed_>=ModelParams::VEL_MAX) {
			target_speed_ = ModelParams::VEL_MAX;
			last_acc_=target_speed_-last_speed;
		}

		cout<<"target_speed = "<<target_speed_<<endl;

		publishImitationData(curr_state, safeAction, step_reward, target_speed_);

		b_update_il = true; // imitation learning: renable data update for imitation data

		delete pb;

		worldModel.ped_sim_[0] -> OutputTime();
}

double Controller::StepReward(PomdpState& state, int action){
	double reward=0;
	if (worldModel.isLocalGoal(state)) {
        reward = ModelParams::GOAL_REWARD;
		return reward;
	}

	//cout<< "   + goal reward:"<< reward << endl;
 	// Safety control: collision; Terminate upon collision
    if(state.car.vel > 0.001 && worldModel.inRealCollision(state) ) { /// collision occurs only when car is moving
		reward = ModelParams::CRASH_PENALTY * (state.car.vel * state.car.vel + ModelParams::REWARD_BASE_CRASH_VEL);  //, closest_ped, closest_dist);
		if(action == PedPomdp::ACT_DEC) reward += 0.1;
		return reward;
	}
	//cout<< "   + collision and ACT_DEC reward:"<< reward << endl;
	// Smoothness control
	reward +=  (action == PedPomdp::ACT_DEC || action == PedPomdp::ACT_ACC) ? -0.1 : 0.0;
	//cout<< "   + action reward:"<< reward << endl;
	// Speed control: Encourage higher speed
	reward += ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX;
	//cout<< "   + vel reward:"<< reward << " REWARD_FACTOR_VEL, car vel, VEL_MAX:" <<ModelParams::REWARD_FACTOR_VEL<<","<< state.car.vel <<","<<ModelParams::VEL_MAX<< endl;

	return reward;
}

void Controller::publishPedsPrediciton() {
    vector<PedStruct> peds = worldBeliefTracker.predictPeds();
    sensor_msgs::PointCloud pc;
    pc.header.frame_id=global_frame_id;
    pc.header.stamp=ros::Time::now();
    for(const auto& ped: peds) {
        geometry_msgs::Point32 p;
        p.x = ped.pos.x;
        p.y = ped.pos.y;
        p.z = 1.0;
        pc.points.push_back(p);
    }
    pedPredictionPub_.publish(pc);
}


void Controller::publishMarker(int id,PedBelief & ped)
{
	//cout<<"belief vector size "<<belief.size()<<endl;
	std::vector<double> belief = ped.prob_goals;
	uint32_t shape = visualization_msgs::Marker::CUBE;
    uint32_t shape_text=visualization_msgs::Marker::TEXT_VIEW_FACING;
	for(int i=0;i<belief.size();i++)
	{
		visualization_msgs::Marker marker;
		visualization_msgs::Marker marker_text;

		marker.header.frame_id=global_frame_id;
		marker.header.stamp=ros::Time::now();
		marker.ns="basic_shapes";
		marker.id=id*ped.prob_goals.size()+i;
		marker.type=shape;
		marker.action = visualization_msgs::Marker::ADD;

		marker_text.header.frame_id=global_frame_id;
		marker_text.header.stamp=ros::Time::now();
		marker_text.ns="basic_shapes";
		marker_text.id=id*ped.prob_goals.size()+i+1000;
		marker_text.type=shape_text;
		marker_text.action = visualization_msgs::Marker::ADD;


		double px=0,py=0;
		px=ped.pos.x;
		py=ped.pos.y;
		marker_text.pose.position.x = px;
		marker_text.pose.position.y = py;
		marker_text.pose.position.z = 0.5;
		marker_text.pose.orientation.x = 0.0;
		marker_text.pose.orientation.y = 0.0;
		marker_text.pose.orientation.z = 0.0;
		marker_text.pose.orientation.w = 1.0;
	//	cout<<"belief entries "<<px<<" "<<py<<endl;
		// Set the scale of the marker -- 1x1x1 here means 1m on a side
		//if(marker.scale.y<0.2) marker.scale.y=0.2;
		marker_text.scale.z = 1.0;
		//
		// Set the color -- be sure to set alpha to something non-zero!
		//marker.color.r = 0.0f;
		//marker.color.g = 1.0f;
		//marker.color.b = 0.0f;
		//marker.color.a = 1.0;
		marker_text.color.r = 0.0;
        marker_text.color.g = 0.0;
		marker_text.color.b = 0.0;//marker.lifetime = ros::Duration();
		marker_text.color.a = 1.0;
        marker_text.text = to_string(ped.id); 

		ros::Duration d1(1/control_freq);
		marker_text.lifetime=d1;


		px=0,py=0;
		px=ped.pos.x;
		py=ped.pos.y;
		marker.pose.position.x = px+i*0.7;
		marker.pose.position.y = py+belief[i]*2;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
	//	cout<<"belief entries "<<px<<" "<<py<<endl;
		// Set the scale of the marker -- 1x1x1 here means 1m on a side
		marker.scale.x = 0.5;
		marker.scale.y = belief[i]*4;
		//if(marker.scale.y<0.2) marker.scale.y=0.2;
		marker.scale.z = 0.2;
		//
		// Set the color -- be sure to set alpha to something non-zero!
		//marker.color.r = 0.0f;
		//marker.color.g = 1.0f;
		//marker.color.b = 0.0f;
		//marker.color.a = 1.0;
		marker.color.r = marker_colors[i][0];
        marker.color.g = marker_colors[i][1];
		marker.color.b = marker_colors[i][2];//marker.lifetime = ros::Duration();
		marker.color.a = 1.0;

		ros::Duration d2(1/control_freq);
		marker.lifetime=d2;
		markers.markers.push_back(marker);
        //markers.markers.push_back(marker_text);
	}
}

void Controller::publishBelief()
{
	//vector<vector<double> > ped_beliefs=RealSimulator->GetBeliefVector(solver->root_->particles());	
	//cout<<"belief vector size "<<ped_beliefs.size()<<endl;
	int i=0;
	ped_is_despot::peds_believes pbs;	
	for(auto & kv: worldBeliefTracker.peds)
	{
		publishMarker(i++,kv.second);
		ped_is_despot::ped_belief pb;
		PedBelief belief = kv.second;
		pb.ped_x=belief.pos.x;
		pb.ped_y=belief.pos.y;
		pb.ped_id=belief.id;
		for(auto & v : belief.prob_goals)
			pb.belief_value.push_back(v);
		pbs.believes.push_back(pb);
	}
	pbs.cmd_vel=worldStateTracker.carvel;
	pbs.robotx=worldStateTracker.carpos.x;
	pbs.roboty=worldStateTracker.carpos.y;
	believesPub_.publish(pbs);
	markers_pub.publish(markers);
	markers.markers.clear();
}


void Controller::publishImitationData(PomdpState& planning_state, int safeAction, float reward, float cmd_vel)
{
	// car for publish
	/*peds_unity_system::car_info p_car;
	p_car.car_pos.x = car_pos_x
	p_car.car_pos.y = car_pos_y
	p_car.car_pos.z = 0
	p_car.car_yaw = car_yaw

	p_IL_data.past_car = p_IL_data.cur_car;
    p_IL_data.cur_car = p_car;
*/
	// ped_for publish
	peds_unity_system::peds_info p_ped;
	for (int i = 0; i < planning_state.num; i++){
		peds_unity_system::ped_info ped;
        ped.ped_id = planning_state.peds[i].id;
        ped.ped_goal_id = planning_state.peds[i].goal;
        ped.ped_speed = 1.2;
        ped.ped_pos.x = planning_state.peds[i].pos.x;
        ped.ped_pos.y = planning_state.peds[i].pos.y;
        ped.ped_pos.z = 0;
        p_ped.peds.push_back(ped);
    }

    p_IL_data.past_peds = p_IL_data.cur_peds;
    p_IL_data.cur_peds = p_ped;

	// belief for pushlish
	int i=0;
	ped_is_despot::peds_believes pbs;	
	for(auto & kv: worldBeliefTracker.peds)
	{
		publishMarker(i++,kv.second);
		ped_is_despot::ped_belief pb;
		PedBelief belief = kv.second;
		pb.ped_x=belief.pos.x;
		pb.ped_y=belief.pos.y;
		pb.ped_id=belief.id;
		for(auto & v : belief.prob_goals)
			pb.belief_value.push_back(v);
		pbs.believes.push_back(pb);
	}
	pbs.cmd_vel=worldStateTracker.carvel;
	pbs.robotx=worldStateTracker.carpos.x;
	pbs.roboty=worldStateTracker.carpos.y;

	p_IL_data.believes = pbs.believes;

	// path for publish
	//p_path is recorded in the path callback function

	// action for publish
	geometry_msgs::Twist p_action_reward;

    p_IL_data.action_reward.linear.x=safeAction;
    p_IL_data.action_reward.linear.y=reward;
    p_IL_data.action_reward.linear.z=cmd_vel;
    //p_action_reward.angular.x = steering_cmd_;


    IL_pub.publish(p_IL_data);

}
