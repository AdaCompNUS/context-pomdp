#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
////#include <sensing_on_road/pedestrian_laser_batch.h>
#include <ped_is_despot/ped_local_frame_vector.h>
////#include <sensing_on_road/pedestrian_vision_batch.h>
#include <fstream>
#include "param.h"

#include <ped_is_despot/pedestrian_array.h>
#include <ped_is_despot/clusters.h>

using namespace std;
struct ped_transforms
{
    tf::Transform transform;
    int label;
    ros::Time last_update;
};

class local_frame
{
public:
    local_frame();
    ~local_frame();

private:
    void publishTransform(const ros::TimerEvent& event);
    //void pedCallback(sensing_on_road::pedestrian_vision_batchConstPtr ped_batch);
    void pedCallback(ped_is_despot::pedestrian_arrayConstPtr ped_array);
    bool getObjectPose(const string& target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const;
	void PublishLocalFrames();
	void loadFilter();
	bool filtered(double,double);
    vector<ped_transforms> ped_transforms_;
    tf::TransformBroadcaster br_;
    tf::TransformListener tf_;
    ros::Timer timer_;
    ros::Subscriber ped_sub_;
    ros::Publisher local_pub_;
    string global_frame_;
    double threshold_, offsetx_, offsety_;
	vector<pair<double,double>> path_record;
	vector<pair<double,double>> object_filter;
};

local_frame::local_frame()
{
    ros::NodeHandle nh;
    ped_sub_ = nh.subscribe("pedestrian_array", 1, &local_frame::pedCallback, this);
    local_pub_ = nh.advertise<ped_is_despot::ped_local_frame_vector>("ped_local_frame_vector", 1);

	loadFilter();
    ros::NodeHandle n("~");
    //n.param("global_frame", global_frame_, string("/odom"));
	
	bool simulation;
	n.param("simulation", simulation, false);
	ModelParams::init_params(simulation);

	n.param("global_frame", global_frame_, ModelParams::rosns + string("/map"));
    n.param("threshold", threshold_, 3.0);
    n.param("offsetx", offsetx_, 0.0);
    n.param("offsety", offsety_, 3.0);
    //timer_ = nh.createTimer(ros::Duration(0.01), &local_frame::publishTransform, this);
    ros::spin();
}
local_frame::~local_frame()
{
	cout<<"entering destructor"<<endl;
	//write the path_record to the file
	ofstream path_out("curr_real_path");
	path_out<<path_record.size()<<endl;
	for(int i=0;i<path_record.size();i++)
	{
		path_out<<int(path_record[i].first)<<" "<<int(path_record[i].second)<<endl;
	}
}


/*
double object_filter[20][20]={
	{117.609802246,120.935005188},
	{120.286,117.424}

106.815246582,
};*/
void local_frame::loadFilter()
{
	double fx,fy;
	ifstream fin("momdp_object_filter.data");
	while(fin>>fx)
	{
		fin>>fy;
		object_filter.push_back(make_pair(fx,fy));
	}
}
bool local_frame::filtered(double x,double y)
{
	double fx,fy;
	for (vector<pair<double,double>>::iterator it = object_filter.begin() ; it != object_filter.end(); ++it)
	{
		//if(fabs(x-object_filter[k][0])<0.3&&fabs(y-object_filter[k][1])<0.3)
		fx=it->first;
		fy=it->second;
		if(fabs(x-fx)<0.3&&fabs(y-fy)<0.3)
		{
			//filterd
			return true;
		}
	}
	//cout<<"number of filter entry "<<ct<<endl;
	return false;
}

/*void local_frame::pedCallback(sensing_on_road::pedestrian_vision_batchConstPtr ped_batch)
{
    ped_is_despot::ped_local_frame_vector plf_vector;
    for(size_t i=0;i<ped_batch->pd_vector.size();i++)
    {
        if(true)
        {
            //transform found, just update the position
            //if(ped_batch->pd_vector[i].confidence > 0.01)
            {
				
                ped_is_despot::ped_local_frame plf;
                plf.header.stamp = ped_batch->header.stamp;
				
				plf.header.frame_id = ModelParams::rosns + "/map";

                tf::Stamped<tf::Pose> in_pose, out_pose;

                //start with pedestrian. no interest on the orientation of ped for now
                in_pose.setIdentity();
                in_pose.frame_id_ = ped_batch->header.frame_id;
                geometry_msgs::Point32 ped_point = ped_batch->pd_vector[i].cluster.centroid;
                in_pose.setOrigin(tf::Vector3(ped_point.x, ped_point.y, ped_point.z));
                if(!getObjectPose(plf.header.frame_id, in_pose, out_pose)) continue;
				plf.ped_id = ped_batch->pd_vector[i].object_label;
                plf.ped_pose.x = out_pose.getOrigin().getX();
                plf.ped_pose.y = out_pose.getOrigin().getY();
                plf.ped_pose.z = out_pose.getOrigin().getZ();
				
                //then update the robot pose with the same frame
                in_pose.setIdentity();
                in_pose.frame_id_ = ModelParams::rosns + ModelParams::laser_frame;
                if(!getObjectPose(plf.header.frame_id, in_pose, out_pose)) continue;

                plf.rob_pose.x = out_pose.getOrigin().getX();
                plf.rob_pose.y = out_pose.getOrigin().getY();
                plf.rob_pose.z = out_pose.getOrigin().getZ();
                plf_vector.ped_local.push_back(plf);
            }
        }
        else
        {}
    }		
		
    local_pub_.publish(plf_vector);
}*/

void local_frame::pedCallback(ped_is_despot::pedestrian_arrayConstPtr ped_array)
{
    ped_is_despot::ped_local_frame_vector plf_vector;
    for(size_t i=0;i<ped_array->pd_vector.size();i++)
    {
        if(true)
        {
            //transform found, just update the position
            //if(ped_batch->pd_vector[i].confidence > 0.01)
            {
				
                ped_is_despot::ped_local_frame plf;
                plf.header.stamp = ped_array->header.stamp;
				
				plf.header.frame_id = ModelParams::rosns + "/map";

                tf::Stamped<tf::Pose> in_pose, out_pose;

                //start with pedestrian. no interest on the orientation of ped for now
                in_pose.setIdentity();
                in_pose.frame_id_ = ped_array->header.frame_id;
                //geometry_msgs::Point32 ped_point = ped_array->pd_vector[i].cluster.centroid;
                geometry_msgs::Point32 ped_point = ped_array->pd_vector[i].global_centroid;
                in_pose.setOrigin(tf::Vector3(ped_point.x, ped_point.y, ped_point.z));
                if(!getObjectPose(plf.header.frame_id, in_pose, out_pose)) continue;
				plf.ped_id = ped_array->pd_vector[i].object_label;
                plf.ped_pose.x = out_pose.getOrigin().getX();
                plf.ped_pose.y = out_pose.getOrigin().getY();
                plf.ped_pose.z = out_pose.getOrigin().getZ();
				
                //then update the robot pose with the same frame
                in_pose.setIdentity();
                in_pose.frame_id_ = ModelParams::rosns + ModelParams::laser_frame;
                if(!getObjectPose(plf.header.frame_id, in_pose, out_pose)) continue;

                plf.rob_pose.x = out_pose.getOrigin().getX();
                plf.rob_pose.y = out_pose.getOrigin().getY();
                plf.rob_pose.z = out_pose.getOrigin().getZ();
                plf_vector.ped_local.push_back(plf);
            }
        }
        else
        {}
    }		
		
    local_pub_.publish(plf_vector);
}

bool local_frame::getObjectPose(const string& target_frame, tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const
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

void local_frame::publishTransform(const ros::TimerEvent& event)
{

	
        tf::Stamped<tf::Pose> in_pose, out_pose;
		in_pose.setIdentity();
		in_pose.frame_id_ = ModelParams::rosns + ModelParams::laser_frame;
		if(!getObjectPose(ModelParams::rosns + "/map", in_pose, out_pose)) {
			cerr<<"transform error within local_frame"<<endl;
		} else {
			if(path_record.size()==0)
			{
				double now_x,now_y;
				now_x=out_pose.getOrigin().getX();
				now_y=out_pose.getOrigin().getY();
				path_record.push_back(make_pair(now_x,now_y));
			}
			else 
			{
				double last_x,last_y,now_x,now_y;
				last_x=path_record[path_record.size()-1].first;
				last_y=path_record[path_record.size()-1].second;
				now_x=out_pose.getOrigin().getX();
				now_y=out_pose.getOrigin().getY();
				double dx=fabs(last_x-now_x);
				double dy=fabs(last_y-now_y);
				if(sqrt(dx*dx+dy*dy)>0.1) //find the next record
				{
					path_record.push_back(make_pair(now_x,now_y));	
				}
			}

		}
		
		
	/*
	for(size_t i=0; i<ped_transforms_.size(); i++)
    {
        stringstream frame_id;
        frame_id<<"ped_"<<ped_transforms_[i].label;
        br_.sendTransform(tf::StampedTransform(ped_transforms_[i].transform, ros::Time::now(), global_frame_, frame_id.str()));
    }
	*/
}

/*
void local_frame::PublishLocalFrames()
{
	ped_is_despot::ped_local_frame_vector plf_vector;
	Car car=world.GetCarPose();
	for(int i=0;i<world.NumPedInView();i++)
	{
		Pedestrian ped=world.GetPedPose(i);	
		ped_is_despot::ped_local_frame plf;
		plf.ped_id=ped.id;
		plf.ped_pose.x=ped.w;
		plf.ped_pose.y=ped.h;
		plf.rob_pose.x=car.w;
		plf.rob_pose.y=car.h;
		plf_vector.push_back(plf);	
	}
	local_pub_.publish(plf_vector);
	world.OneStep();
	
}*/
int main(int argc, char** argv){
    ros::init(argc, argv, "local_frame");
    local_frame *lf = new local_frame();
	//PublishLocalFrames();   //this is for simulation
    //
	delete lf;
    return 0;
};
