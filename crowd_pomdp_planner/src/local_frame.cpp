#include <geometry_msgs/Point32.h>
#include <msg_builder/ped_local_frame.h>
#include <msg_builder/ped_local_frame_vector.h>
#include <msg_builder/pedestrian.h>
#include <msg_builder/pedestrian_array.h>
#include <ros/console.h>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <ros/subscriber.h>
#include <ros/time.h>
#include <ros/timer.h>
#include <rosconsole/macros_generated.h>
#include <std_msgs/Header.h>
#include <stddef.h>
#include <tf/exceptions.h>
#include <tf/LinearMath/Transform.h>
#include <tf/LinearMath/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include "param.h"

using namespace std;
struct PedTransforms {
	tf::Transform transform;
	int label;
	ros::Time last_update;
};

class LocalFrame {
public:
	LocalFrame();
	~LocalFrame();

private:
	void PublishTransform(const ros::TimerEvent& event);
	void AgentArrayCallback(msg_builder::pedestrian_arrayConstPtr ped_array);
	bool GetObjectPose(const string& target_frame,
			tf::Stamped<tf::Pose>& in_pose,
			tf::Stamped<tf::Pose>& out_pose) const;
	void PublishLocalFrames();
	void LoadFilter();
	bool Filtered(double, double);

	vector<PedTransforms> ped_transforms_;
	tf::TransformBroadcaster br_;
	tf::TransformListener tf_;
	ros::Timer timer_;
	ros::Subscriber ped_sub_;
	ros::Publisher local_pub_;
	string global_frame_;
	double threshold_, offsetx_, offsety_;
	vector<pair<double, double>> path_record;
	vector<pair<double, double>> object_filter;
};

LocalFrame::LocalFrame() {
	ros::NodeHandle nh;
	ped_sub_ = nh.subscribe("pedestrian_array", 1, &LocalFrame::AgentArrayCallback,
			this);
	local_pub_ = nh.advertise<msg_builder::ped_local_frame_vector>(
			"ped_local_frame_vector", 1);

	LoadFilter();
	ros::NodeHandle n("~");
	bool simulation;
	n.param("simulation", simulation, false);
	ModelParams::InitParams(simulation);

	n.param("global_frame", global_frame_,
			ModelParams::ROS_NS + string("/map"));
	n.param("threshold", threshold_, 3.0);
	n.param("offsetx", offsetx_, 0.0);
	n.param("offsety", offsety_, 3.0);
	ros::spin();
}

LocalFrame::~LocalFrame() {
	cout << "entering destructor" << endl;
	//write the path_record to the file
	std::ofstream path_out("curr_real_path");
	path_out << path_record.size() << endl;
	for (int i = 0; i < path_record.size(); i++) {
		path_out << int(path_record[i].first) << " "
				<< int(path_record[i].second) << endl;
	}
}

void LocalFrame::LoadFilter() {
	double fx, fy;
	ifstream fin("momdp_object_filter.data");
	while (fin >> fx) {
		fin >> fy;
		object_filter.push_back(make_pair(fx, fy));
	}
}

bool LocalFrame::Filtered(double x, double y) {
	double fx, fy;
	for (vector<pair<double, double>>::iterator it = object_filter.begin();
			it != object_filter.end(); ++it) {
		fx = it->first;
		fy = it->second;
		if (fabs(x - fx) < 0.3 && fabs(y - fy) < 0.3)
			return true;
	}
	return false;
}

void LocalFrame::AgentArrayCallback(msg_builder::pedestrian_arrayConstPtr ped_array) {
	msg_builder::ped_local_frame_vector plf_vector;
	for (size_t i = 0; i < ped_array->pd_vector.size(); i++) {
		msg_builder::ped_local_frame plf;
		plf.header.stamp = ped_array->header.stamp;

		plf.header.frame_id = ModelParams::ROS_NS + "/map";

		tf::Stamped<tf::Pose> in_pose, out_pose;

		//start with pedestrian. no interest on the orientation of ped for now
		in_pose.setIdentity();
		in_pose.frame_id_ = ped_array->header.frame_id;

		geometry_msgs::Point32 ped_point =
				ped_array->pd_vector[i].global_centroid;
		in_pose.setOrigin(
				tf::Vector3(ped_point.x, ped_point.y, ped_point.z));
		if (!GetObjectPose(plf.header.frame_id, in_pose, out_pose))
			continue;
		plf.ped_id = ped_array->pd_vector[i].object_label;
		plf.ped_pose.x = out_pose.getOrigin().getX();
		plf.ped_pose.y = out_pose.getOrigin().getY();
		plf.ped_pose.z = out_pose.getOrigin().getZ();

		//then update the robot pose with the same frame
		in_pose.setIdentity();
		in_pose.frame_id_ = ModelParams::ROS_NS
				+ ModelParams::LASER_FRAME;
		if (!GetObjectPose(plf.header.frame_id, in_pose, out_pose))
			continue;

		plf.rob_pose.x = out_pose.getOrigin().getX();
		plf.rob_pose.y = out_pose.getOrigin().getY();
		plf.rob_pose.z = out_pose.getOrigin().getZ();
		plf_vector.ped_local.push_back(plf);

	}

	local_pub_.publish(plf_vector);
}

bool LocalFrame::GetObjectPose(const string& target_frame,
		tf::Stamped<tf::Pose>& in_pose, tf::Stamped<tf::Pose>& out_pose) const {
	out_pose.setIdentity();

	try {
		tf_.transformPose(target_frame, in_pose, out_pose);
	} catch (tf::LookupException& ex) {
		ROS_ERROR("No Transform available Error: %s\n", ex.what());
		return false;
	} catch (tf::ConnectivityException& ex) {
		ROS_ERROR("Connectivity Error: %s\n", ex.what());
		return false;
	} catch (tf::ExtrapolationException& ex) {
		ROS_ERROR("Extrapolation Error: %s\n", ex.what());
		return false;
	}
	return true;
}

void LocalFrame::PublishTransform(const ros::TimerEvent& event) {
	tf::Stamped<tf::Pose> in_pose, out_pose;
	in_pose.setIdentity();
	in_pose.frame_id_ = ModelParams::ROS_NS + ModelParams::LASER_FRAME;
	if (!GetObjectPose(ModelParams::ROS_NS + "/map", in_pose, out_pose)) {
		cerr << "transform error within local_frame" << endl;
	} else {
		if (path_record.size() == 0) {
			double now_x, now_y;
			now_x = out_pose.getOrigin().getX();
			now_y = out_pose.getOrigin().getY();
			path_record.push_back(make_pair(now_x, now_y));
		} else {
			double last_x, last_y, now_x, now_y;
			last_x = path_record[path_record.size() - 1].first;
			last_y = path_record[path_record.size() - 1].second;
			now_x = out_pose.getOrigin().getX();
			now_y = out_pose.getOrigin().getY();
			double dx = fabs(last_x - now_x);
			double dy = fabs(last_y - now_y);
			if (sqrt(dx * dx + dy * dy) > 0.1) //find the next record
				path_record.push_back(make_pair(now_x, now_y));
		}
	}
}

int main(int argc, char** argv) {
	ros::init(argc, argv, "local_frame");
	LocalFrame *lf = new LocalFrame();
	delete lf;
	return 0;
}
