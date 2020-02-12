#include <csignal>
#include <iostream>

#include <despot/core/globals.h>

#include "param.h"
#include "coord.h"

#include <ros/ros.h>
#include <msg_builder/PomdpCmd.h>
#include <msg_builder/car_info.h>
#include <std_msgs/Float32.h>

int tick = 0;
double pub_freq = 12;
float time_scale = 1.0;

const double acceleration = 4.8;
const double deceleration = 4.7;

sig_atomic_t emergency_break = 0;

using namespace std;

void sig_break(int param) {
	emergency_break = 1;
	std::cerr << "Emergency break!" << std::endl;
}

class VelPublisher {
public:
	VelPublisher() :
			contoller_vel(0), target_vel(0), target_acc(0), init_controller_vel(0) {
		ros::NodeHandle n("~");
		n.param<float>("time_scale", time_scale, 1.0);

		cout << "=> VelPublisher params: " << endl;
		cout << "=> time_scale: " << time_scale << endl;
		cout << "=> vel pub pub_freq: " << pub_freq << endl;

		target_steer = 0;
		input_data_ready = false;
	}

	void Spin() {
		ros::NodeHandle nh;

		action_sub = nh.subscribe("cmd_action_pomdp", 1,
				&VelPublisher::ActionCallBack, this);
		ego_sub = nh.subscribe("ego_state", 1, &VelPublisher::EgostateCallback,
				this);

		ros::Timer timer = nh.createTimer(
				ros::Duration(1 / pub_freq / time_scale),
				&VelPublisher::PublishSpeed, this);

		cmd_speed_pub = nh.advertise<std_msgs::Float32>("pomdp_cmd_speed", 1);
		cmd_steer_pub = nh.advertise<std_msgs::Float32>("pomdp_cmd_steer", 1);
		cmd_accel_pub = nh.advertise<std_msgs::Float32>("pomdp_cmd_accel", 1);
		ros::spin();
	}

	void ActionCallBack(msg_builder::PomdpCmd pomdp_vel) {
		if (pomdp_vel.target_speed == -1) {
			contoller_vel = 0.0;
			target_vel = 0.0;
			target_steer = 0.0;
			return;
		}

		target_vel = pomdp_vel.target_speed;
		contoller_vel = pomdp_vel.cur_speed;
		target_acc = pomdp_vel.acc;
		target_steer = pomdp_vel.steer;

		if (target_vel <= 0.0001) {

			target_vel = 0.0;
		}

		input_data_ready = true;

		cout << "VelPublisher get current vel from topic: " << contoller_vel << endl;
	}

	void EgostateCallback(msg_builder::car_info car) {
		car_state_ = car;
	}

	double CalPubAcc() {
		if (emergency_break)
			return -1;

		double throttle = 0.0;
		double real_vel = car_state_.car_speed;

		if (target_vel < real_vel + 0.02 && target_vel > real_vel - 0.02) {
			throttle = 0.025; // maintain cur vel
		} else if (target_vel >= real_vel + 0.02) {
			throttle = (target_vel - real_vel - 0.02) * 1.0;
			throttle = max(min(0.55, throttle), 0.025);
		} else if (target_vel < real_vel - 0.05) {
			throttle = 0.0;
		} else {
			throttle = (target_vel - real_vel) * 3.0;
			throttle = max(-1.0, throttle);
		}

		return throttle;
	}

	double CalPubSteer() {
		return target_steer / (car_state_.max_steer_angle / 180.0 * M_PI);
	}

	double CalPubSpeed() {
		return target_vel;
	}

	void PublishSpeed(const ros::TimerEvent& event) {
		if (!input_data_ready)
			return;

		double delta = acceleration / pub_freq;
		if (target_vel > contoller_vel + delta) {
			double delta = acceleration / pub_freq;
			contoller_vel += delta;
		} else if (target_vel < contoller_vel - delta) {
			double delta = deceleration / pub_freq;
			contoller_vel -= delta;
		} else
			contoller_vel = target_vel;

		std_msgs::Float32 speed_topic;
		speed_topic.data = CalPubSpeed(); // debugging
		cmd_speed_pub.publish(speed_topic);

		std_msgs::Float32 acc_topic;
		acc_topic.data = CalPubAcc(); // debugging
		cmd_accel_pub.publish(acc_topic);

		std_msgs::Float32 steer_topic;
		steer_topic.data = CalPubSteer(); // debugging
		cmd_steer_pub.publish(steer_topic);
	}

public:

	bool input_data_ready;
	double contoller_vel, init_controller_vel;
	double target_vel, target_acc, target_steer;

	ros::Subscriber action_sub, ego_sub;
	ros::Publisher cmd_speed_pub, cmd_accel_pub, cmd_steer_pub;

	msg_builder::car_info car_state_;
};

int main(int argc, char**argv) {
	ros::init(argc, argv, "vel_publisher");
	signal(SIGUSR1, sig_break);
	VelPublisher velpub;
	velpub.Spin();
	return 0;
}
