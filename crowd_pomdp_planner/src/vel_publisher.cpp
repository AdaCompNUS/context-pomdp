#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>

#include <csignal>
#include <iostream>
#include "param.h"
#include "coord.h"
#include <despot/core/globals.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include "msg_builder/car_info.h"
#include <msg_builder/PomdpCmd.h>

int tick = 0;
double pub_freq = 12; //10;
float time_scale = 1.0;
//const double acceleration0 = 0.7;
//const double acceleration1 = 0.9;
const double acceleration0 = /*0.5*/4.8;
//const double acceleration1 = 0.7;
const double acceleration1 = /*1.0*/4.7;
//const double alpha0 = 1.0;
//const double alpha1 = 1.0;

sig_atomic_t emergency_break = 0;

using namespace std;

void sig_break(int param) {
	emergency_break = 1;
	std::cerr << "Emergency break!" << std::endl;
}

class VelPublisher {
public:
	VelPublisher() :
			curr_vel(0), target_vel(0) {
		ros::NodeHandle n("~");
		n.param("drive_mode", b_drive_mode_, 0);
		n.param<std::string>("cmd_mode", drive_net_mode, "action");
		n.param<float>("time_scale", time_scale, 1.0);

		cout << "=> VelPublisher params: " << endl;

		cout << "=> use drive_net: " << b_drive_mode_ << endl;
		cout << "=> drive_net mode: " << drive_net_mode << endl;
		cout << "=> time_scale: " << time_scale << endl;
		cout << "=> vel pub pub_freq: " << pub_freq << endl;

		steering = 0;
		input_data_ready = false;
	}

	void spin() {
		ros::NodeHandle nh;

		action_sub = nh.subscribe("cmd_vel_pomdp", 1, &VelPublisher::actionCallBack, this);
		odom_Sub = nh.subscribe("odom", 1, &VelPublisher::odomCallback, this);
		ego_sub = nh.subscribe("ego_state", 1, &VelPublisher::egostateCallback,
				this);

		ros::Timer timer = nh.createTimer(
				ros::Duration(1 / pub_freq / time_scale),
				&VelPublisher::publishSpeed, this);

		cmd_speed_pub = nh.advertise<std_msgs::Float32>("pomdp_cmd_speed", 1);
		cmd_steer_pub = nh.advertise<std_msgs::Float32>("pomdp_cmd_steer", 1);
		cmd_accel_pub = nh.advertise<std_msgs::Float32>("pomdp_cmd_accel", 1);
		ros::spin();
	}

	virtual void actionCallBack(msg_builder::PomdpCmd pomdp_vel) = 0;

	virtual void publishSpeed(const ros::TimerEvent& event) = 0;

	virtual void odomCallback(nav_msgs::Odometry odo) = 0;

	virtual void egostateCallback(msg_builder::car_info car) = 0;

	double cal_pub_acc() {
	if (emergency_break)
            return -1;

        double throttle = 0.0;

        if (target_vel < real_vel + 0.02 && target_vel > real_vel - 0.02 ){
            throttle = 0.025; // maintain cur vel
        } else if(target_vel >= real_vel + 0.02){
            throttle = (target_vel - real_vel - 0.02) * 1.0;
            throttle = max(min(0.55, throttle),0.025);
        } else if(target_vel < real_vel - 0.05){
            throttle = 0.0;
        } else {
            throttle = (target_vel - real_vel) * 3.0;
            throttle = max(-1.0, throttle);
        }

        return throttle;
	}

	double cal_pub_steer() {
		return steering / (car_state_.max_steer_angle / 180.0 * M_PI);
	}

	double cal_pub_speed() {
		return target_vel;
	}

	void _publishSpeed() {
		std_msgs::Float32 speed_topic;
		speed_topic.data = cal_pub_speed(); // debugging
		cmd_speed_pub.publish(speed_topic);

		std_msgs::Float32 acc_topic;
		acc_topic.data = cal_pub_acc(); // debugging
		cmd_accel_pub.publish(acc_topic);

		std_msgs::Float32 steer_topic;
		steer_topic.data = cal_pub_steer(); // debugging
		cmd_steer_pub.publish(steer_topic);
	}

	bool input_data_ready;
	double curr_vel, real_vel, target_vel, init_curr_vel, steering;
	double target_acc;
	int b_drive_mode_;

	std::string drive_net_mode;
	ros::Subscriber vel_sub, steer_sub, action_sub, odom_sub, odom_Sub, ego_sub;
	ros::Publisher cmd_speed_pub, cmd_accel_pub, cmd_steer_pub;

	msg_builder::car_info car_state_;
};

/*class VelPublisher1 : public VelPublisher {
 public:

 VelPublisher1(): VelPublisher() {}

 void velCallBack(geometry_msgs::TwistConstPtr pomdp_vel) {
 target_vel = pomdp_vel->linear.x;
 if(target_vel > curr_vel)
 curr_vel = curr_vel*(1-alpha0)+target_vel*alpha0;
 else
 curr_vel = curr_vel*(1-alpha1)+target_vel*alpha1;
 }

 void publishSpeed(const ros::TimerEvent& event) {
 _publishSpeed();
 }
 };*/

class VelPublisher2: public VelPublisher {
	void actionCallBack(msg_builder::PomdpCmd pomdp_vel) {
		if (pomdp_vel.target_speed == -1) {
			curr_vel = 0.0;
			target_vel = 0.0;
			steering = 0.0;
			return;
		}

		target_vel = pomdp_vel.target_speed;
		curr_vel = pomdp_vel.cur_speed;
		real_vel = pomdp_vel.cur_speed;
		target_acc = pomdp_vel.acc;
		steering = pomdp_vel.steer;

		if (target_vel <= 0.0001) {

			target_vel = 0.0;
		}

		input_data_ready = true;

		cout << "VelPublisher get current vel from topic: " << curr_vel << endl;
	}

	void odomCallback(nav_msgs::Odometry odo) {
		COORD velocity(odo.twist.twist.linear.x, odo.twist.twist.linear.y);
		double yaw = tf::getYaw(odo.pose.pose.orientation);
		COORD heading(cos(yaw), sin(yaw));

		real_vel = velocity.dot(heading);
		if (real_vel > ModelParams::VEL_MAX * 1.2) {
			cerr << "ERROR: Unusual car vel (too large): " << real_vel
					<< " ModelParams::VEL_MAX=" << ModelParams::VEL_MAX << endl;
			// raise(SIGABRT);
		}
	}

	void egostateCallback(msg_builder::car_info car) {
		car_state_ = car;
	}

	void publishSpeed(const ros::TimerEvent& event) {
		if (!input_data_ready)
			return;

		double delta = acceleration0 / pub_freq;
		if (target_vel > curr_vel + delta) {
			double delta = acceleration0 / pub_freq;
			curr_vel += delta;
		} else if (target_vel < curr_vel - delta) {
			double delta = acceleration1 / pub_freq;
			curr_vel -= delta;
		} else
			curr_vel = target_vel;

		_publishSpeed();
	}

//    void speedCallback(nav_msgs::Odometry odo)
//    {
//    	cout<<"update real speed "<<odo.twist.twist.linear.x<<endl;
//    	curr_vel=odo.twist.twist.linear.x;
//    }

};

int main(int argc, char**argv) {
	ros::init(argc, argv, "vel_publisher");
	signal(SIGUSR1, sig_break);
	VelPublisher2 velpub;
	velpub.spin();
	return 0;
}
