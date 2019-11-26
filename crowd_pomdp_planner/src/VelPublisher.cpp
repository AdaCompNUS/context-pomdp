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
		n.param("use_drivenet", b_use_drive_net_, 0);
		n.param<std::string>("drivenet_mode", drive_net_mode, "action");
		n.param<float>("time_scale", time_scale, 1.0);

		cout << "=> VelPublisher params: " << endl;

		cout << "=> use drive_net: " << b_use_drive_net_ << endl;
		cout << "=> drive_net mode: " << drive_net_mode << endl;
		cout << "=> time_scale: " << time_scale << endl;
		cout << "=> vel pub pub_freq: " << pub_freq << endl;

		steering = 0;
		input_data_ready = false;
	}

	void spin() {
		ros::NodeHandle nh;

		if (b_use_drive_net_ == despot::IMITATION
				& drive_net_mode == "action") {
			action_sub = nh.subscribe("cmd_vel_drive_net", 1,
					&VelPublisher::actionCallBack, this);
		} else if (b_use_drive_net_ == despot::IMITATION
				& drive_net_mode == "vel") {
			vel_sub = nh.subscribe("cmd_vel_drive_net", 1,
					&VelPublisher::velCallBack, this);
			// pomdp offers no steering signal
		} else if (b_use_drive_net_ == despot::IMITATION
				& drive_net_mode == "steer") {
			steer_sub = nh.subscribe("cmd_vel_drive_net", 1,
					&VelPublisher::steerCallBack, this);
			vel_sub = nh.subscribe("cmd_vel_pomdp", 1,
					&VelPublisher::velCallBack, this);
		} else if (b_use_drive_net_ == despot::NO
				|| b_use_drive_net_ == despot::LETS_DRIVE
				|| b_use_drive_net_ == despot::JOINT_POMDP
				|| b_use_drive_net_ == despot::ROLL_OUT)
			action_sub = nh.subscribe("cmd_vel_pomdp", 1,
					&VelPublisher::actionCallBack, this);

		speedSub = nh.subscribe("odom", 1, &VelPublisher::odomCallback, this);
		ego_sub = nh.subscribe("ego_state", 1, &VelPublisher::egostateCallback,
				this);

		ros::Timer timer = nh.createTimer(
				ros::Duration(1 / pub_freq / time_scale),
				&VelPublisher::publishSpeed, this);
		cmd_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
		cmd_steer_pub = nh.advertise<std_msgs::Float32>("pomdp_cmd_steer", 1);
		cmd_accel_pub = nh.advertise<std_msgs::Float32>("pomdp_cmd_accel", 1);
		ros::spin();
	}

	virtual void actionCallBack(geometry_msgs::TwistConstPtr pomdp_vel) = 0;

	virtual void steerCallBack(geometry_msgs::TwistConstPtr pomdp_vel) = 0;

	virtual void velCallBack(geometry_msgs::TwistConstPtr pomdp_vel) = 0;

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

	void _publishSpeed() {
		geometry_msgs::Twist cmd;

		cmd.angular.z = steering; //0;
		cmd.linear.x = emergency_break ? 0 : curr_vel;
		double pub_acc = cal_pub_acc();
		cmd.linear.y = pub_acc; // target_acc;
		cmd_pub.publish(cmd);

		std_msgs::Float32 acc_topic;
		acc_topic.data = pub_acc; // debugging
		cmd_accel_pub.publish(acc_topic);

		std_msgs::Float32 steer_topic;
		steer_topic.data = cal_pub_steer(); // debugging
		cmd_steer_pub.publish(steer_topic);
	}

	bool input_data_ready;
	double curr_vel, real_vel, target_vel, init_curr_vel, steering;
	double target_acc;
	int b_use_drive_net_;

	std::string drive_net_mode;
	ros::Subscriber vel_sub, steer_sub, action_sub, odom_sub, speedSub, ego_sub;
	ros::Publisher cmd_pub, cmd_accel_pub, cmd_steer_pub;

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
	void actionCallBack(geometry_msgs::TwistConstPtr pomdp_vel) {
		if (pomdp_vel->linear.x == -1) {
			curr_vel = 0.0;
			target_vel = 0.0;
			steering = 0.0;
			return;
		}
		target_vel = pomdp_vel->linear.x;
		curr_vel = pomdp_vel->linear.y;
		real_vel = pomdp_vel->linear.y;
		target_acc = pomdp_vel->linear.z;
		steering = pomdp_vel->angular.z;

		if (target_vel <= 0.0001) {

			target_vel = 0.0;
		}

		input_data_ready = true;

		cout << "VelPublisher get current vel from topic: " << curr_vel << endl;
	}

	void velCallBack(geometry_msgs::TwistConstPtr pomdp_vel) {
		if (pomdp_vel->linear.x == -1) {
			curr_vel = 0.0;
			target_vel = 0.0;
			return;
		}

		target_vel = pomdp_vel->linear.x;
		curr_vel = pomdp_vel->linear.y;

		if (target_vel <= 0.0001) {

			target_vel = 0.0;
		}
	}

	void steerCallBack(geometry_msgs::TwistConstPtr pomdp_vel) {
		if (pomdp_vel->linear.x == -1) {
			steering = 0.0;
			return;
		}

		steering = pomdp_vel->angular.z;
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
