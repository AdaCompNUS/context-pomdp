#!/usr/bin/env python2

import sys, os, pdb
import numpy as np

from util import * 
from path_smoothing import distance

import rospy
from std_msgs.msg import Float32
from msg_builder.msg import peds_car_info as PedsCarInfo
from msg_builder.msg import car_info as CarInfo # panpan
from msg_builder.msg import peds_info as PedsInfo
from msg_builder.msg import ped_info as PedInfo
from msg_builder.msg import TrafficAgentArray
from msg_builder.msg import TrafficAgent
from geometry_msgs.msg import Twist

freq = 10.0
acc = 1.5
delta = acc/freq
max_speed = 5.0

class SpeedController(object):
    def __init__(self):
        self.proximity = 10000000
        self.player_pos = None
        self.player_yaw = None
        self.player_vel = None
        self.peds_pos = []

        self.cmd_speed_pub = rospy.Publisher("/cmd_speed", Float32, queue_size=1)
        self.cmd_accel_pub = rospy.Publisher("/cmd_accel", Float32, queue_size=1)
        rospy.Subscriber("/agent_array", TrafficAgentArray, self.cb_peds, queue_size=1)
        rospy.Subscriber("/ego_state", CarInfo, self.cb_car, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / freq), self.compute_speed_and_publish)

    def cal_proximty(self):
        player_pos = self.player_pos
        player_yaw = self.player_yaw

        self.proximity = 10000000

        for ped_pos in self.peds_pos:
            if in_front(player_pos, player_yaw, ped_pos):
                dist = distance(player_pos, ped_pos)
                if dist < self.proximity:
                    self.proximity = dist

    def calculate_player_speed(self):
        self.cal_proximty()

        vel = self.player_vel
        yaw = np.deg2rad(self.player_yaw)
        v_2d = np.array([vel[0], vel[1], 0])
        forward = np.array([math.cos(yaw), math.sin(yaw), 0])
        speed = np.vdot(forward, v_2d)  # np.linalg.norm(v)

        return speed


    def compute_speed_and_publish(self, tick):
        if self.player_pos is None:
            return

        curr_vel = self.calculate_player_speed()
        cmd_speed = 0
        cmd_accel = 0

        if self.proximity > 10:
            cmd_speed = curr_vel + delta;
            cmd_accel = acc
        elif self.proximity > 8:
            cmd_speed = curr_vel
            cmd_accel = 0
        elif self.proximity < 6:
            cmd_speed = curr_vel - delta;
            cmd_accel = -acc

        if curr_vel > max_speed and cmd_accel > 0:
            cmd_speed = curr_vel
            cmd_accel = 0

        self.cmd_speed_pub.publish(cmd_speed)
        self.cmd_accel_pub.publish(cmd_accel)

    def cb_peds(self, msg):
        self.peds_pos = [[ped.pose.position.x, ped.pose.position.y] for ped in msg.agents]

    def cb_car(self, msg):
        self.player_pos = [msg.car_pos.x, msg.car_pos.y]
        self.player_yaw = np.rad2deg(msg.car_yaw)
        self.player_vel = [msg.car_vel.x, msg.car_vel.y]

if __name__ == '__main__':
    rospy.init_node('speed_controller')
    speed_controller = SpeedController()
    rospy.spin() 

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        speed_controller.update()
        rate.sleep()
