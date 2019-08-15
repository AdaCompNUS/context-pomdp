#!/usr/bin/env python2

import sys, os, pdb
import numpy

from util import * 
from path_smoothing import distance

import rospy
from peds_unity_system.msg import peds_car_info as PedsCarInfo
from peds_unity_system.msg import car_info as CarInfo # panpan
from peds_unity_system.msg import peds_info as PedsInfo
from peds_unity_system.msg import ped_info as PedInfo
from carla_connector2.msg import TrafficAgentArray
from carla_connector2.msg import TrafficAgent
from geometry_msgs.msg import Twist

freq = 10.0
acc = 1.5
delta = acc/freq
max_speed = 3.0


class SpeedController(object):
    def __init__(self):
        self.proximity = 10000000
        self.player_pos = None
        self.player_yaw = None
        self.player_vel = None
        self.peds_pos = None

        self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/agent_array", TrafficAgentArray, self.cb_peds, queue_size=1)
        rospy.Subscriber("/IL_car_info", CarInfo, self.cb_car, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / freq), self.compute_speed_and_publish)

    def cal_proximty(self):
        player_pos = self.player_pos
        self.proximity = 10000000

        for ped_pos in self.peds_pos:
            dist = distance(player_pos, ped_pos)
            if dist < self.proximity:
                self.proximity = dist

    def calculate_player_speed(self):
        self.cal_proximty()

        vel = self.player_vel
        yaw = numpy.deg2rad(self.player_yaw)
        v_2d = numpy.array([vel[0], vel[1], 0])
        forward = numpy.array([math.cos(yaw), math.sin(yaw), 0])
        speed = numpy.vdot(forward, v_2d)  # numpy.linalg.norm(v)

        return speed


    def compute_speed_and_publish(self, tick):
        if self.player_pos == None or self.peds_pos == None:
            return

        cmd = Twist();
        curr_vel = self.calculate_player_speed()
        cmd.angular.z = 0

        if self.proximity > 10:
            cmd.linear.x = curr_vel + delta;
            cmd.linear.y = acc
        elif self.proximity > 8:
            cmd.linear.x = curr_vel
            cmd.linear.y = 0
        elif self.proximity < 6:
            cmd.linear.x = curr_vel - delta;
            cmd.linear.y = -acc

        if curr_vel >2 and cmd.linear.y > 0:
            cmd.linear.x = curr_vel
            cmd.linear.y = 0

        self.pub_cmd_vel.publish(cmd);


    def cb_peds(self, msg):
        self.peds_pos = [[ped.pose.position.x, ped.pose.position.y] for ped in msg.agents]

    def cb_car(self, msg):
        self.player_pos = [msg.car_pos.x, msg.car_pos.y]
        self.player_yaw = msg.car_yaw
        self.player_vel = [msg.car_vel.x, msg.car_vel.y]

if __name__ == '__main__':
    rospy.init_node('speed_controller')
    speed_controller = SpeedController()
    rospy.spin() 

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        speed_controller.update()
        rate.sleep()
