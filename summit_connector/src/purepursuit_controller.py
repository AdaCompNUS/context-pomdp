#!/usr/bin/env python2

from util import *
import carla

import numpy as np


import rospy
import csv
import math
import sys
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32
from nav_msgs.msg import Path as NavPath
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from msg_builder.msg import car_info as CarInfo  # panpan


PURSUIT_DIST = 5.0  ##1.5 for golfcart
RATIO_ANGULAR = 0.3
WHEEL_DIST = 2.66
# MAX_ANGULAR = 0.20
MAX_ANGULAR = 0.8
MAX_STEERING = 0.66
MAP_FRAME = 'map'
const_speed = 0.47
goal_reached = 0


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def norm_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def angle_diff(m1, m0):
    "m1 - m0"
    return norm_angle(m1 - m0)


class Path(object):
    def __init__(self):
        # self.path = load_path(path_fn, reverse)
        self.path = []
        rospy.Subscriber("plan", NavPath, self.cb_path, queue_size=1)

    def cb_path(self, msg):
        path = []
        for i in range(0, len(msg.poses)):
            x = msg.poses[i].pose.position.x
            y = msg.poses[i].pose.position.y
            path.append((x, y))
        self.path = path

    def nearest(self, p):
        return min(enumerate(self.path), key=lambda point: dist(point[1], p))

    def ahead(self, i, d):
        pi = self.path[i]
        while i < len(self.path) and dist(pi, self.path[i]) < d:
            i += 1
        return i

    def pursuit(self, p, d=PURSUIT_DIST):
        if self.path == []:
            return None
        ni, np = self.nearest(p)
        j = self.ahead(ni, d)
        if j >= len(self.path):
            goal_reached = 1
        return self.path[j] if j < len(self.path) else None

    def pursuit_tan(self, p, d=PURSUIT_DIST):
        if self.path == []:
            return None
        if len(self.path) == 1:
            return None
        ni, np = self.nearest(p)
        j = self.ahead(ni, d)
        if j >= len(self.path):
            return None
        if j == len(self.path) - 1:
            return math.atan2(self.path[j][1] - self.path[j - 1][1], self.path[j][0] - self.path[j - 1][0])
        else:
            return math.atan2(self.path[j + 1][1] - self.path[j][1], self.path[j + 1][0] - self.path[j][0])


class Pursuit(object):
    '''
    Input: the current state of the player vehicle, the path to follow
    Output: control of the player vehicle
    '''

    def __init__(self):
        super(Pursuit, self).__init__()

        self.car_steer = 0.0
        self.path = Path()
        self.car_info = None
        self.tm = rospy.Timer(rospy.Duration(0.1), self.cb_pose_timer)  ##0.2 for golfcart; 0.05
        rospy.Subscriber("ego_state", CarInfo, self.cb_car_info, queue_size=1)
        self.cmd_steer_pub = rospy.Publisher("/purepursuit_cmd_steer", Float32, queue_size=1)
        self.length = 2.8
        self.rear_length = 1.4
        self.max_steer_angle = 30

    def cb_car_info(self, car_info):
        self.car_info = car_info
        if car_info.initial:
            self.length = (carla.Vector2D(car_info.front_axle_center.x, car_info.front_axle_center.y) -
                           carla.Vector2D(car_info.rear_axle_center.x, car_info.rear_axle_center.y)).length()
            self.rear_length = (carla.Vector2D(car_info.car_pos.x, car_info.car_pos.y) -
                           carla.Vector2D(car_info.rear_axle_center.x, car_info.rear_axle_center.y)).length()
            self.max_steer_angle = self.car_info.max_steer_angle
            print('car length {}, rear length {}'.format(self.length, self.rear_length))

    def cb_pose_timer(self, event):
        if self.car_info is None:
            return

        position = (self.car_info.car_pos.x - math.cos(self.car_info.car_yaw) * self.rear_length,
                    self.car_info.car_pos.y - math.sin(self.car_info.car_yaw) * self.rear_length)
        pursuit_angle = self.path.pursuit_tan(position)
        pursuit_point = self.path.pursuit(position)

        if pursuit_angle is None:
            return

        car_yaw = self.car_info.car_yaw
        last_steer = self.car_steer
        angular_offset = self.calc_angular_diff(position, pursuit_point, car_yaw)
        offset = dist(position, pursuit_point)

        relative_point = (offset * math.cos(angular_offset), offset * math.sin(angular_offset))
        if relative_point[1] == 0:
            self.car_steer = 0
        else:
            turning_radius = (relative_point[0] ** 2 + relative_point[1] ** 2) / (
                        2 * abs(relative_point[1]))  # Intersecting chords theorem.
            steering_angle = math.atan2(self.length, turning_radius)
            if relative_point[1] < 0:
                steering_angle *= -1
            self.car_steer = steering_angle / np.deg2rad(self.max_steer_angle)
            self.car_steer = max(-1.0, min(1.0, self.car_steer))

        self.publish_steer()

    def calc_angular_diff(self, position, pursuit_point, car_yaw):
        target = math.atan2(pursuit_point[1] - position[1], pursuit_point[0] - position[0])
        return angle_diff(target, car_yaw)

    def publish_steer(self):
        steer_to_pub = Float32()
        steer_to_pub.data = self.car_steer
        self.cmd_steer_pub.publish(steer_to_pub)


if __name__ == '__main__':
    rospy.init_node('purepursuit')
    pursuit = Pursuit()
    rospy.spin()
