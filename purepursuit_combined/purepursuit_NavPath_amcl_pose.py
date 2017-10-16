#!/usr/bin/env python
########
#Copyright (c) Movel AI Pte Ltd. All rights reserved.
########

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

PURSUIT_DIST = 2.00
# PATH_FN = sys.argv[1]

# PATH_FN = 'straight'
# PATH_FN = 'vslam_path_dd.csv'
# PATH_FN = 'path_fusion_p3dx.csv'

SPEED = 0.3
RATIO_ANGULAR = 0.3
MAX_ANGULAR = 0.20

MAP_FRAME = 'map'


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def norm_angle(a):
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a

def angle_diff(m1, m0):
    "m1 - m0"
    return norm_angle(m1 - m0)

class Path(object):
    def __init__(self):
        #self.path = load_path(path_fn, reverse)
        self.path = []
        rospy.Subscriber("plan", NavPath, self.cb_path, queue_size=1)

    def cb_path(self, msg):
        path = []
        for i in range(0, len(msg.poses)):
            x = msg.poses[i].pose.position.x
            y = msg.poses[i].pose.position.y
            path.append((x,y))
        self.path = path
        #print path


    def nearest(self, p):
        return min(enumerate(self.path), key=lambda (_, a): dist(a, p))

    def ahead(self, i, d):
        pi = self.path[i]
        while i < len(self.path) and dist(pi, self.path[i]) < d:
            i += 1
        return i

    def pursuit(self, p, d=PURSUIT_DIST):
        ni, np = self.nearest(p)
        j = self.ahead(ni, d)
        return self.path[j] if j<len(self.path) else None

class Pursuit(object):
    #POSE_TOPIC_TYPE = PoseStamped
    #POSE_TOPIC_TYPE = Odometry
    POSE_TOPIC_TYPE = PoseWithCovarianceStamped
    def __init__(self, reverse=False):
        self.reverse = False
        self.curr_vel = Twist()
        self.path = Path()
        rospy.Subscriber("amcl_pose", self.POSE_TOPIC_TYPE, self.cb_pose, queue_size=1)
        rospy.Subscriber("speed_limit", Float32, self.cb_speed_limit, queue_size=1)
        self.pub_vel = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pub_line = rospy.Publisher("pursuit_line", Marker, queue_size=1)
        self.pub_path = rospy.Publisher('pursuit_path', NavPath, queue_size=1, latch=True)
        self.publish_path()
        self.tm = rospy.Timer(rospy.Duration(0.05), self.cb_timer)

        self.speed_limit = SPEED * 5

    def cb_timer(self, event):
        self.pub_vel.publish(self.curr_vel)
        return

    def cb_speed_limit(self, msg):
        self.speed_limit = msg.data

    def cb_pose(self, msg):
        if (self.POSE_TOPIC_TYPE is PoseStamped):
            pose = msg.pose
        else:
            pose = msg.pose.pose
        pos = pose.position
        p = (pos.x, pos.y)
        a = self.path.pursuit(p)
        #print a

        if a is None:
            print 'Goal reached!'
            # self.pub_vel.publish(Twist())
            self.curr_vel = Twist()
            #self.reverse_path()
            return

        q = pose.orientation
        q = [q.x, q.y, q.z, q.w]
        # print euler_from_quaternion(q)
        _, _, theta = euler_from_quaternion(q)

        angular = self.calc_angular(p, a, theta)

        # if abs(angular) >= MAX_ANGULAR * 0.9:
            # # speed = SPEED/5
            # speed = 0
        # else:
            # speed = SPEED

        # a smooth speed profile
        speed = (1 - min(1, abs(angular)*2/MAX_ANGULAR)) * SPEED
        speed = min(self.speed_limit, speed)
        vel = Twist()
        vel.angular.z = angular
        vel.linear.x = speed
        # print angular, speed
        # self.pub_vel.publish(vel)
        print vel
        self.curr_vel = vel
        self.publish_path()
        self.publish_pursuit_line(p, a, msg.header.frame_id)


    def calc_angular(self, p, a, theta):
        # print (a[1] - p[1], a[0] - p[0])
        target = math.atan2(a[1] - p[1], a[0] - p[0])
        # print target, theta, angle_diff(target, theta)
        r = angle_diff(target, theta) * RATIO_ANGULAR
        if r > MAX_ANGULAR:
            r = MAX_ANGULAR
        if r < -MAX_ANGULAR:
            r = -MAX_ANGULAR
        # if angle_diff(target, theta) > 0:
            # r = ANGULAR
        # else:
            # r = -ANGULAR
        return r

    def publish_path(self):
        path = self.path.path
        msg = NavPath()
        msg.header.frame_id = MAP_FRAME
        msg.header.stamp = rospy.get_rostime()
        for p in path:
            pose = PoseStamped()
            pose.header.frame_id = MAP_FRAME
            pose.header.stamp = rospy.get_rostime()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.orientation.w = 1
            msg.poses.append(pose)
        self.pub_path.publish(msg)

    def publish_pursuit_line(self, p, a, frame_id):
        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = rospy.get_rostime()
        line.ns = "pursuit_line"
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.color.r = 0.8
        line.color.g = 0.2
        line.color.a = 1.0
        line.scale.x = 0.02
        line.points.append(Point(p[0], p[1], 0))
        line.points.append(Point(a[0], a[1], 0))
        self.pub_line.publish(line)


def main():
    rospy.init_node('purepursuit')
    reverse = rospy.get_param('~reverse', False)
    pursuit = Pursuit(reverse)
    rospy.spin()

if __name__=='__main__':
    main()
