#!/usr/bin/env python
''' ROS node implementing a virtual bumper using laser scan
    params: slowing_distance, stopping_distance, scan angle, max_speed
'''

import sys
import math
import numpy as np

# Ros libraries
import rospy

# Ros Messages
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class VirtualBumper:
    ''' wrapper for laser scan callback'''
    def __init__(self, slow_dist, stop_dist, angle, max_speed):
        '''Initialize ros publisher, ros subscriber'''

        self.alpha = 0.1
        self.rolling_range = 0

        # topic where we publish
        self.speed_limit = rospy.Publisher("/speed_limit",
                                           Float32, queue_size=1)

        # angle on each side to consider for virtual bumper
        self.angle = angle
        self.slow_dist = slow_dist
        self.stop_dist = stop_dist
        self.max_speed = max_speed
        # subscribed Topic
        self.subscriber = rospy.Subscriber("/laser_scan",
                                           LaserScan, self.callback, queue_size=1,
                                           buff_size=2**24)


    def callback(self, ros_data):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''
        # take scan values from plus / minus of angle
        ang_inc = ros_data.angle_increment
        num_scans = int((2*math.pi/360  * self.angle) / ang_inc)
        min_range = np.nanmin(ros_data.ranges[:num_scans] + ros_data.ranges[-num_scans:])

        if np.isnan(min_range):
            min_range = 0

        self.rolling_range = (1.0 - self.alpha ) * self.rolling_range + self.alpha * min_range

        if self.rolling_range > self.slow_dist:
            limit = self.max_speed
        elif self.rolling_range > self.stop_dist:
            limit = (self.rolling_range - self.stop_dist) / (self.slow_dist - self.stop_dist) \
                * self.max_speed
        else:
            limit = 0

        self.speed_limit.publish(limit)


def main(args):
    '''Initializes and cleanup ros node'''
    bumper = VirtualBumper(float(args[1]), float(args[2]), float(args[3]), float(args[4]))
    rospy.init_node('virtual_bumper', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS virtual bumper module")

if __name__ == '__main__':
    main(sys.argv)
