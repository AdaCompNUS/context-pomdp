#! /usr/bin/env python2

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32

class JoystickTeleop(object):
    def __init__(self):
        self.joy_sub = rospy.Subscriber("joy", Joy, self.joy_callback, queue_size=1)
        self.cmd_accel_pub = rospy.Publisher('/cmd_accel', Float32, queue_size=1)
        self.cmd_steer_pub = rospy.Publisher('/cmd_steer', Float32, queue_size=1)

    def joy_callback(self, data):
        self.cmd_accel_pub.publish(Float32(data.axes[1]))
        self.cmd_steer_pub.publish(Float32(-data.axes[2]))

if __name__ == '__main__':
    rospy.init_node('joystick_teleop')
    joystick_teleop = JoystickTeleop()
    rospy.spin()
