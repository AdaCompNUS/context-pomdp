#!/usr/bin/env python
import math
import rospy
from geometry_msgs.msg import Twist, Pose, Pose2D, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler


#TODO package them into a module like pose_util
def to_pose(x, y, theta):
    p = Pose()
    p.position.x = x
    p.position.y = y
    t = quaternion_from_euler(0,0, theta)
    p.orientation.x = t[0]
    p.orientation.y = t[1]
    p.orientation.z = t[2]
    p.orientation.w = t[3]
    return p

def to_pose2d(x, y, theta):
    p = Pose2D()
    p.x = x
    p.y = y
    p.theta = theta
    return p

class PathRecorder(object):
    def __init__(self, topic):
        self.msg = Path()
        self.msg.header.frame_id = 'odom'
        self.pub = rospy.Publisher(topic, Path, queue_size=2)

    def append(self, pose):
        self.msg.poses.append(pose)
        self.msg.header.stamp = rospy.get_rostime()
        self.pub.publish(self.msg)

pub_pose = None
pose_recorder = None
vslam_recorder = None

class VslamConverter(object):
    def __init__(self, cam_offset):
        self.vslam_recorder = PathRecorder('vslam_path')
        rospy.Subscriber("/ORB_SLAM/pose", PoseStamped, self.cb_slam_pose, queue_size=1)
        self.pub_slam2d_pose = rospy.Publisher("vslam2d_pose", PoseStamped, queue_size=2)
        self.pub_slam2d_posecov = rospy.Publisher('vslam2d_posecov', PoseWithCovarianceStamped, queue_size=2)
        self.cam_offset = cam_offset

    def cb_slam_pose(self, msg):
        cam_offset = self.cam_offset
        pos = msg.pose.position
        y = pos.y
        x = math.sqrt(pos.x**2 + pos.z**2)
        if pos.x < 0:
            x = -x
        q = msg.pose.orientation
        # q = [q.x, q.y, q.z, q.w]
        # assuming the camera is placed horizentally
        # q = [0, q.y, 0, q.w]
	q = [0,0,q.z,q.w]
        # print euler_from_quaternion(q)
        a0, _, theta = euler_from_quaternion(q)
        if abs(a0 - math.pi) < 1e-2:
            # euler angle flipped by 180 degree
            theta = math.pi - theta
        elif abs(a0) > 1e-2:
            print "Invalid angle: ", a0
        # print a0, theta

        # transform the heading angle so it is align with the vanila odometry frame
        theta = -theta
        theta += math.pi / 2 # theta points to the robot's right side, rotate pi/2 to get heading

        # rotate coordinates 90 degree so it aligns with vanila odometry
        # x, y = y, -x
        theta -= math.pi / 2

        # translate the camera to robot
        tx = cam_offset[0] * math.cos(theta) - cam_offset[1] * math.sin(theta)
        ty = cam_offset[0] * math.sin(theta) + cam_offset[1] * math.cos(theta)
        x = x - tx
        y = y - ty

        p = PoseStamped()
        p.pose = to_pose(x, y, theta)
        p.header = msg.header
        self.pub_slam2d_pose.publish(p)
        self.vslam_recorder.append(p)

        pc = PoseWithCovarianceStamped()
        pc.header = msg.header
        pc.pose.pose = p.pose
        cov = pc.pose.covariance
        for i in range(6):
            cov[i*6+i] = 0.10
        self.pub_slam2d_posecov.publish(pc)




def main():
    global pub_pose, pose_recorder, pub_slam2d_pose, vslam_recorder
    rospy.init_node('posetrans')

    cam_offset_x = rospy.get_param('~cam_offset_x')
    cam_offset_y = rospy.get_param('~cam_offset_y')
    cam_offset = (cam_offset_x, cam_offset_y)

    vc = VslamConverter(cam_offset)
    rospy.spin()

if __name__=='__main__':
    main()

