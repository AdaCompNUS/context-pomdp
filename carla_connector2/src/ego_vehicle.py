#!/usr/bin/env python2

import random
import math
import numpy as np

import rospy
import tf
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, Vector3, Polygon, Point32
from nav_msgs.msg import Odometry

from drunc import Drunc
import carla
from network_agent_path import NetworkAgentPath
from peds_unity_system.msg import car_info as CarInfo # panpan
from util import *

class EgoVehicle(Drunc):
    def __init__(self):
        super(EgoVehicle, self).__init__()

        # Create path.
        self.path = NetworkAgentPath.rand_path(self, 20, 1.0)
            
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.bmw.*'))
        vehicle_bp.set_attribute('role_name', 'ego_vehicle')
        spawn_position = self.path.get_position()
        spawn_trans = carla.Transform()
        spawn_trans.location.x = spawn_position.x
        spawn_trans.location.y = spawn_position.y
        spawn_trans.location.z = 2.0
        spawn_trans.rotation.yaw = self.path.get_yaw()
        self.actor = self.world.spawn_actor(vehicle_bp, spawn_trans)
            
        self.odom_broadcaster = tf.TransformBroadcaster()
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
        self.car_info_pub = rospy.Publisher('/IL_car_info', CarInfo, queue_size=1)
        
        self.world.on_tick(self.world_tick_callback)

    def get_position(self):
        location = self.actor.get_location()
        return carla.Vector2D(location.x, location.y)
    
    def publish_odom(self):
        current_time = rospy.Time.now() 

        frame_id = "odom"
        child_frame_id = "base_link"
        pos = self.actor.get_location()
        vel = self.actor.get_velocity()
        yaw = self.actor.get_transform().rotation.yaw
        v_2d = np.array([vel.x, vel.y, 0])
        forward = np.array([math.cos(np.deg2rad(yaw)), math.sin(np.deg2rad(yaw)), 0])
        speed = np.vdot(forward, v_2d)
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, np.deg2rad(yaw))
        w_yaw = self.actor.get_angular_velocity().z

        self.odom_broadcaster.sendTransform(
            (pos.x, pos.y, pos.z),
            odom_quat,
            current_time,
            child_frame_id,
            frame_id
        )

        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = frame_id
        odom.pose.pose = Pose(Point(pos.x, pos.y, 0), Quaternion(*odom_quat))
        odom.child_frame_id = child_frame_id
        odom.twist.twist = Twist(Vector3(vel.x, vel.y, vel.z), Vector3(0, 0, w_yaw))
        self.odom_pub.publish(odom)
    
    def publish_il_car_info(self):
        car_info_msg = CarInfo()

        pos = self.actor.get_location()
        vel = self.actor.get_velocity()
        yaw = self.actor.get_transform().rotation.yaw
        v_2d = np.array([vel.x, vel.y, 0])
        forward = np.array([math.cos(np.deg2rad(yaw)), math.sin(np.deg2rad(yaw)), 0])
        speed = np.vdot(forward, v_2d)
        
        car_info_msg.car_pos.x = pos.x
        car_info_msg.car_pos.y = pos.y
        car_info_msg.car_pos.z = 0
        car_info_msg.car_yaw = yaw
        car_info_msg.car_speed = speed
        car_info_msg.car_steer = self.actor.get_control().steer             
        car_info_msg.car_vel.x = vel.x
        car_info_msg.car_vel.y = vel.y
        car_info_msg.car_vel.z = vel.z
            
        car_info_msg.car_bbox = Polygon()
        corners = get_bounding_box_corners(self.actor)
        for corner in corners:
            car_info_msg.car_bbox.points.append(Point32(
                x=corner.x, y=corner.y, z=0.0))

        self.car_info_pub.publish(car_info_msg)

    def world_tick_callback(self, snapshot):
        if not self.path.resize():
            print('Warning : path too short.')
            return

        self.path.cut(self.get_position())
        
        if not self.path.resize():
            print('Warning : path too short.')
            return

        self.publish_odom()
        self.publish_il_car_info()

if __name__ == '__main__':
    rospy.init_node('ego_vehicle')
    ego_vehicle = EgoVehicle()
    rospy.spin()
