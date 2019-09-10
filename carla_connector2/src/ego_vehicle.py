#!/usr/bin/env python2

from drunc import Drunc
import carla

import random
import math
import numpy as np

import rospy
import tf
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, Vector3, Polygon, Point32, PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Float32, Bool

from network_agent_path import NetworkAgentPath
from peds_unity_system.msg import car_info as CarInfo # panpan
from util import *

import tf2_geometry_msgs
import tf2_ros
import geometry_msgs
import time
from tf import TransformListener
import tf.transformations as tftrans

class EgoVehicle(Drunc):
    def __init__(self):
        super(EgoVehicle, self).__init__()

        # time.sleep(2)
        # Create path.
        self.actor = None
        self.speed = 0.0
        while self.actor is None:

            scale=1.0
            shift_x= 0.0
            shift_y= 0.0

            # the complex cross 
            # scale=0.01
            # shift_x=-0.145 #(-0.145), 
            # shift_y= -0.19

            # y-shape
            # scale=0.01
            # shift_x=0.030
            # shift_y= -0.108
            
            spawn_min, spawn_max = self.get_shrinked_range(scale=scale, shift_x=shift_x, shift_y= shift_y, draw_range=False)

            # a narrow road for peds to cross
            # spawn_min = carla.Vector2D(903.4-20, 1531.7-20)
            # spawn_max = carla.Vector2D(903.4+20, 1531.7+20)
            
            # y-shaped region, msekel, for cars
            # spawn_min = carla.Vector2D(900.4, 350.33) 
            # spawn_max = carla.Vector2D(1000.4, 450.33)

            # single road without building, msekel
            # spawn_min = carla.Vector2D(1036.5-20, 507.21-20)
            # spawn_max = carla.Vector2D(1036.5+20, 507.21+20)
            
            self.path = NetworkAgentPath.rand_path(self, 20, 1.0)
                
            vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.etron'))
            vehicle_bp.set_attribute('role_name', 'ego_vehicle')
            spawn_position = self.path.get_position()
            spawn_trans = carla.Transform()
            spawn_trans.location.x = spawn_position.x
            spawn_trans.location.y = spawn_position.y
            spawn_trans.location.z = 0.5
            spawn_trans.rotation.yaw = self.path.get_yaw()

            self.actor = self.world.try_spawn_actor(vehicle_bp, spawn_trans)

            self.actor.set_collision_enabled(False)

        time.sleep(1) # wait for the vehicle to drop

        self.cmd_speed = 0
        self.cmd_accel = 0
        self.cmd_steer = 0

        self.cmd_speed_sub = rospy.Subscriber('/cmd_speed', Float32, self.cmd_speed_callback, queue_size=1)
        self.cmd_accel_sub = rospy.Subscriber('/cmd_accel', Float32, self.cmd_accel_callback, queue_size=1)
        self.cmd_steer_sub = rospy.Subscriber('/cmd_steer', Float32, self.cmd_steer_callback, queue_size=1)

        self.odom_broadcaster = tf.TransformBroadcaster()
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
        self.car_info_pub = rospy.Publisher('/IL_car_info', CarInfo, queue_size=1)
        self.plan_pub = rospy.Publisher('/plan', NavPath, queue_size=1)

        self.broadcaster = None
        self.publish_odom_transform()
        self.transformer = TransformListener()

    def get_shrinked_range(self, scale = 1.0, shift_x = 0.0, shift_y = 0.0, draw_range = False):
        if scale == 1.0:
            return self.map_bounds_min, self.map_bounds_max # TODO: I want to get the actual map range here
        else: 
            map_range = self.map_bounds_max - self.map_bounds_min
            new_map_range = carla.Vector2D(map_range.x * scale, map_range.y * scale)
            map_margin_range = map_range - new_map_range
            map_margin_range.x = map_margin_range.x / 2.0
            map_margin_range.y = map_margin_range.y / 2.0

            shift = carla.Vector2D(shift_x * map_range.x, shift_y * map_range.y) 

            spawn_min = carla.Vector2D(
                self.map_bounds_min.x + map_margin_range.x, 
                self.map_bounds_min.y + map_margin_range.y) + shift
            spawn_max = carla.Vector2D(
                self.map_bounds_max.x - map_margin_range.x,
                self.map_bounds_max.y - map_margin_range.y) + shift

            if draw_range:
                self.world.debug.draw_arrow(
                    carla.Location(spawn_min.x, spawn_min.y, 0), 
                    carla.Location(spawn_max.x, spawn_min.y, 0), 
                    thickness = 3.0, arrow_size = 3.0, color = carla.Color(255,0,0))
                self.world.debug.draw_arrow(
                    carla.Location(spawn_min.x, spawn_min.y, 0), 
                    carla.Location(spawn_min.x, spawn_max.y, 0), 
                    thickness = 3.0, arrow_size = 3.0, color = carla.Color(0,255,0))

            return spawn_min, spawn_max

    def dispose(self):
        self.actor.destroy()

    def get_position(self):
        location = self.actor.get_location()
        return carla.Vector2D(location.x, location.y)

    def get_cur_ros_pose(self):
        cur_pose = geometry_msgs.msg.PoseStamped()
        # cur_pose = geometry_msgs.msg.TransformStamped()
   
        cur_pose.header.stamp = rospy.Time.now()
        cur_pose.header.frame_id = "/map"
  
        cur_pose.pose.position.x = self.actor.get_location().x
        cur_pose.pose.position.y = self.actor.get_location().y
        cur_pose.pose.position.z = self.actor.get_location().z
  
        quat = tf.transformations.quaternion_from_euler(
                float(0),float(0),float(np.deg2rad(self.actor.get_transform().rotation.yaw)))
        cur_pose.pose.orientation.x = quat[0]
        cur_pose.pose.orientation.y = quat[1]
        cur_pose.pose.orientation.z = quat[2]
        cur_pose.pose.orientation.w = quat[3]

        return cur_pose
    
    def get_cur_ros_transform(self):
        transformStamped = geometry_msgs.msg.TransformStamped()
   
        transformStamped.header.stamp = rospy.Time.now()
        transformStamped.header.frame_id = "map"
        transformStamped.child_frame_id = 'odom'
  
        transformStamped.transform.translation.x = self.actor.get_location().x
        transformStamped.transform.translation.y = self.actor.get_location().y
        transformStamped.transform.translation.z = self.actor.get_location().z
  
        quat = tf.transformations.quaternion_from_euler(
                float(0),float(0),float(
                    np.deg2rad(self.actor.get_transform().rotation.yaw)))
        transformStamped.transform.rotation.x = quat[0]
        transformStamped.transform.rotation.y = quat[1]
        transformStamped.transform.rotation.z = quat[2]
        transformStamped.transform.rotation.w = quat[3]

        return transformStamped    

    def publish_odom_transform(self):
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = self.get_cur_ros_transform()
        self.broadcaster.sendTransform(static_transformStamped)

    def get_transform_wrt_odom_frame(self):
        try:
            (trans, rot) = self.transformer.lookupTransform("map", "odom", rospy.Time(0.2))
        except:
            return None

        cur_pose = self.get_cur_ros_pose()

        transform = tftrans.concatenate_matrices(
            tftrans.translation_matrix(trans), tftrans.quaternion_matrix(rot))
        inversed_transform = tftrans.inverse_matrix(transform)

        inv_translation = tftrans.translation_from_matrix(inversed_transform)
        inv_quaternion = tftrans.quaternion_from_matrix(inversed_transform)

        transformStamped = geometry_msgs.msg.TransformStamped()
        transformStamped.transform.translation.x = inv_translation[0]
        transformStamped.transform.translation.y = inv_translation[1]
        transformStamped.transform.translation.z = inv_translation[2]
        transformStamped.transform.rotation.x = inv_quaternion[0]
        transformStamped.transform.rotation.y = inv_quaternion[1]
        transformStamped.transform.rotation.z = inv_quaternion[2]
        transformStamped.transform.rotation.w = inv_quaternion[3]

        cur_transform_wrt_odom = tf2_geometry_msgs.do_transform_pose(
            cur_pose, transformStamped)
        
        translation = cur_transform_wrt_odom.pose.position

        quaternion = (
                cur_transform_wrt_odom.pose.orientation.x,
                cur_transform_wrt_odom.pose.orientation.y,
                cur_transform_wrt_odom.pose.orientation.z,
                cur_transform_wrt_odom.pose.orientation.w)

        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)   

        return translation, yaw

    def publish_odom(self):
        # Check if result available.
        result = self.get_transform_wrt_odom_frame()
        if result is None:
            return

        current_time = rospy.Time.now() 
        
        frame_id = "odom"
        child_frame_id = "base_link"

        (translation, yaw) = result
        pos = carla.Location(translation.x, translation.y, translation.z)
        vel = self.actor.get_velocity()
        v_2d = np.array([vel.x, vel.y, 0])
        forward = np.array([math.cos(yaw), math.sin(yaw), 0])
        speed = np.vdot(forward, v_2d)
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        w_yaw = self.actor.get_angular_velocity().z

        pi_2 = 2*3.1415926
        print_yaw = yaw
        if yaw < 0:
            print_yaw = yaw + pi_2
        if yaw >= pi_2:
            print_yaw = yaw - pi_2
        # print("yaw of base_link = {}".format(print_yaw))

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
        # get pos and yaw w.r.t. the map frame
        pos = self.actor.get_location()
        yaw = np.deg2rad(self.actor.get_transform().rotation.yaw)
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        odom.pose.pose = Pose(Point(pos.x, pos.y, 0), Quaternion(*odom_quat))
        odom.child_frame_id = child_frame_id
        odom.twist.twist = Twist(Vector3(vel.x, vel.y, vel.z), Vector3(0, 0, w_yaw))
        self.odom_pub.publish(odom)
    
    def publish_il_car_info(self):
        car_info_msg = CarInfo()

        pos = self.actor.get_location()
        pos2D = carla.Vector2D(pos.x, pos.y)
        vel = self.actor.get_velocity()
        yaw = np.deg2rad(self.actor.get_transform().rotation.yaw)
        v_2d = np.array([vel.x, vel.y, 0])
        forward = np.array([math.cos(yaw), math.sin(yaw), 0])
        speed = np.vdot(forward, v_2d)
        self.speed = speed
        
        car_info_msg.car_pos.x = pos.x
        car_info_msg.car_pos.y = pos.y
        car_info_msg.car_pos.z = pos.z
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

        wheels = self.actor.get_physics_control().wheels
        # TODO I think that CARLA might have forgotten to divide by 100 here.
        wheel_positions = [w.position / 100 for w in wheels]

        front_axle_center = (wheel_positions[0] + wheel_positions[1]) / 2
        rear_axle_center = (wheel_positions[2] + wheel_positions[3]) / 2
        
        car_info_msg.front_axle_center.x = front_axle_center.x
        car_info_msg.front_axle_center.y = front_axle_center.y
        car_info_msg.front_axle_center.z = front_axle_center.z
        car_info_msg.rear_axle_center.x = rear_axle_center.x
        car_info_msg.rear_axle_center.y = rear_axle_center.y
        car_info_msg.rear_axle_center.z = rear_axle_center.z
        car_info_msg.max_steer_angle = wheels[0].max_steer_angle

        self.car_info_pub.publish(car_info_msg)

    def publish_plan(self):
        current_time = rospy.Time.now()

        gui_path = NavPath()
        gui_path.header.frame_id = 'map'
        gui_path.header.stamp = current_time

        
        values = [(carla.Vector2D(self.actor.get_location().x, self.actor.get_location().y), self.actor.get_transform().rotation.yaw)]
        # Exclude last point because no yaw information.
        values += [(self.path.get_position(i), self.path.get_yaw(i)) for i in range(len(self.path.route_points) - 1)]
        for (position, yaw) in values:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = current_time
            pose.pose.position.x = position.x
            pose.pose.position.y = position.y
            pose.pose.position.z = 0
            quaternion = tf.transformations.quaternion_from_euler(0, 0, np.deg2rad(yaw))
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            gui_path.poses.append(pose)
        
        self.plan_pub.publish(gui_path)

    def cmd_speed_callback(self, speed):
        self.cmd_speed = speed.data

    def cmd_accel_callback(self, accel):
        self.cmd_accel = accel.data

    def cmd_steer_callback(self, steer):
        self.cmd_steer = steer.data

    def draw_path(self, path):
        color_i = 255
        last_loc = None
        for i in range(len(path.route_points)):
            pos = path.get_position(i)
            loc = carla.Location(pos.x, pos.y, 0.1)
            if last_loc is not None:
                self.world.debug.draw_line(last_loc,loc,life_time = 0.1, 
                    color = carla.Color(color_i,color_i,0,0))
            last_loc = carla.Location(pos.x, pos.y, 0.1)

    def update(self):
        # Calculate control and send to CARLA.
        # print("controlling vehicle with acc={} cur_vel={}".format(self.cmd_accel, self.speed))
        control = self.actor.get_control()
        control.gear = 1
        control.steer = self.cmd_steer
        if self.cmd_accel > 0:
            control.throttle = self.cmd_accel
            control.brake = 0.0
            control.reverse = False
        elif self.cmd_accel == 0:
            control.throttle = 0.0
            control.brake = 0.0
            control.reverse = False
        else:
            if self.speed <= 1.5: # no reverse
                control.throttle = 0                
                control.brake = 1.0
                control.reverse = False
            else:
                control.throttle = -self.cmd_accel
                control.brake = 0.0
                control.reverse = True

        # debugging freezing the robot
        # control.throttle = 0                
        # control.brake = 1.0
        # control.reverse = False 

        self.actor.apply_control(control)

        # Publish info.
        if not self.path.resize():
            print('Warning : path too short.')
            return
        self.path.cut(self.get_position())
        if not self.path.resize():
            print('Warning : path too short.')
            return
        self.draw_path(self.path)
        self.publish_odom()
        self.publish_il_car_info()
        self.publish_plan()

if __name__ == '__main__':
    rospy.init_node('ego_vehicle')
    rospy.wait_for_message("/meshes_spawned", Bool)
    ego_vehicle = EgoVehicle()

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        ego_vehicle.update()
        rate.sleep()

    ego_vehicle.dispose()
