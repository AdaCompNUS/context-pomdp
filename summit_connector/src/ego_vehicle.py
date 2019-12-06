#!/usr/bin/env python2

from summit import Summit
import carla

import random
import math
import numpy as np
import sys

import rospy
import tf
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, Vector3, Polygon, Point32, PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Float32, Bool, Int32

from network_agent_path import NetworkAgentPath
from msg_builder.msg import car_info as CarInfo  # panpan
from util import *

import tf2_geometry_msgs
import tf2_ros
import geometry_msgs
import time
from tf import TransformListener
import tf.transformations as tftrans

change_left = -1
remain = 0
change_right = 1


class EgoVehicle(Summit):
    def __init__(self):
        super(EgoVehicle, self).__init__()

        # Initialize fields.
        self.gamma_cmd_accel = 0
        self.gamma_cmd_steer = 0
        self.imitation_cmd_accel = 0
        self.imitation_cmd_steer = 0
        self.pp_cmd_steer = 0
        self.pomdp_cmd_accel = 0
        self.pomdp_cmd_steer = 0
        self.lane_decision = 0

        self.start_time = None
        self.last_decision = remain

        # ROS stuff.
        self.control_mode = rospy.get_param('~control_mode', 'gamma')
        print('Ego_vehicle control mode: {}'.format(self.control_mode))
        sys.stdout.flush()

        self.pomdp_cmd_accel_sub = rospy.Subscriber('/pomdp_cmd_accel', Float32,
                                                    self.pomdp_cmd_accel_callback, queue_size=1)
        self.pomdp_cmd_steer_sub = rospy.Subscriber('/pomdp_cmd_steer', Float32,
                                                    self.pomdp_cmd_steer_callback, queue_size=1)
        self.gamma_cmd_accel_sub = rospy.Subscriber('/gamma_cmd_accel', Float32,
                                                    self.gamma_cmd_accel_callback, queue_size=1)
        self.gamma_cmd_steer_sub = rospy.Subscriber('/gamma_cmd_steer', Float32,
                                                    self.gamma_cmd_steer_callback, queue_size=1)
        self.imitation_cmd_accel_sub = rospy.Subscriber('/imitation_cmd_accel',
                                                        Float32, self.imitation_cmd_accel_callback, queue_size=1)
        self.imitation_cmd_steer_sub = rospy.Subscriber('/imitation_cmd_steer',
                                                        Float32, self.imitation_cmd_steer_callback, queue_size=1)

        self.pp_cmd_accel_sub = rospy.Subscriber('/purepursuit_cmd_steer',
                                                 Float32, self.pp_cmd_steer_callback, queue_size=1)

        if self.control_mode == 'gamma' or self.control_mode == 'joint_pomdp' or self.control_mode == 'other':
            self.gamma_lane_decision_sub = rospy.Subscriber('/gamma_lane_decision', Int32,
                                                        self.gamma_lane_change_decision_callback, queue_size=1)
        if self.control_mode == 'imitation':
            self.imitation_lane_decision_sub = rospy.Subscriber('/imitation_lane_decision', Int32,
                                                        self.gamma_lane_change_decision_callback, queue_size=1)

        self.odom_broadcaster = tf.TransformBroadcaster()
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
        self.car_info_pub = rospy.Publisher('/ego_state', CarInfo, queue_size=1)
        self.plan_pub = rospy.Publisher('/plan', NavPath, queue_size=1)
        self.ego_dead_pub = rospy.Publisher('/ego_dead', Bool, queue_size=1)

        # Create path.
        self.actor = None
        self.speed = 0.0
        while self.actor is None:
            scenario_segment_map = self.network_segment_map.intersection(
                carla.OccupancyMap(self.scenario_min, self.scenario_max))
            # scenario_segment_map = self.network_segment_map.intersection(
            #     carla.OccupancyMap(carla.Vector2D(400, 454), carla.Vector2D(402, 456)))
            scenario_segment_map.seed_rand(self.rng.getrandbits(32))
            self.path = NetworkAgentPath.rand_path(self, 20, 1.0, scenario_segment_map, min_safe_points=100,
                                                   rng=self.rng)

            vehicle_bp = self.rng.choice(self.world.get_blueprint_library().filter('vehicle.mini.cooperst'))
            vehicle_bp.set_attribute('role_name', 'ego_vehicle')
            spawn_position = self.path.get_position()
            spawn_trans = carla.Transform()
            spawn_trans.location.x = spawn_position.x
            spawn_trans.location.y = spawn_position.y
            spawn_trans.location.z = 0.5
            spawn_trans.rotation.yaw = self.path.get_yaw()

            print("Ego-vehicle at {} {}".format(spawn_position.x, spawn_position.y))

            self.actor = self.world.try_spawn_actor(vehicle_bp, spawn_trans)

            # if self.actor:
            #    self.actor.set_collision_enabled(True)

        self.world.wait_for_tick(1.0)  # Wait for collision to be applied.

        time.sleep(1)  # wait for the vehicle to drop
        self.publish_odom()
        self.publish_il_car_info()
        self.publish_plan()
        rospy.wait_for_message("/agents_ready", Bool)  # wait for agents to initialize

        self.broadcaster = None
        self.publish_odom_transform()
        self.transformer = TransformListener()

        self.update_timer = rospy.Timer(rospy.Duration(1.0 / 20), self.update)

    def dispose(self):
        self.update_timer.shutdown()
        self.actor.destroy()

    def get_position(self):
        location = self.actor.get_location()
        return carla.Vector2D(location.x, location.y)

    def get_cur_ros_pose(self):
        cur_pose = geometry_msgs.msg.PoseStamped()

        cur_pose.header.stamp = rospy.Time.now()
        cur_pose.header.frame_id = "/map"

        cur_pose.pose.position.x = self.actor.get_location().x
        cur_pose.pose.position.y = self.actor.get_location().y
        cur_pose.pose.position.z = self.actor.get_location().z

        quat = tf.transformations.quaternion_from_euler(
            float(0), float(0), float(np.deg2rad(self.actor.get_transform().rotation.yaw)))
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
            float(0), float(0), float(
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

        pi_2 = 2 * 3.1415926
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

        # WARNING: this section is hardcoded against the internals of gamma controller.
        # GAMMA assumes these calculations, and is required for controlling ego vehicle
        # using GAMMA. When not controlling using GAMMA, pref_vel is practically not used,
        # since the other crowd agents calculate their velocities without knowledge of the
        # ego vehicle's pref_vel anyway.
        target_position = self.path.get_position(5)
        pref_vel = 6.0 * (target_position - pos2D).make_unit_vector()
        car_info_msg.car_pref_vel.x = pref_vel.x
        car_info_msg.car_pref_vel.y = pref_vel.y

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

        values = [(carla.Vector2D(self.actor.get_location().x, self.actor.get_location().y),
                   self.actor.get_transform().rotation.yaw)]
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

    def pomdp_cmd_accel_callback(self, accel):
        self.pomdp_cmd_accel = accel.data

    def pomdp_cmd_steer_callback(self, steer):
        self.pomdp_cmd_steer = steer.data

    def gamma_cmd_accel_callback(self, accel):
        self.gamma_cmd_accel = accel.data

    def gamma_cmd_steer_callback(self, steer):
        self.gamma_cmd_steer = steer.data

    def imitation_cmd_accel_callback(self, accel):
        self.imitation_cmd_accel = accel.data

    def imitation_cmd_steer_callback(self, steer):
        self.imitation_cmd_steer = steer.data

    def pp_cmd_steer_callback(self, steer):
        self.pp_cmd_steer = steer.data

    def gamma_lane_change_decision_callback(self, decision):

        self.lane_decision = int(decision.data)
        if self.lane_decision == self.last_decision:
            return
        if self.lane_decision * self.last_decision == -1:
            self.last_decision = self.lane_decision
            return
        print('change lane decision {}'.format(self.lane_decision))
        sys.stdout.flush()
        self.last_decision = self.lane_decision
        self.update_path(self.lane_decision)

    def draw_path(self, path):
        color_i = 255
        last_loc = None
        for i in range(len(path.route_points)):
            pos = path.get_position(i)
            loc = carla.Location(pos.x, pos.y, 0.1)
            if last_loc is not None:
                self.world.debug.draw_line(last_loc, loc, life_time=0.1,
                                           color=carla.Color(color_i, color_i, 0, 0))
            last_loc = carla.Location(pos.x, pos.y, 0.1)

    def send_control(self):
        # Calculate control and send to CARLA.
        # print("controlling vehicle with acc={} cur_vel={}".format(self.cmd_accel, self.speed))
        control = self.actor.get_control()

        if self.control_mode == 'gamma':
            cmd_accel = self.gamma_cmd_accel
            cmd_steer = self.gamma_cmd_steer
        elif self.control_mode == 'imitation':
            cmd_accel = self.imitation_cmd_accel
            cmd_steer = self.pp_cmd_steer
            # print('Publishing imitation cmd ({}, {}, {})'.format(cmd_accel, self.lane_decision, cmd_steer))
            # sys.stdout.flush()
        elif self.control_mode == 'joint_pomdp':
            cmd_accel = self.pomdp_cmd_accel
            cmd_steer = self.pomdp_cmd_steer
        elif self.control_mode == 'other':
            cmd_accel = self.pomdp_cmd_accel
            cmd_steer = self.pp_cmd_steer

        control.steer = cmd_steer
        if cmd_accel > 0:
            control.throttle = cmd_accel
            control.brake = 0.0
        elif cmd_accel == 0:
            control.throttle = 0.0
            control.brake = 0.0
        else:
            control.throttle = 0
            control.brake = 1.0
        control.manual_gear_shift = True
        control.gear = 1

        # debugging freezing the robot
        # control.throttle = 0                
        # control.brake = 1.0
        # control.reverse = False 

        self.actor.apply_control(control)

    def update_path(self, lane_decision):
        if lane_decision == remain:
            return

        pos = self.actor.get_location()
        ego_veh_pos = carla.Vector2D(pos.x, pos.y)
        yaw = np.deg2rad(self.actor.get_transform().rotation.yaw)

        forward_vec = carla.Vector2D(math.cos(yaw), math.sin(yaw))
        sidewalk_vec = forward_vec.rotate(np.deg2rad(90))  # rotate clockwise by 90 degree

        ego_veh_pos_in_new_lane = None
        if lane_decision == change_left:
            ego_veh_pos_in_new_lane = ego_veh_pos - 4.0 * sidewalk_vec
        else:
            ego_veh_pos_in_new_lane = ego_veh_pos + 4.0 * sidewalk_vec

        cur_route_point = self.network.get_nearest_route_point(
            self.path.get_position(0))  # self.network.get_nearest_route_point(ego_veh_pos)
        new_route_point = self.network.get_nearest_route_point(ego_veh_pos_in_new_lane)

        lane_change_probability = 1.0
        if new_route_point.edge == cur_route_point.edge and new_route_point.lane != cur_route_point.lane:
            if self.rng.uniform(0.0, 1.0) <= lane_change_probability:
                new_path_candidates = self.network.get_next_route_paths(new_route_point, self.path.min_points - 1,
                                                                        self.path.interval)
                new_path = NetworkAgentPath(self, self.path.min_points, self.path.interval)
                new_path.route_points = self.rng.choice(new_path_candidates)[0:self.path.min_points]
                self.path = new_path

    def update(self, event):

        # Publish info.
        if not self.path.resize():
            self.ego_dead_pub.publish(True)
            return
        else:
            self.path.cut(self.get_position())
            if not self.path.resize():
                self.ego_dead_pub.publish(True)
                return

        self.send_control()
        # self.draw_path(self.path)
        self.publish_odom()
        self.publish_il_car_info()
        self.publish_plan()


if __name__ == '__main__':
    rospy.init_node('ego_vehicle')
    init_time = rospy.Time.now()

    rospy.wait_for_message("/meshes_spawned", Bool)
    ego_vehicle = EgoVehicle()
    rospy.on_shutdown(ego_vehicle.dispose)
    rospy.spin()
