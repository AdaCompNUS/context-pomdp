#!/usr/bin/env python2

from summit import Summit
import carla
import rospy
import msg_builder.msg
import nav_msgs.msg
import cv2
import numpy as np
import math, time
import torch
from Data_processing.global_params import error_handler, print_long, config


def get_position(actor):
    pos3D = actor.get_location()
    return carla.Vector2D(pos3D.x, pos3D.y)


def get_forward_direction(actor):
    forward = actor.get_transform().get_forward_vector()
    return carla.Vector2D(forward.x, forward.y)


def get_bounding_box(actor):
    bbox = actor.bounding_box
    loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
    forward_vec = get_forward_direction(
        actor).make_unit_vector()
    sideward_vec = forward_vec.rotate(np.deg2rad(90))

    half_y_len = bbox.extent.y
    half_x_len = bbox.extent.x

    corners = [loc - half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec - half_y_len * sideward_vec,
               loc - half_x_len * forward_vec - half_y_len * sideward_vec]
    return corners


def overlay(img1, img2):
    return np.copyto(img1, img2, where=(img2 != 0))


class SummitDQL(Summit):
    def __init__(self):
        super(SummitDQL, self).__init__()

        self.ego_state = None
        self.ego_path = []
        self.network_agents = []
        self.sidewalk_agents = []
        self.range = 10
        self.resolution = 0.2
        self.ego_position = None
        self.ego_x_dir = None
        self.ego_y_dir = None
        self.prev_exo_network_state_frame = self.create_frame()
        self.prev_control = [0.0, 0.0]

        self.data_patience_clock = 0
        self.data_patience = 50

        self.query_data = self.make_query_data_dict()

        self.ego_state_sub = rospy.Subscriber(
            '/ego_state', msg_builder.msg.car_info,
            self.ego_state_callback, queue_size=1)
        self.ego_path_sub = rospy.Subscriber(
            '/plan', nav_msgs.msg.Path,
            self.ego_path_callback, queue_size=1)
        self.network_agents_sub = rospy.Subscriber(
            '/crowd/network_agents', msg_builder.msg.CrowdNetworkAgentArray,
            self.network_agents_callback, queue_size=1)
        self.sidewalk_agents_sub = rospy.Subscriber(
            '/crowd/sidewalk_agents', msg_builder.msg.CrowdSidewalkAgentArray,
            self.sidewalk_agents_callback, queue_size=1)

    def dispose(self):
        pass

    def make_query_data_dict(self):
        return {
            'true_steer_normalized': 0.0,
            'true_acc': 0.0,
            'true_vel': 0.0,
            'true_lane': 0.0,
            'nn_input': np.zeros(
                (1, 1, config.total_num_channels, config.imsize, config.imsize), dtype=config.data_type),
            'nn_semantic_input': [0.0, 0.0]
        }

    def update_coordinate_frame(self):
        try:
            actor = self.world.get_actor(self.ego_state.id)
            transform = actor.get_transform()
            position = carla.Vector2D(transform.location.x, transform.location.y)
            x_dir = transform.get_forward_vector()
            x_dir = carla.Vector2D(x_dir.x, x_dir.y)
            y_dir = x_dir.rotate(np.deg2rad(90))

            self.ego_position = position
            self.ego_x_dir = x_dir
            self.ego_y_dir = y_dir
        except Exception as e:
            error_handler(e)

    def pixel(self, point):
        v = point - self.ego_position
        x = carla.Vector2D.dot_product(v, self.ego_x_dir)
        y = carla.Vector2D.dot_product(v, self.ego_y_dir)
        return (int((self.range + y) / self.resolution), int((self.range - x) / self.resolution))

    def create_frame(self):
        return np.zeros((int(2 * self.range / self.resolution), int(2 * self.range / self.resolution)), dtype=np.uint8)

    def ego_state_callback(self, ego_state):
        self.ego_state = ego_state
        self.update_coordinate_frame()

    def ego_path_callback(self, ego_path):
        self.ego_path = [carla.Vector2D(p.pose.position.x, p.pose.position.y) for p in ego_path.poses]

    def network_agents_callback(self, agents):
        self.network_agents = agents.agents

    def sidewalk_agents_callback(self, agents):
        self.sidewalk_agents = agents.agents

    def draw_ego_path(self):
        try:
            frame = self.create_frame()
            pts = np.array(
                [[self.pixel(p) for p in self.ego_path[1:]]])  # Exclude first point, which is in ego current state.
            cv2.polylines(frame, pts, False, 255, thickness=1)
            frame = cv2.GaussianBlur(frame, (7, 7), 1)
            return frame
        except Exception as e:
            error_handler(e)

    def draw_ego_state(self):
        frame = self.create_frame()
        ego_actor = self.world.get_actor(self.ego_state.id)
        pts = np.array([[self.pixel(carla.Vector2D(p.x, p.y)) for p in get_bounding_box(ego_actor)]])
        cv2.fillPoly(frame, pts, 255)
        return frame

    def draw_exo_network_state(self):
        try:
            self.data_patience_clock = 0
            frame = self.create_frame()
            for agent in self.network_agents:
                actor = self.world.get_actor(agent.id)
                if actor is not None:
                    pts = np.array([[self.pixel(p) for p in get_bounding_box(actor)]])
                    cv2.fillPoly(frame, pts, 255)
            frame = cv2.GaussianBlur(frame, (7, 7), 1)
            return frame
        except Exception as e:
            error_handler(e)

    def draw_state_frame(self):
        frame = self.draw_ego_path()
        overlay(frame, self.draw_exo_network_state())
        overlay(frame, self.draw_ego_state())
        return frame

    def get_signed_angle_diff(self, vector1, vector2):
        theta = math.atan2(vector1.y, vector1.x) - math.atan2(vector2.y, vector2.x)
        theta = np.rad2deg(theta)
        if theta > 180:
            theta -= 360
        elif theta < -180:
            theta += 360
        return np.deg2rad(theta)

    def get_forward_direction(self, actor):
        forward = actor.get_transform().get_forward_vector()
        return carla.Vector2D(forward.x, forward.y)

    def get_variables(self):
        ego_actor = self.world.get_actor(self.ego_state.id)
        ego_position = carla.Vector2D(ego_actor.get_location().x, ego_actor.get_location().y)
        ego_bbox = get_bounding_box(ego_actor)

        speed = ego_actor.get_velocity()
        speed = (speed.x ** 2 + speed.y ** 2) ** 0.5
        path_length_offset = (self.ego_path[1] - ego_position).length()
        path_angle_offset = self.get_signed_angle_diff((self.ego_path[5] - ego_position),
                                                       self.get_forward_direction(ego_actor))

        collision = 0
        for agent in self.network_agents:
            actor = self.world.get_actor(agent.id)
            if actor is not None:
                bbox = get_bounding_box(actor)
                if not carla.OccupancyMap(ego_bbox).intersection(carla.OccupancyMap(bbox)).is_empty:
                    collision = 1
                    break

        return {'speed': speed, 'path_length_offset': path_length_offset, 'path_angle_offset': path_angle_offset,
                'collision': collision}

    def get_nn_input(self):
        try:
            self.update_coordinate_frame()
            exo_network_state_frame = self.draw_exo_network_state()
            frames = np.array([self.draw_ego_path(),
                               exo_network_state_frame,
                               self.prev_exo_network_state_frame], dtype=config.data_type)
            self.prev_exo_network_state_frame = exo_network_state_frame

            return frames.astype(np.float32), np.array(self.prev_control, dtype=np.float32)
        except Exception as e:
            error_handler(e)

    def get_labels(self):
        return 0.0, 0.0, 0.0, 0.0

    def record_control(self, control):
        self.prev_control = control

    def check_alive(self):
        if self.data_is_alive:
            self.data_patience_clock += 1
            print_long("============================ data_lock %d, is_alive %d ======================" %
                       (self.data_patience_clock, self.data_is_alive))

            # data supply is missing
            if self.data_patience_clock >= self.data_patience:
                self.data_is_alive = False
                return False

            return True
        else:
            # still waiting for first data
            print_long('Waiting for triggering')
            return True

    def data_valid(self):
        return True

    def test_terminal(self):
        return False
