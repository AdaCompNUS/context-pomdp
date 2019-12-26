#!/usr/bin/env python2

from summit import Summit
import carla
import rospy
import msg_builder.msg
from msg_builder.msg import ActionDistrib
import nav_msgs.msg
import cv2
import numpy as np
from std_msgs.msg import Float32, Int32
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from msg_builder.msg import InputImages
import sys, traceback, os
from inspect import getframeinfo


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)
    sys.stdout.flush()
    exit(-1)


def print_long(msg):
    frameinfo = getframeinfo(sys._getframe(1))
    print('[{}:{}:{}] {}.'.format(os.path.basename(frameinfo.filename), frameinfo.function, frameinfo.lineno, msg))
    sys.stdout.flush()


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
        self.range = 30
        self.resolution = 0.1
        self.bounds_min = None
        self.bounds_max = None
        self.bounds = None

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

    def pixel(self, point):
        return (
            int((point.y - self.bounds_min.y) / self.resolution), int((self.bounds_max.x - point.x) / self.resolution))

    def pixel_zero_center(self, point):
        return (
            int((point.y + self.range) / self.resolution), int((self.range - point.x) / self.resolution))

    def create_frame(self):
        return np.zeros((int(2 * self.range / self.resolution), int(2 * self.range / self.resolution), 3),
                        dtype=np.uint8)

    def ego_state_callback(self, ego_state):
        self.ego_state = ego_state
        self.bounds_min = carla.Vector2D(ego_state.car_pos.x - self.range, ego_state.car_pos.y - self.range)
        self.bounds_max = carla.Vector2D(ego_state.car_pos.x + self.range, ego_state.car_pos.y + self.range)
        self.bounds = carla.OccupancyMap(self.bounds_min, self.bounds_max)

    def ego_path_callback(self, ego_path):
        self.ego_path = [carla.Vector2D(p.pose.position.x, p.pose.position.y) for p in ego_path.poses]

    def network_agents_callback(self, agents):
        self.network_agents = agents.agents

    def sidewalk_agents_callback(self, agents):
        self.sidewalk_agents = agents.agents

    def draw_network_lanes(self):
        frame = self.create_frame()
        if self.network_segment_map is not None and self.bounds is not None:
            for segment in self.network_segment_map.intersection(self.bounds).get_segments():
                cv2.line(frame, self.pixel(segment.start), self.pixel(segment.end), (255, 255, 255), thickness=1)
        return frame

    def draw_ego_path(self):
        frame = self.create_frame()
        if self.ego_path is not None:
            pts = np.array(
                [[self.pixel(p) for p in self.ego_path[1:]]])  # Exclude first point, which is in ego current state.
            cv2.polylines(frame, pts, False, (255, 255, 255), thickness=1)
        return frame

    def draw_ego_state(self):
        frame = self.create_frame()
        if self.ego_state is not None:
            pts = np.array([[self.pixel(carla.Vector2D(p.x, p.y)) for p in self.ego_state.car_bbox.points]])
            cv2.fillPoly(frame, pts, (0, 255, 0))
        return frame

    def draw_exo_network_state(self):
        frame = self.create_frame()
        for agent in self.network_agents:
            actor = self.world.get_actor(agent.id)
            if actor is not None:
                bounding_box = get_bounding_box(actor)
                if bounding_box is not None and all(p is not None for p in
                                                    bounding_box) and self.bounds_min is not None and self.bounds_max is not None:
                    pts = np.array([[self.pixel(p) for p in bounding_box]])
                    cv2.fillPoly(frame, pts, (0, 0, 255))
        return frame

    def draw_state_frame(self):
        frame = self.draw_network_lanes()
        overlay(frame, self.draw_ego_path())
        overlay(frame, self.draw_exo_network_state())
        overlay(frame, self.draw_ego_state())
        return frame

    def draw_info_frame(self):
        frame = self.draw_decisions()
        overlay(frame, self.draw_action_probs())
        overlay(frame, self.draw_input_images())

        return frame
