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
        self.imitation_cmd_accel, self.imitation_cmd_vel, self.imitation_cmd_steer, self.imitation_cmd_lane = \
            None, None, None, None
        self.imitation_acc_probs, self.imitation_steer_probs, self.imitation_lane_probs, self.imitation_vel_probs = \
            None, None, None, None
        self.imitation_lane_image = None
        self.imitation_hist_image, self.imitation_hist1_image, self.imitation_hist2_image, self.imitation_hist3_image \
            = None, None, None, None
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

        rospy.Subscriber('/imitation_cmd_accel', Float32, self.imitation_cmd_accel_callback, queue_size=1)
        rospy.Subscriber('/imitation_cmd_speed', Float32, self.imitation_cmd_vel_callback, queue_size=1)
        rospy.Subscriber('/imitation_cmd_steer', Float32, self.imitation_cmd_steer_callback, queue_size=1)
        rospy.Subscriber('/imitation_lane_decision', Int32, self.imitation_cmd_lane_callback, queue_size=1)
        rospy.Subscriber('/imitation_action_distribs', ActionDistrib, self.imitation_probs_callback, queue_size=1)
        rospy.Subscriber("/imitation_input_images", InputImages, self.imitation_image_callback)

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

    def imitation_cmd_accel_callback(self, accel):
        self.imitation_cmd_accel = accel.data

    def imitation_cmd_vel_callback(self, vel):
        self.imitation_cmd_vel = vel.data

    def imitation_cmd_steer_callback(self, steer):
        self.imitation_cmd_steer = steer.data

    def imitation_cmd_lane_callback(self, lane):
        self.imitation_cmd_lane = lane.data

    def imitation_probs_callback(self, action_probs):
        self.imitation_acc_probs = action_probs.acc_probs
        self.imitation_steer_probs = action_probs.steer_probs
        self.imitation_lane_probs = action_probs.lane_probs
        self.imitation_vel_probs = action_probs.vel_probs

    def imitation_image_callback(self, images):
        try:
            # print('images received of type {}'.format(images.lane.encoding))
            self.imitation_lane_image = self.process_image(images.lane)
            self.imitation_hist_image = self.process_image(images.hist0)
            self.imitation_hist1_image = self.process_image(images.hist1)
            if images.lane.encoding == '32FC1':
                self.imitation_hist2_image = self.process_image(images.hist2)
                self.imitation_hist3_image = self.process_image(images.hist3)
        except Exception as e:
            error_handler(e)

    def process_image(self, image_msg):
        try:
            image = image_msg.data
            height = image_msg.height
            width = image_msg.width
            encoding = image_msg.encoding
            border = 2
            # print('(height, width)={}, image={}'.format((height, width), image))
            if encoding == '32FC1':
                image = (255 * np.ndarray((height, width), dtype=np.float32, buffer=image)).astype(np.uint8)
                up_scale = 2
                image = cv2.resize(image, (up_scale * image.shape[0], up_scale * image.shape[1]),
                                   interpolation=cv2.INTER_AREA)
            elif encoding == '8UC1':
                image = np.ndarray((height, width), dtype=np.uint8, buffer=image)
            return cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        except Exception as e:
            print_long(e)
            raise e

    def draw_input_images(self):
        frame = self.create_frame()
        self.draw_image(frame, self.imitation_lane_image, 25.0, -28.0)
        self.draw_image(frame, self.imitation_hist_image, 25.0, -9.0)
        self.draw_image(frame, self.imitation_hist1_image, 25.0, 10.0)
        if self.imitation_hist2_image is not None:
            self.draw_image(frame, self.imitation_hist2_image, 5.0, -28.0)
            self.draw_image(frame, self.imitation_hist3_image, 5.0, -9.0)
        return frame

    def draw_image(self, frame, image, pos_x, pos_y):
        try:
            if image is not None:
                # print('image shape {}'.format(image.shape))
                sys.stdout.flush()
                (x_offset, y_offset) = self.pixel_zero_center(carla.Vector2D(pos_x, pos_y))
                frame[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1], 0] = image
                frame[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1], 1] = image
                frame[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1], 2] = image

        except Exception as e:
            error_handler(e)

    def draw_decisions(self):
        frame = self.create_frame()
        bar_anchor_y = None
        bar_anchor_x = -1.0
        bar_width = 2.0
        bar_gap = 3.0
        if self.ego_state is not None:
            if self.imitation_cmd_accel is not None:
                bar_anchor_y = 0.0
                self.draw_text(frame, 'Acc', bar_anchor_x, bar_anchor_y, txt_shift_x=0.0)
                bar_anchor_y += 5.0
                self.draw_bar(frame, bar_anchor_x, bar_anchor_y, (255, 0, 0), bar_width, self.imitation_cmd_accel)
            if self.imitation_cmd_vel is not None:
                bar_anchor_y += bar_width + bar_gap
                self.draw_text(frame, 'Vel', bar_anchor_x, bar_anchor_y, txt_shift_x=0.0)
                bar_anchor_y += 5.0
                self.draw_bar(frame, bar_anchor_x, bar_anchor_y, (255, 255, 0),
                        bar_width, self.imitation_cmd_vel/5.0)
            if self.imitation_cmd_steer is not None:
                bar_anchor_y += bar_width + bar_gap
                self.draw_text(frame, 'Ang', bar_anchor_x, bar_anchor_y, txt_shift_x=0.0)
                bar_anchor_y += 5.0
                self.draw_bar(frame, bar_anchor_x, bar_anchor_y, (0, 255, 0), bar_width, self.imitation_cmd_steer)
            if self.imitation_cmd_lane is not None:
                bar_anchor_y += bar_width + bar_gap
                self.draw_text(frame, 'Lane', bar_anchor_x, bar_anchor_y, txt_shift_x=0.0)
                bar_anchor_y += 5.0
                self.draw_bar(frame, bar_anchor_x, bar_anchor_y, (0, 0, 255), bar_width, self.imitation_cmd_lane)
        return frame

    def draw_action_probs(self):
        frame = self.create_frame()
        bar_width = 2.0
        bar_gap = 0.2
        if self.ego_state is not None:
            bar_anchor_x, bar_anchor_y = 9.0, 0.0
            if self.imitation_acc_probs is not None and len(self.imitation_acc_probs) > 0:
                self.draw_probs(frame, self.imitation_acc_probs, 'Acc_probs', (255, 0, 0), bar_anchor_x, bar_anchor_y,
                            bar_gap, bar_width)
                bar_anchor_x -= 3.0
            if self.imitation_steer_probs is not None and len(self.imitation_steer_probs) > 0:
                self.draw_probs(frame, self.imitation_steer_probs, 'Ang_probs', (0, 255, 0), bar_anchor_x, bar_anchor_y,
                            bar_gap, bar_width)
                bar_anchor_x -= 3.0
            if self.imitation_lane_probs is not None and len(self.imitation_lane_probs) > 0:
                self.draw_probs(frame, self.imitation_lane_probs, 'Lane_probs', (0, 0, 255), bar_anchor_x, bar_anchor_y,
                            bar_gap, bar_width)
                bar_anchor_x -= 3.0
            if self.imitation_vel_probs is not None and len(self.imitation_vel_probs) > 0:
                self.draw_probs(frame, self.imitation_vel_probs, 'Vel_probs',
                        (255, 255, 0), bar_anchor_x, bar_anchor_y,
                            bar_gap, bar_width)
                bar_anchor_x -= 3.0
        return frame

    def draw_probs(self, frame, probs, text, color, bar_anchor_x, bar_anchor_y, bar_gap, bar_width):
        if probs is not None:
            bar_anchor_y = 0.0
            self.draw_text(frame, text, bar_anchor_x, bar_anchor_y, txt_shift_x=0.0)
            bar_anchor_y += 10.0
            for prob in probs:
                prob = float(prob.data)
                self.draw_bar(frame, bar_anchor_x, bar_anchor_y, color, bar_width, prob)
                bar_anchor_y += bar_width + bar_gap

    def draw_bar(self, frame, bar_anchor_x, bar_anchor_y, bar_color, bar_width, bar_height):
        scale = 2.5
        base_shift_x = 25.0
        base_shift_y = 25.0
        pos_x = - base_shift_x + bar_anchor_x
        pos_y = - base_shift_y + bar_anchor_y
        bar_left_bottom = self.pixel_zero_center(carla.Vector2D(pos_x, pos_y))
        bar_right_top = self.pixel_zero_center(carla.Vector2D(pos_x + scale * bar_height, pos_y + bar_width))
        cv2.rectangle(frame, bar_left_bottom, bar_right_top, bar_color, 3, lineType=cv2.FILLED)

    def draw_text(self, frame, text, bar_anchor_x, bar_anchor_y, txt_shift_x=0.0, txt_shift_y=0.0):
        font = 0.5
        base_shift_x = 25.0
        base_shift_y = 25.0
        pos_x = - base_shift_x + bar_anchor_x
        pos_y = - base_shift_y + bar_anchor_y
        txt_pos = self.pixel_zero_center(
            carla.Vector2D(pos_x - txt_shift_x, pos_y - txt_shift_y))
        cv2.putText(frame, text, txt_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

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
