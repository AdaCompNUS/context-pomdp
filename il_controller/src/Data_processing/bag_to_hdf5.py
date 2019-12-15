'''ORGANIZE THIS MESS! use ordered dict, hist2 from future'''
'''rosbag to hdf5 file'''
'''use simulated time stamp for data association'''

import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2

sys.path.append(ros_path)

from global_params import config
from global_params import print_long

if config.pycharm_mode:
    import pyros_setup

    pyros_setup.configurable_import().configure('mysetup.cfg').activate()

import os

#
# import cv2
# import sys

sys.path.append('../')
sys.path.append('./')
import rosbag
from math import sin, cos
from collections import OrderedDict
import fnmatch
import argparse
import deepdish as dd
import numpy as np
import pdb
import math
import glob
from skimage.draw import polygon, line

from transforms import *

from msg_builder.msg import peds_car_info, ActionReward

import time
import copy

print_time = False

# default_map_dim = 1024

GOAL_LIST = [(19.9, 0), (0, -19.9), (0, 19.9), (-19.9, 0), (-7, 19.9),
             (-6, 19.9), (-5, 19.9)]

# values copied from the c++ project
ACT_CUR = 0
ACT_ACC = 1
ACT_DEC = 2
TIME_REWARD = 0.1
GOAL_REWARD = 0.0

# learning mode:
#     'im': sample fixed length samples per trajectory out-of-order
#     'rl': sample whole trajectory in order
learning_mode = 'im'

agent_file = None


class CoordFrame:
    def __init__(self):
        self.center = []
        self.origin = []
        self.x_axis = []
        self.y_axis = []


def parse_bag_file(filename=None):
    bag = None
    try:
        bag = rosbag.Bag(filename)
    except Exception as e:
        error_handler(e)
        print("when processing: " + filename)
        exit(1)

    # OrderedDicts: keys are stored exactly following the insertion order
    map_dict = OrderedDict()
    plan_dict = OrderedDict()
    ped_dict = OrderedDict()
    car_dict = OrderedDict()
    act_reward_dict = OrderedDict()
    obs_dict = OrderedDict()
    lane_dict = OrderedDict()

    for topic, msg, timestamp in bag.read_messages():
        if topic == '/map':
            map_dict[timestamp.to_nsec()] = msg
        elif topic == '/il_data':
            plan_dict[timestamp.to_nsec()] = msg.plan
            ped_dict[timestamp.to_nsec()] = msg.cur_peds.peds
            car_dict[timestamp.to_nsec()] = msg.cur_car
            act_reward_dict[timestamp.to_nsec()] = msg.action_reward
            # print('msg.action_reward={}'.format(msg.action_reward))
        elif topic == '/local_obstacles':
            obs_dict[timestamp.to_nsec()] = msg.contours
        elif topic == '/local_lanes':
            lane_dict[timestamp.to_nsec()] = msg.lane_segments

    bag.close()

    # at least 1 message should be there
    if (len(plan_dict.keys()) < 1 or len(ped_dict.keys()) < 1 or
            len(car_dict.keys()) < 1 or len(list(act_reward_dict.keys())) < 1 or
            len(list(obs_dict.keys())) < 1 or len(list(lane_dict.keys())) < 1):
        print("invalid bag file: incomplete topics" + filename)
        print("map_dict {} entries, plan_dict {} entries, ped_dict {} entries, car_dict {} entries, act_reward_dict {}"
              " entries, obs_dict {} entries, lane_dict {} entries,".format(
            len(list(map_dict.keys())),
            len(list(plan_dict.keys())),
            len(list(ped_dict.keys())),
            len(list(car_dict.keys())),
            len(list(act_reward_dict.keys())),
            len(list(obs_dict.keys())),
            len(list(lane_dict.keys()))
        ))
        is_valid = False
    else:
        is_valid = True

    return map_dict, plan_dict, ped_dict, car_dict, act_reward_dict, obs_dict, lane_dict, is_valid


def combine_topics_in_one_dict(map_dict, plan_dict, ped_dict, car_dict, act_reward_dict, obs_dict, lane_dict):
    # print("Associating time for data with %d source time steps..." % len(act_reward_dict.keys()))
    combined_dict = OrderedDict()

    hist_ts = np.zeros(config.num_hist_channels, dtype=np.int32)

    hist_agents = []
    for i in range(0, config.num_hist_channels):
        hist_agents.append(peds_car_info())

    num_acc = np.zeros(config.num_acc_bins)

    for timestamp in list(act_reward_dict.keys()):

        obs_ts_neareast = min(obs_dict, key=lambda x: abs(x - timestamp))
        local_obstacles = obs_dict[obs_ts_neareast]

        lane_ts_neareast = min(lane_dict, key=lambda x: abs(x - timestamp))
        local_lanes = lane_dict[lane_ts_neareast]

        hist_agents, hist_complete = \
            track_history(hist_ts, hist_agents, car_dict, ped_dict, timestamp)

        if not hist_complete:
            continue

        # print('act_reward_dict[timestamp].acceleration_id={}'.format(act_reward_dict[timestamp].acceleration_id))

        action_reward_data = ActionReward()
        action_reward_data.cur_speed = act_reward_dict[timestamp].cur_speed  # current velocity of the car
        action_reward_data.target_speed = act_reward_dict[timestamp].target_speed  # target velocity of the car
        action_reward_data.steering_normalized = act_reward_dict[timestamp].steering_normalized  # steering
        action_reward_data.lane_change = act_reward_dict[timestamp].lane_change  # reward
        action_reward_data.acceleration_id = act_reward_dict[timestamp].acceleration_id  # acc
        action_reward_data.step_reward = act_reward_dict[timestamp].step_reward  # reward

        combined_dict[timestamp] = {
            'agents': copy.deepcopy(hist_agents),
            'plan': plan_dict[timestamp],
            'action_reward': action_reward_data,
            'obstacles': local_obstacles,
            'lanes': local_lanes
        }

        num_acc[int(action_reward_data.acceleration_id.data)] += 1

    # print('Original acc distribution {}'.format(num_acc / np.sum(num_acc)))

    return combined_dict, map_dict


def was_near_goal(combined_dict):
    dist_thresh = 2
    reach_goal = False

    min_path_len = 1000000
    for timestamp in list(combined_dict.keys()):
        goal_dist = 1000000
        cur_plan = combined_dict[timestamp]['plan']
        if len(cur_plan.poses) <= 3:
            goal_dist = 0

        if len(cur_plan.poses) < min_path_len:
            min_path_len = len(cur_plan.poses)

        if goal_dist < dist_thresh:
            reach_goal = True
            break

    if reach_goal is False:
        print("[1] plan min step: {}".format(min_path_len))

    return reach_goal


def trim_episode_end(combined_dict):
    try:
        all_keys = list(combined_dict.keys())

        trim_start = max(len(all_keys) - 6, 0)  # trim two seconds of data
        for i in range(trim_start, len(all_keys)):
            if all_keys[i] in combined_dict.keys():
                del combined_dict[all_keys[i]]
    except Exception as e:
        print_long('combined_dict.keys()={}'.format(list(combined_dict.keys())))
        error_handler(e)

    return combined_dict


def trim_data_after_reach_goal(combined_dict, car_goal):
    if car_goal is None:
        # get goal coordinates
        start_ts = min(list(combined_dict.keys()))
        start_plan = combined_dict[start_ts]['plan']
        #
        car_goal = get_goal(start_plan)

    # get time at which car is within 2 meters of goal
    trim_time = None
    dist_thresh = 2

    min_goal_dist = 1000000
    cur_car_pos = None
    goal_dist = None

    for timestamp in all_keys:
        cur_car_pos = combined_dict[timestamp]['agents'][0].car.car_pos

        goal_dist = math.sqrt((car_goal[0] - cur_car_pos.x) ** 2 +
                              (car_goal[1] - cur_car_pos.y) ** 2)

        min_goal_dist = min(goal_dist, min_goal_dist)

        if goal_dist < dist_thresh:
            trim_time = timestamp
            break
    #
    # delete all data after trim time
    if trim_time is None:  # never reach goal
        print("[Warning] Goal proximity path min length: {}, car_goal ({},{}), car_pos ({},{}), last_dist {}".format(
            min_goal_dist, car_goal[0], car_goal[1], cur_car_pos.x, cur_car_pos.y, goal_dist))
        pass  # do_nothing
    else:
        # remove data points after reaching goal
        for i in range(all_keys.index(trim_time) + 1, len(all_keys)):
            del combined_dict[all_keys[i]]
    return True


def check_path_step_res(cur_plan):
    p1 = (cur_plan.poses[0].pose.position.x, cur_plan.poses[0].pose.position.y)
    p2 = (cur_plan.poses[1].pose.position.x, cur_plan.poses[1].pose.position.y)
    step_dist = euclid_dist(p1, p2)
    print("=======================================step resolution: ", step_dist)


def track_history(hist_ts, hist_agents, car_dict, ped_dict, timestamp):
    history_is_complete = False

    for i in reversed(range(1, len(hist_ts))):
        hist_agents[i] = peds_car_info()
        if hist_ts[i - 1] != 0:
            hist_agents[i].car = hist_agents[i - 1].car
            hist_agents[i].peds = hist_agents[i - 1].peds
            hist_ts[i] = hist_ts[i - 1]

            if config.num_hist_channels == i + 1:
                history_is_complete = True
        else:
            hist_agents[i].car = None
            hist_agents[i].peds = None

    hist_agents[0] = peds_car_info()
    hist_agents[0].car = car_dict[timestamp]
    hist_agents[0].peds = ped_dict[timestamp]
    hist_ts[0] = timestamp

    return hist_agents, history_is_complete


def get_goal(plan):
    last_point = (plan.poses[-1].pose.position.x,
                  plan.poses[-1].pose.position.y)
    goal_coord = min(GOAL_LIST, key=lambda x: euclid_dist(x, last_point))
    return goal_coord


def euclid_dist(x1, x2):
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


def point_to_pixel(x, y, coord_frame, resolution, dim):
    dir = [x - coord_frame.center[0], y - coord_frame.center[1]]
    x_coord = dir[0] * coord_frame.x_axis[0] + dir[1] * coord_frame.x_axis[1]
    y_coord = dir[0] * coord_frame.y_axis[0] + dir[1] * coord_frame.y_axis[1]

    indices = np.array(
        [(x_coord + config.image_half_size_meters) / resolution,
         (y_coord + config.image_half_size_meters) / resolution],
        dtype=np.float32)

    if np.any(indices < 0) or np.any(indices > (dim - 1)):
        return np.array([-1, -1], dtype=np.float32)

    return indices


def point_to_pixels_no_checking(x, y, coord_frame, resolution):
    dir = [x - coord_frame.center[0], y - coord_frame.center[1]]
    x_coord = dir[0] * coord_frame.x_axis[0] + dir[1] * coord_frame.x_axis[1]
    y_coord = dir[0] * coord_frame.y_axis[0] + dir[1] * coord_frame.y_axis[1]

    # print("dir {}, x_coord {}, y_coord {}".format(dir, x_coord, y_coord))
    indices = np.array(
        [(x_coord + config.image_half_size_meters) / resolution,
         (y_coord + config.image_half_size_meters) / resolution],
        dtype=np.float32)

    return indices


def get_length(idx_list_bins):
    length = 0
    for bin in idx_list_bins:
        length += len(bin)
    return length


target_portions_acc = [0.3, 0.4, 0.3]


def sample_idx(shuffled_idx, acc_list, sampled_idx_with_acc_bins, target_length, bias_sampling):
    try:
        idx = next(shuffled_idx)
        # print('Proposing idx {}'.format(idx))
        acc_id = acc_list[idx]

        if bias_sampling:
            if len(sampled_idx_with_acc_bins[acc_id]) >= target_length * target_portions_acc[acc_id]:
                return idx, False
        sampled_idx_with_acc_bins[acc_id].append(idx)
        return idx, True
    except Exception as e:
        # print(e, flush=True)
        return None, False


# now create data in the required format here don't forget to add random data for ped goal beliefs and history too
# coz pedestrians are not constant, also add car dimensions for history image
'''coordinate system should be consistent; bottom left is origin, x is 2nd dim, y is 1st dim, x increases up, 
y increases up, theta increases in anti clockwise '''


def create_h5_data(data_dict,
                   map_dict,
                   down_sample_ratio=config.default_ratio,
                   gamma=config.gamma):
    # !! down_sample_ratio should be 1/2^n, e.g., 1/2^5=0.03125 !!
    print("Processing data with %d source time steps..." % len(data_dict.keys()))

    dim, map_intensity, map_intensity_scale, map_ts, new_dim, coord_frame, raw_map_array, resolution = \
        parse_map_data(down_sample_ratio, map_dict)

    map_array, hist_env_maps, hist_car_maps, lane_map, obs_map, goal_map = \
        create_maps(map_dict, map_ts, raw_map_array)

    # get data for other topics
    output_dict = OrderedDict()
    timestamps = list(data_dict.keys())
    acc_list = [int(data['action_reward'].acceleration_id.data) for ts, data in data_dict.items()]

    # ts_acc_pairs = zip(data_dict.keys(), acc_list)

    # sample data points to h5
    # Change to sample a fixed number of points in the trajectory

    target_length, shuffled_idx = shuffle_and_sample_indices(timestamps)
    sampled_idx_with_acc_bins = [[], [], []]
    abandoned_idx = []
    sampled_idx = []
    bias_sampling = config.data_balancing

    while not (len(sampled_idx) == target_length):
        old_output_dict_length = len(output_dict.keys())

        try:
            idx, use_idx = sample_idx(shuffled_idx, acc_list, sampled_idx_with_acc_bins, target_length, bias_sampling)

            if idx is not None:
                pass  # print('idx {}, acc{}, use? {}'.format(idx, acc_list[idx], use_idx))

            if idx is None:  # list has been exhausted.
                if len(abandoned_idx) > 0 and get_length(sampled_idx_with_acc_bins) < target_length:
                    bias_sampling = False
                    shuffled_idx = iter(abandoned_idx)
                    # print('reassigning shuffled_idx: {}'.format(shuffled_idx))
                    continue
                else:
                    # print('End sampling')
                    break
            elif not use_idx:  # skip data point
                abandoned_idx.append(idx)
                # print('abandon idx: {}'.format(idx))
                continue

        except Exception as e:
            error_handler(e)
            print("--Sampling warning details: already sampled {}, to sample {}, total data {}".format(
                get_length(sampled_idx_with_acc_bins), target_length, len(timestamps)))
            return dict(output_dict), False

        ts = timestamps[idx]

        create_dict_entry(idx, output_dict)

        clear_maps(hist_env_maps, hist_car_maps, lane_map, obs_map, goal_map)

        hist_cars, hist_exo_agents = get_history_agents(data_dict, ts)

        if map_array is None:
            coord_frame = None

        agents_are_valid, _ = process_exo_agents(hist_cars, hist_exo_agents, hist_env_maps, dim, resolution,
                                                 coord_frame=coord_frame)

        if not agents_are_valid:
            return dict(output_dict), False  # report invalid peds

        process_maps(down_sample_ratio, idx, map_array, output_dict, hist_env_maps)

        process_car(idx, ts, output_dict, data_dict, hist_cars, hist_car_maps, goal_map,
                    dim, down_sample_ratio, resolution, coord_frame)

        process_actions(data_dict, idx, output_dict, ts)

        process_obstacles(idx, ts, output_dict, data_dict, hist_cars[0], obs_map, down_sample_ratio, resolution,
                          coord_frame)

        process_lanes(idx, ts, output_dict, data_dict, hist_cars[0], lane_map,
                      down_sample_ratio, resolution, coord_frame)

        process_parametric_agents(idx, output_dict, hist_exo_agents, hist_cars)

        if len(output_dict.keys()) > old_output_dict_length:
            sampled_idx.append(idx)

    process_values(data_dict, gamma, output_dict, sampled_idx, timestamps)

    if len(output_dict.keys()) == 0:
        print("[Investigate] Bag results in no data!")
        # pdb.set_trace()
        return dict(output_dict), False
    else:
        print("Creating data with %d time steps..." % len(output_dict.keys()))
        print('Sampled acc distribution: {} {} {}'.format(
            len(sampled_idx_with_acc_bins[0]), len(sampled_idx_with_acc_bins[1]), len(sampled_idx_with_acc_bins[2])))

    return dict(output_dict), True


def shuffle_and_sample_indices(timestamps):
    shuffled_idx = list(range(len(timestamps)))
    if learning_mode == 'rl':
        random.shuffle(shuffled_idx)
        shuffled_idx = iter(shuffled_idx)
        sample_length = -1
    else:
        random.shuffle(shuffled_idx)
        shuffled_idx = iter(shuffled_idx)
        sample_length = int(min(config.num_samples_per_traj, int(len(timestamps) / config.min_samples_gap)))
    return sample_length, shuffled_idx


def draw_polygon_edges(points, image, intensity, intensity_scale, is_contour):
    try:
        for i, p in enumerate(points):
            r0 = int(round(p[0]))
            c0 = int(round(p[1]))
            if i + 1 < len(points):
                r1 = int(round(points[i + 1][0]))
                c1 = int(round(points[i + 1][1]))
            elif is_contour:
                r1 = int(round(points[0][0]))
                c1 = int(round(points[0][1]))
            else:  # not a contour
                continue

            cv2.line(image, (r0, c0), (r1, c1), color=intensity * intensity_scale)
    except Exception as e:
        print('image={}'.format(image))
        error_handler(e)


def draw_car(car, image, coord_frame, down_sample_ratio, resolution, pyramid_image=False, pyramid_points=False):
    start_time = time.time()

    if car is not None:
        image_space_car = get_image_space_car(car, coord_frame, resolution)
        if config.data_type == np.float32:
            draw_polygon_edges(image_space_car, image, intensity=1.0, intensity_scale=1.0, is_contour=True)
        elif config.data_type == np.uint8:
            draw_polygon_edges(image_space_car, image, intensity=255, intensity_scale=1, is_contour=True)

    if pyramid_image:
        return image_to_pyramid_image(image, down_sample_ratio), time.time() - start_time
    if pyramid_points:
        return image_to_pyramid_pixels(image, down_sample_ratio), time.time() - start_time


def draw_car_state(car, image, coord_frame, down_sample_ratio, resolution, pyramid_image=False, pyramid_points=False):
    try:
        if car is not None:
            image_space_car_state = get_image_space_car_state(car, coord_frame, resolution)
            if config.data_type == np.float32:
                draw_polygon_edges(image_space_car_state, image, intensity=1.0, intensity_scale=1.0, is_contour=False)
            elif config.data_type == np.uint8:
                draw_polygon_edges(image_space_car_state, image, intensity=255, intensity_scale=1, is_contour=False)
        if pyramid_image:
            return image_to_pyramid_image(image, down_sample_ratio)
        if pyramid_points:
            return image_to_pyramid_pixels(image, down_sample_ratio)

    except Exception as e:
        error_handler(e)


def draw_agent(agent, car, image, coord_frame, resolution, dim, down_sample_ratio, pyramid_image=False,
               pyramid_points=False):
    start_time = time.time()
    if agent is not None and car is not None:
        image_space_agent, is_out_map = \
            get_image_space_agent(agent, car, coord_frame, resolution, dim)
        if not is_out_map:
            if config.data_type == np.float32:
                draw_polygon_edges(image_space_agent, image, intensity=0.5, intensity_scale=1.0, is_contour=True)
            elif config.data_type == np.uint8:
                draw_polygon_edges(image_space_agent, image, intensity=127, intensity_scale=1, is_contour=True)

    if pyramid_image:
        return image_to_pyramid_image(image, down_sample_ratio), time.time() - start_time
    if pyramid_points:
        return image_to_pyramid_pixels(image, down_sample_ratio), time.time() - start_time


def draw_path(path, image, coord_frame, resolution, dim, down_sample_ratio, pyramid_image=False, pyramid_points=False):
    start_time = time.time()
    path_pixels = []
    for pos in path.poses:
        pix_pos = point_to_pixel(pos.pose.position.x, pos.pose.position.y, coord_frame, resolution, dim)
        if not np.any(pix_pos == -1):  # skip path points outside the map
            path_pixels.append(pix_pos)

    if config.data_type == np.float32:
        draw_polygon_edges(path_pixels, image, intensity=1.0, intensity_scale=1.0, is_contour=False)
    elif config.data_type == np.uint8:
        draw_polygon_edges(path_pixels, image, intensity=255, intensity_scale=1, is_contour=False)

    if pyramid_image:
        return image_to_pyramid_image(image, down_sample_ratio), time.time() - start_time
    if pyramid_points:
        return image_to_pyramid_pixels(image, down_sample_ratio), time.time() - start_time


def draw_lanes(lanes, image, car, coord_frame, down_sample_ratio, resolution, pyramid_image=False,
               pyramid_points=False):
    start_time = time.time()
    # print('Rendering {} lane segments'.format(len(lanes)))

    for lane_seg in lanes:
        image_space_lane = get_image_space_lane(lane_seg, car, coord_frame, resolution)
        if config.data_type == np.float32:
            draw_polygon_edges(image_space_lane, image, intensity=1.0, intensity_scale=1.0, is_contour=False)
        elif config.data_type == np.uint8:
            draw_polygon_edges(image_space_lane, image, intensity=255, intensity_scale=1, is_contour=False)

    if pyramid_image:
        return image_to_pyramid_image(image, down_sample_ratio), time.time() - start_time
    if pyramid_points:
        return image_to_pyramid_pixels(image, down_sample_ratio), time.time() - start_time


def draw_obstacles(obstacles, image, car, coord_frame, down_sample_ratio, resolution, pyramid_image=False,
                   pyramid_points=False):
    start_time = time.time()

    for obs in obstacles:
        image_space_obstacle = get_image_space_obstacle(obs, car, coord_frame, resolution)
        if config.data_type == np.float32:
            draw_polygon_edges(image_space_obstacle, image, intensity=1.0, intensity_scale=1.0, is_contour=True)
        elif config.data_type == np.uint8:
            draw_polygon_edges(image_space_obstacle, image, intensity=255, intensity_scale=1, is_contour=False)

    if pyramid_image:
        return image_to_pyramid_image(image, down_sample_ratio), time.time() - start_time
    if pyramid_points:
        return image_to_pyramid_pixels(image, down_sample_ratio), time.time() - start_time


def process_exo_agents(hist_cars, hist_exo_agents, hist_env_maps, dim, resolution,
                       down_sample_ratio=config.default_ratio,
                       coord_frame=None):
    start = time.time()
    try:
        nearest_agent_ids = get_nearest_agent_ids(hist_exo_agents[0], hist_cars[0])  # return ped id

        if not config.use_hist_channels:
            for ts, car in enumerate(hist_cars):
                draw_car_state(car, hist_env_maps[ts], coord_frame, down_sample_ratio, resolution)

        for agent_id in nearest_agent_ids:
            agent_history = get_exo_agent_history(agent_id, hist_exo_agents)
            for ts, hist_agent in enumerate(agent_history):
                draw_agent(hist_agent, hist_cars[ts], hist_env_maps[ts], coord_frame, resolution, dim,
                           down_sample_ratio)

    except Exception as e:
        error_handler(e)

    end = time.time()
    return True, end - start


def create_maps(map_dict, map_ts, raw_map_array):
    if len(map_dict.keys()) > 0:
        return create_maps_inner(map_dict[map_ts], raw_map_array)
    else:
        return create_null_maps()


def create_maps_inner(map_dict_entry, raw_map_array):
    map_array = np.array(raw_map_array, dtype=config.data_type)
    goal_map = None
    lane_map = np.zeros((map_dict_entry.info.height, map_dict_entry.info.width), dtype=config.data_type)
    obs_map = np.zeros((map_dict_entry.info.height, map_dict_entry.info.width), dtype=config.data_type)
    if config.use_goal_channel:
        goal_map = np.zeros((map_dict_entry.info.height, map_dict_entry.info.width), dtype=config.data_type)
    hist_ped_maps = []
    hist_car_maps = []
    for i in range(config.num_hist_channels):
        hist_ped_maps.append(np.zeros((map_dict_entry.info.height, map_dict_entry.info.width), dtype=config.data_type))
        if config.use_hist_channels:
            hist_car_maps.append(
                np.zeros((map_dict_entry.info.height, map_dict_entry.info.width), dtype=config.data_type))

    return map_array, hist_ped_maps, hist_car_maps, lane_map, obs_map, goal_map


def create_null_maps():
    map_array, goal_map = None, None
    lane_map = np.zeros((config.default_map_dim, config.default_map_dim), dtype=config.data_type)
    obs_map = np.zeros((config.default_map_dim, config.default_map_dim), dtype=config.data_type)
    if config.use_goal_channel:
        goal_map = np.zeros((config.default_map_dim, config.default_map_dim), dtype=config.data_type)
    hist_ped_maps = []
    hist_car_maps = []
    for i in range(config.num_hist_channels):
        hist_ped_maps.append(np.zeros((config.default_map_dim, config.default_map_dim), dtype=config.data_type))
        if config.use_hist_channels:
            hist_car_maps.append(np.zeros((config.default_map_dim, config.default_map_dim), dtype=config.data_type))

    return map_array, hist_ped_maps, hist_car_maps, lane_map, obs_map, goal_map


def clear_maps(hist_env_maps, hist_car_maps, lane_map, obs_map, goal_map):
    null_value = 0
    if config.data_type == np.float32:
        null_value = 0.0
    elif config.data_type == np.uint8:
        null_value = 0
    lane_map.fill(null_value)
    obs_map.fill(null_value)
    if config.use_goal_channel:
        goal_map.fill(null_value)
    for hist_env_map in hist_env_maps:
        hist_env_map.fill(null_value)
    if config.use_hist_channels:
        for hist_car_map in hist_car_maps:
            hist_car_map.fill(null_value)


def reward_function(prev_steer_data, steer_data, acc_data):
    reward = 0.0
    # action penalties
    if prev_steer_data * steer_data < 0:  # penalize change of steering direction
        reward += -0.1

    if int(acc_data) == ACT_DEC or int(acc_data) == ACT_ACC:  # penalize accelarations
        reward += -0.1
    # if int(acc_data) == ACT_DEC:  # penalize accelarations
    # reward += -0.1

    # time penalty
    reward += -TIME_REWARD

    # goal reward

    reward += GOAL_REWARD

    # collision penalty
    if False:  # only collision free trajectories are used, so collision penalty should be zero
        reward += 0

    # print("calculated reward for data point:", reward)

    return reward


def process_values(data_dict, gamma, output_dict, sample_idx, timestamps):
    reward_arr = np.zeros(len(timestamps), dtype=np.float32)
    prev_steer_data = None
    for idx in range(len(timestamps)):
        ts = timestamps[idx]
        if config.reward_mode == 'data':
            reward_arr[idx] = data_dict[ts]['action_reward'].step_reward.data
        elif config.reward_mode == 'func':
            steer_data = data_dict[ts]['action_reward'].steering_normalized.data
            prev_ts = ts
            if idx >= 1:
                prev_ts = timestamps[idx - 1]
            try:
                prev_steer_data = data_dict[prev_ts]['action_reward'].steering_normalized.data
            except Exception as e:
                error_handler(e)

            acc_data = data_dict[ts]['action_reward'].acceleration_id.data
            reward_arr[idx] = reward_function(prev_steer_data, steer_data, acc_data)

        if idx in sample_idx:
            output_dict[idx]['reward'] = np.array(
                [reward_arr[idx]], dtype=np.float32)
    value_arr = np.zeros(len(timestamps), dtype=np.float32)
    # calculate value array from reward array
    for i in range(len(timestamps) - 1, -1,
                   -1):  # enumerate in the descending order
        if i == (len(timestamps) - 1):
            value_arr[i] = reward_arr[i]
        else:
            value_arr[i] = reward_arr[i] + (gamma * value_arr[i + 1])
        if i in sample_idx:
            output_dict[i]['value'] = np.array(
                [value_arr[i]], dtype=np.float32)


def process_actions(data_dict, idx, output_dict, ts):
    # parse other info
    tmp = np.zeros(3, dtype=np.float32)
    tmp[config.label_linear] = float(data_dict[ts]['action_reward'].cur_speed.data)
    tmp[config.label_cmdvel] = float(data_dict[ts]['action_reward'].target_speed.data)
    output_dict[idx]['vel'] = tmp

    output_dict[idx]['steer_norm'] = np.array(
        [float(data_dict[ts]['action_reward'].steering_normalized.data)], dtype=np.float32)
    # print('output_dict[idx][steer_norm]={}'.format(output_dict[idx]['steer_norm'][0]))
    output_dict[idx]['acc_id'] = np.array(
        [float(data_dict[ts]['action_reward'].acceleration_id.data)], dtype=np.int32)
    output_dict[idx]['lane_change'] = np.array(
        [float(data_dict[ts]['action_reward'].lane_change.data)], dtype=np.int32)


def process_car(data_idx, ts, output_dict, data_dict, hist_cars, hist_car_images, goal_image,
                dim, down_sample_ratio, resolution, coord_frame):
    process_car_inner(output_dict[data_idx], data_dict[ts], hist_cars, hist_car_images, goal_image,
                      dim, down_sample_ratio, coord_frame, resolution)


def process_car_inner(output_dict_entry, data_dict_entry, hist_cars, hist_car_images, goal_image,
                      dim, down_sample_ratio, coord_frame, resolution, mode='offline'):
    start = time.time()
    output_dict_entry['car'] = {'goal': None, 'hist': None, 'semantic': None}
    try:
        output_dict_entry['car']['semantic'] = [(car.car_speed if car else 0.0) for car in hist_cars]

        elapsed_time = 0.0
        path = data_dict_entry['plan']

        if config.use_goal_channel:
            if mode == 'offline':
                output_dict_entry['car']['goal'], elapsed_time = draw_path(path, goal_image, coord_frame, resolution,
                                                                           dim, down_sample_ratio, pyramid_points=True)
            elif mode == 'online':
                output_dict_entry['car']['goal'], elapsed_time = draw_path(path, goal_image, coord_frame, resolution,
                                                                           dim, down_sample_ratio, pyramid_image=True)
        if print_time:
            print("Construct path time: " + str(elapsed_time) + " s")

        # parse car data
        if config.use_hist_channels:
            output_dict_entry['car']['hist'], elapsed_time = construct_car_data(hist_car_images, hist_cars,
                                                                                coord_frame, resolution,
                                                                                down_sample_ratio, mode)
        if print_time:
            print("Construct car time: " + str(elapsed_time) + " s")

    except Exception as e:
        error_handler(e)
    finally:
        end = time.time()
        return end - start


def process_maps(down_sample_ratio, idx, map_array, output_dict, hist_ped_maps):
    process_maps_inner(down_sample_ratio, map_array, output_dict[idx], hist_ped_maps)


def process_maps_inner(down_sample_ratio, map_array, output_dict_entry, hist_ped_maps):
    start = time.time()
    if map_array is not None:
        map_array = rescale_image(map_array, down_sample_ratio)
        map_array = np.array(map_array, dtype=np.int32)
        map_array = normalize(map_array, config.data_type)

    hist_env_map = []
    for i in range(len(hist_ped_maps)):
        ped_array = rescale_image(hist_ped_maps[i], down_sample_ratio)
        ped_array = normalize(ped_array, config.data_type)
        if map_array is not None:
            ped_array = np.maximum(map_array, ped_array)
        hist_env_map.append(ped_array)

    # combine the static map and the pedestrian map
    output_dict_entry['maps'] = hist_env_map
    end = time.time()
    return end - start


def process_obstacles(data_idx, ts, output_dict, data_dict, cur_car, obs_image, down_sample_ratio, resolution,
                      coord_frame):
    process_obstacles_inner(output_dict[data_idx], data_dict[ts], cur_car, obs_image, down_sample_ratio, coord_frame,
                            resolution)


def process_obstacles_inner(output_dict_entry, data_dict_entry, cur_car, obs_image, down_sample_ratio, coord_frame,
                            resolution, mode='offline'):
    obstacles = data_dict_entry['obstacles']

    if mode == 'offline':
        output_dict_entry['obs'], _ = draw_obstacles(obstacles, obs_image, cur_car, coord_frame, down_sample_ratio,
                                                     resolution, pyramid_points=True)
    elif mode == 'online':
        output_dict_entry['obs'], _ = draw_obstacles(obstacles, obs_image, cur_car, coord_frame, down_sample_ratio,
                                                     resolution, pyramid_image=True)


def process_lanes(data_idx, ts, output_dict, data_dict, cur_car, lane_image, down_sample_ratio, resolution,
                  coord_frame):
    process_lanes_inner(output_dict[data_idx], data_dict[ts], cur_car, lane_image, down_sample_ratio, coord_frame,
                        resolution)


def process_lanes_inner(output_dict_entry, data_dict_entry, cur_car, lane_image, down_sample_ratio, coord_frame,
                        resolution, mode='offline'):
    start = time.time()
    lanes = data_dict_entry['lanes']

    if mode == 'offline':
        output_dict_entry['lane'], _ = draw_lanes(lanes, lane_image, cur_car, coord_frame, down_sample_ratio,
                                                  resolution,
                                                  pyramid_points=True)
    elif mode == 'online':
        output_dict_entry['lane'], _ = draw_lanes(lanes, lane_image, cur_car, coord_frame, down_sample_ratio,
                                                  resolution,
                                                  pyramid_image=True)
    end = time.time()
    return end - start


def process_parametric_agents(idx, output_dict, hist_exo_agents, hist_cars):
    process_parametric_agents_inner(output_dict[idx], hist_exo_agents, hist_cars)


def get_extent_from_bb(car_pos, car_yaw, car_bbox):
    bb_x = 0.0
    bb_y = 0.0
    tan_dir = (-math.sin(car_yaw), math.cos(car_yaw))
    along_dir = (math.cos(car_yaw), math.sin(car_yaw))
    for point in car_bbox.points:
        p = (point.x - car_pos.x, point.y - car_pos.y)
        proj = p[0] * tan_dir[0] + p[1] * tan_dir[1]
        bb_y = max(bb_y, math.fabs(proj))
        proj = p[0] * along_dir[0] + p[1] * along_dir[1]
        bb_x = max(bb_x, math.fabs(proj))

    return bb_x, bb_y


def process_parametric_agents_inner(output_dict_entry, hist_exo_agents, hist_cars):
    hist_agent_states = []
    try:
        no_info_value = -100.0

        nearest_ped_ids = get_nearest_agent_ids(hist_exo_agents[0], hist_cars[0])  # return ped id

        for ts in range(config.num_hist_channels):
            car = hist_cars[ts]
            car_bb_x, car_bb_y = get_extent_from_bb(car.car_pos, car.car_yaw, car.car_bbox)

            hist_agent_states.append(car.car_pos.x)
            hist_agent_states.append(car.car_pos.y)
            hist_agent_states.append(car.car_yaw)
            hist_agent_states.append(car_bb_x)
            hist_agent_states.append(car_bb_y)

            valid_ped = 0

            for ped_no in range(len(nearest_ped_ids)):
                if valid_ped < config.num_agents_in_map:
                    ped_id = nearest_ped_ids[ped_no]
                    hist_ped = get_exo_agent_history(ped_id, hist_exo_agents)

                    ped = hist_ped[ts]
                    if ped is not None:
                        hist_agent_states.append(ped.ped_pos.x)
                        hist_agent_states.append(ped.ped_pos.y)
                        hist_agent_states.append(ped.heading)
                        hist_agent_states.append(ped.bb_x)
                        hist_agent_states.append(ped.bb_y)
                    else:
                        for k in range(0, 5):
                            hist_agent_states.append(no_info_value)

                    valid_ped += 1

            if valid_ped < config.num_agents_in_map:
                for i in range(valid_ped, config.num_agents_in_map):
                    for k in range(0, 5):
                        hist_agent_states.append(no_info_value)

    except Exception as e:
        print(e)

    output_dict_entry['cart_agents'] = hist_agent_states


def normalize(array, data_type=np.float32):
    try:
        scale = 1.0
        if data_type == np.float32:
            scale = 1.0
        elif data_type == np.uint8:
            scale = 255
        if np.max(array) > 0:
            array = (array.astype(np.float32) / np.max(array) * scale).astype(data_type)
    except Exception as e:
        error_handler(e)
    finally:
        return array


def get_exo_agent_history(ped_id, hist_exo_agents):
    exo_agent_hist = []

    for i in range(len(hist_exo_agents)):
        try:
            if hist_exo_agents[i] is not None and ped_id in hist_exo_agents[i].keys():
                exo_agent_hist.append(hist_exo_agents[i][ped_id])
            else:
                exo_agent_hist.append(None)
        except Exception as e:
            error_handler(e)
            print("!!! Investigate: not able to extract info on ped {} "
                  "in one of history time step. Ped hist {}".format(ped_id, hist_exo_agents[i]))

    return exo_agent_hist


def history_complete(ped_id, hist_exo_agents):
    has_hist = True
    hist_count = 0

    try:
        # print('------------------------------------------------')
        # print(hist_exo_agents)
        # print('------------------------------------------------')

        for i in range(1, len(hist_exo_agents)):
            if hist_exo_agents[i] is not None:
                has_hist = has_hist and (ped_id in hist_exo_agents[i].keys())
                hist_count += int(ped_id in hist_exo_agents[i].keys())
            else:
                has_hist = False
    except Exception as e:
        error_handler(e)

    return has_hist, hist_count


def get_history_agents(data_dict, ts):
    # agent_file.write("\n====== ts {} ======\n".format(ts))

    hist_cars, hist_exo_agents, _ = \
        get_bounded_history(data_dict[ts], 'agents')

    debug_history(hist_cars, hist_exo_agents)

    return hist_cars, hist_exo_agents


def debug_history(hist_cars, hist_exo_agents):
    car = hist_cars[0]
    agents = hist_exo_agents[0]
    # print("Frame with {} agents".format(len(agents)))
    min_dist = 10000000
    max_dist = -10000000
    near_count = 0

    for id in agents.keys():
        agent = agents[id]
        dist_to_car = math.sqrt((agent.ped_pos.x - car.car_pos.x) ** 2 + (agent.ped_pos.y - car.car_pos.y) ** 2)
        if dist_to_car < 15.0:
            # print("agent {} nearby, dist {}".format(id, dist_to_car))
            near_count += 1
        min_dist = min(dist_to_car, min_dist)
        max_dist = max(dist_to_car, max_dist)

    # print("Min / max distance to car {} / {}".format(min_dist, max_dist))
    # print("{} nearby agents".format(near_count))


def get_bounded_history(data_dict_entry, flag_hist):
    hist_cars = []
    hist_exo_agents = []
    hist_count = 0
    try:
        for i in range(config.num_hist_channels):

            if data_dict_entry[flag_hist][i] is not None:
                if data_dict_entry[flag_hist][i].peds is not None:
                    hist_exo_agents.append(make_agent_id_dict(data_dict_entry[flag_hist][i].peds))
                    # print_long("hist step {} exist".format(i))
                else:
                    hist_exo_agents.append(None)
                if data_dict_entry[flag_hist][i].car is not None:
                    hist_cars.append(data_dict_entry[flag_hist][i].car)
                else:
                    hist_cars.append(None)
                hist_count += \
                    int(data_dict_entry[flag_hist][i].peds is not None or data_dict_entry[flag_hist][i].car is not None)
            else:
                hist_exo_agents.append(None)
                hist_cars.append(None)
            # agent_file.write("{},{} ".format(hist_cars[i].car_pos.x, hist_cars[i].car_pos.y))

    except Exception as e:
        error_handler(e)

    return hist_cars, hist_exo_agents, hist_count


def create_dict_entry(idx, output_dict):
    output_dict[idx] = {
        'maps': None,
        'ped': [dict({}) for x in range(0)],
        'car': {
            'goal': None,
            'hist': None,
            'semantic': None,
        },
        'cart_agents': None,
        'vel': None,
        'steer_norm': None,
        'acc_id': None,
        'lane_change': None,
        'reward': None,
        'value': None,
        'obs': None,
        'lane': None
    }


def delete_dict_entry(idx, output_dict):
    del output_dict[idx]


def parse_map_data(down_sample_ratio, map_dict):
    if len(map_dict.keys()) > 0:
        # get map data if it exists
        map_ts = list(map_dict.keys())[0]

        dim, map_intensity, map_intensity_scale, new_dim, coord_frame, raw_map_array, resolution = \
            parse_map_data_from_dict(down_sample_ratio, map_dict[map_ts])

        return dim, map_intensity, map_intensity_scale, map_ts, new_dim, coord_frame, raw_map_array, resolution
    else:
        # create dummy map if no map data.
        map_ts = None
        raw_map_array = None
        coord_frame = None
        dim, map_intensity, map_intensity_scale, new_dim, resolution = \
            create_null_map_data(down_sample_ratio)
        return dim, map_intensity, map_intensity_scale, map_ts, new_dim, coord_frame, raw_map_array, resolution


def parse_map_data_from_dict(down_sample_ratio, map_dict_entry):
    # get map data
    raw_map_array = np.asarray(map_dict_entry.data).reshape(
        (map_dict_entry.info.height, map_dict_entry.info.width))
    resolution = map_dict_entry.info.resolution
    # origin: left top corner of the image
    origin = [map_dict_entry.info.origin.position.x, map_dict_entry.info.origin.position.y]
    dim = int(map_dict_entry.info.height)  # square assumption
    coord_frame = CoordFrame()
    coord_frame.origin = origin
    coord_frame.center = [origin[0] + resolution * dim / 2.0, origin[0] + resolution * dim / 2.0]
    coord_frame.x_axis = [1, 0]
    coord_frame.y_axis = [0, 1]
    # map_intensity = np.max(raw_map_array)

    map_intensity = np.max(raw_map_array)
    map_intensity_scale = 1500.0

    new_dim = int(dim * down_sample_ratio)
    return dim, map_intensity, map_intensity_scale, new_dim, coord_frame, raw_map_array, resolution


def create_null_map_data(down_sample_ratio=config.default_ratio):
    resolution = config.image_half_size_meters * 2.0 / config.default_map_dim  # 0.0390625
    dim = config.default_map_dim
    map_intensity = 1.0
    map_intensity_scale = 1500.0

    new_dim = int(dim * down_sample_ratio)
    return dim, map_intensity, map_intensity_scale, new_dim, resolution


def rescale_image(image, down_sample_ratio=config.default_ratio):
    image1 = image.copy()

    try:
        iters = int(math.log(1 / down_sample_ratio, 2))
        for i in range(iters):
            image1 = cv2.pyrDown(image1)

    except Exception as e:
        error_handler(e)

    return image1


def image_to_pyramid_pixels(image, down_sample_ratio=config.default_ratio):
    nonzero_points = None
    try:
        # !! input points are in Euclidean space (x,y), output points are in image space (row, column) !!
        # down sample the image
        arr1 = rescale_image(image, down_sample_ratio)
        arr1 = normalize(arr1, config.data_type)
        nonzero_points = extract_nonzero_points(arr1)
    except Exception as e:
        error_handler(e)
    return nonzero_points


def image_to_pyramid_image(image, down_sample_ratio=config.default_ratio, debug=False):
    try:
        # !! input points are in Euclidean space (x,y), output points are in image space (row, column) !!
        # down sample the image
        pyramid_image = rescale_image(image, down_sample_ratio)
        pyramid_image = normalize(pyramid_image, config.data_type)

        if debug:
            print_long('image points = {}'.format(extract_nonzero_points(pyramid_image)))

        return pyramid_image
    except Exception as e:
        error_handler(e)


def extract_nonzero_points(arr1):
    format_point = None
    try:
        indexes = np.where(arr1 != 0)
        format_point = np.array(
            [(x, y, arr1[x, y]) for x, y in zip(indexes[0], indexes[1])],
            dtype=config.data_type)
    except Exception as e:
        error_handler(e)  # do nothing
    return format_point


def get_nearest_agent_ids(agents, car):
    dist_list = {}
    for idx, ped in agents.items():
        dist_list[idx] = dist(ped, car)
    dist_list = OrderedDict(sorted(dist_list.items(), key=lambda a: a[1]))
    return list(dist_list.keys())


def make_agent_id_dict(agent_list):
    agent_dict = {}
    for agent in agent_list:
        agent_dict[agent.ped_id] = agent
    return agent_dict


def dist(agent, car):
    return np.sqrt((agent.ped_pos.x - car.car_pos.x) ** 2 +
                   (agent.ped_pos.y - car.car_pos.y) ** 2)


def construct_car_data(car_images, hist_cars, coord_frame, resolution, down_sample_ratio, mode='offline'):
    car_hist_data = []
    start_time = time.time()
    try:
        for ts, car in enumerate(hist_cars):
            if mode == 'offline':
                car_hist_data.append(
                    draw_car(car, car_images[ts], coord_frame, down_sample_ratio, resolution, pyramid_points=True)[0])
            elif mode == 'online':
                car_hist_data.append(
                    draw_car(car, car_images[ts], coord_frame, down_sample_ratio, resolution, pyramid_image=True)[0])
    except Exception as e:
        error_handler(e)

    elapsed_time = time.time() - start_time
    return car_hist_data, elapsed_time


def get_image_space_car(car, coord_frame, resolution):
    if coord_frame is None:
        coord_frame = select_null_map_coord_frame(car)

    X = []
    for point in car.car_bbox.points:
        pixels = point_to_pixels_no_checking(point.x, point.y, coord_frame, resolution)
        X.append([pixels[0], pixels[1], 1])

    return np.asarray(X)


def get_image_space_car_state(car, coord_frame, resolution):
    if coord_frame is None:
        coord_frame = select_null_map_coord_frame(car)

    X = []
    car_dir = (car.car_speed * math.cos(car.car_yaw), car.car_speed * math.sin(car.car_yaw))
    pixels = point_to_pixels_no_checking(car.car_pos.x, car.car_pos.y, coord_frame, resolution)
    X.append([pixels[0], pixels[1], 1])
    pixels = point_to_pixels_no_checking(car.car_pos.x + car_dir[0],
                                         car.car_pos.y + car_dir[1],
                                         coord_frame, resolution)

    X.append([pixels[0], pixels[1], 1])
    return np.asarray(X)


def get_image_space_obstacle(obs, car, coord_frame, resolution):
    X = []
    if coord_frame is None:
        coord_frame = select_null_map_coord_frame(car)
    for point in obs.points:
        pixels = point_to_pixels_no_checking(point.x, point.y, coord_frame, resolution)
        X.append([pixels[0], pixels[1], 1])

    return np.asarray(X)


def get_image_space_lane(lane_segment, car, coord_frame, resolution):
    X = []
    if coord_frame is None:
        coord_frame = select_null_map_coord_frame(car)
    pixels = point_to_pixels_no_checking(lane_segment.start.x, lane_segment.start.y, coord_frame, resolution)
    X.append([pixels[0], pixels[1], 1])
    pixels = point_to_pixels_no_checking(lane_segment.end.x, lane_segment.end.y, coord_frame, resolution)
    X.append([pixels[0], pixels[1], 1])

    return np.asarray(X)


def select_null_map_coord_frame(car):
    frame = CoordFrame()

    frame.origin = [car.car_pos.x - config.image_half_size_meters,
                    car.car_pos.y - config.image_half_size_meters]
    frame.center = [car.car_pos.x, car.car_pos.y]
    frame.x_axis = [math.cos(car.car_yaw), math.sin(car.car_yaw)]
    frame.y_axis = [-math.sin(car.car_yaw), math.cos(car.car_yaw)]
    return frame


def get_image_space_agent(agent, car, coord_frame, resolution, dim):
    if coord_frame is None and car is not None:
        coord_frame = select_null_map_coord_frame(car)
    if agent is None:
        return None
    else:
        theta = agent.heading

        x1, y1, x2, y2, x3, y3, x4, y4 = agent.bb_y, agent.bb_x, -agent.bb_y, agent.bb_x, \
                                         -agent.bb_y, -agent.bb_x, agent.bb_y, -agent.bb_x
        # homogeneous coordinates
        agent_bb = np.asarray([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]])
        # rotate and translate the car
        T = np.asarray([[cos(theta), -sin(theta), agent.ped_pos.x],
                        [sin(theta), cos(theta), agent.ped_pos.y],
                        [0, 0, 1]])
        agent_bb = np.array([T.dot(x.T) for x in agent_bb], dtype=np.float32)

        image_space_agent = []
        for x in agent_bb:
            pixels = point_to_pixels_no_checking(x[0], x[1], coord_frame, resolution)
            image_space_agent.append([pixels[0], pixels[1], 1])
        image_space_agent = np.array(image_space_agent)

        is_out_map = check_out_map(image_space_agent, dim)

        return image_space_agent, is_out_map


def get_image_space_agent_history(agent_history, hist_cars, coord_frame, resolution, dim):
    is_out_map = True
    transformed_history = []
    try:
        i = 0
        for hist_agent in agent_history:

            transformed_agent, agent_is_out_map = \
                get_image_space_agent(hist_agent, hist_cars[i], coord_frame, resolution, dim)
            transformed_history.append(transformed_agent)

            i += 1
            # agent should be included if it enters the map ar any of the history time steps
            if agent_is_out_map is False:
                is_out_map = False
            else:
                pass  # print("frame out of map, dist to origin {}".format(dist))

    except Exception as e:
        error_handler(e)
    finally:
        return transformed_history, is_out_map


def check_out_map(X, dim):
    for pixel_point in X:
        if np.any(pixel_point < 0) or np.any(pixel_point >= dim):
            return True
    return False


# save in hdf5 format and load in hdf5 in dataloader class of pytorch
def main(bagspath, nped, start_file, end_file, thread_id):
    global agent_file
    # Walk into subdirectories recursively and collect all txt files

    agent_file = open("agent_record.txt", 'w')

    # print("Initialize agent file {}".format(agent_file))

    txt_files = collect_txt_files(bagspath)

    # filtered_files = filter_txt_files(bagspath, txt_files)

    incomplete_file_counter = 0
    investigate_files = list([])

    # print("thread %d processing %d files" % (thread_id, len(filtered_files)))

    for index, txt_file in enumerate(txt_files):

        if index >= end_file or index < start_file:
            continue

        h5_name = get_h5_name_from_txt_name(txt_file)

        if os.path.isfile(h5_name):
            print("file already processed, proceed")
            continue

        if not is_valid_file(txt_file):
            continue

        start_time = time.time()

        bag_file = find_bag_with_txt_name(txt_file)

        if bag_file != None:
            print("Thread %d: %s" % (thread_id, bag_file))
            #
            # create h5 table from the bag, one h5 for one trajectory;
            # The trajectories are later combined and reshaped in combine.py

            if True:
                #
                # Walk into subdirectories recursively and collect all txt files
                #
                map_dict, plan_dict, ped_dict, car_dict, \
                act_reward_dict, obs_dict, lane_dict, topics_complete = parse_bag_file(filename=bag_file)

                if topics_complete:  # all topics in the bag are non-empty
                    per_step_data_dict, map_dict = combine_topics_in_one_dict(
                        map_dict, plan_dict, ped_dict, car_dict, act_reward_dict, obs_dict, lane_dict)

                    per_step_data_dict = trim_episode_end(per_step_data_dict)
                    end_time = time.time()

                    print("Thread %d parsed: elapsed time %f s" % (thread_id, end_time - start_time))

                    h5_data_dict, data_valid = create_h5_data(per_step_data_dict, map_dict)
                    end_time = time.time()
                    print("Thread %d done: elapsed time %f s" % (thread_id, end_time - start_time))
                    if data_valid:
                        # save h5 data into file
                        try:
                            dd.io.save(h5_name, h5_data_dict)
                        except Exception as e:
                            print(e)
                            print('h5 name: ', h5_name)
                            print('txt file name: ', txt_file)
                            print('h5 file name from txt: ', get_h5_name_from_txt_name(txt_file))

                        end_time = time.time()
                        print("Thread %d written: elapsed time %f s" % (thread_id, end_time - start_time))
                    else:
                        print(
                            "Bag data incomplete "
                            "(investigate)")
                        investigate_files.append(txt_file)

                        continue
                else:  # topics_complete == False
                    print("Warning!!!!!: doing nothing coz invalid bag file")
                    incomplete_file_counter += 1
                    continue

    # print("number of filtered files %d" % len(filtered_files))
    print("no of files with imcomplete topics %d" % incomplete_file_counter)

    agent_file.close()


import codecs


def parse_goals_using_h5_name(h5_name, peds_goal_path):
    goal_string = 'goals_' + \
                  h5_name.split('/')[-1].split('_case')[0]
    ped_goal_file = None
    for root, dirnames, filenames in os.walk(peds_goal_path):
        for filename in fnmatch.filter(filenames,
                                       '*' + goal_string + '*'):
            # append the absolute path for the file
            ped_goal_file = os.path.join(root, filename)
    if ped_goal_file is None:
        print("no goal file found, peds_goal_path, file, file.split('.')[0], goal_string, h5_name:")
        print(peds_goal_path)
        print(goal_string)
        print(h5_name)
        assert False
        pass
    with open(ped_goal_file, 'r') as f:
        ped_goal_list = f.read().split('\n')
        ped_goal_list = [
            list(map(float,
                     x.strip().split(' '))) for x in ped_goal_list
        ]
    return ped_goal_list


def get_h5_name_from_txt_name(txt_file):
    h5_name = txt_file.split('.')[0] + '.h5'
    return h5_name


def find_bag_with_txt_name(txt_file):
    bag_file_list = glob.glob(txt_file.split('.')[0] + '_*')

    for bag_file in bag_file_list:
        if '.active' in bag_file:
            print("removing " + bag_file)
            os.remove(bag_file)

    bag_file_list = glob.glob(txt_file.split('.')[0] + '_*')

    if len(bag_file_list) == 0:
        print("no bag file found for txt")
        bag_file = None
    else:
        # sort with decending data as stated in the filename
        bag_file_list.sort(reverse=True)
        bag_file = bag_file_list[0]
    return bag_file


def filter_txt_files(bagspath, txt_files):
    # container for files to be converted to h5 data
    filtered_files = list([])
    # for debugging: temporary container of files for manual investigation later
    # Filter trajectories that don't reach goal or collide before reaching goal
    for txtfile in txt_files:
        #
        if is_valid_file(txtfile):
            filtered_files.append(txtfile)

    filtered_files.sort()
    print("%d filtered files found in %s" % (len(filtered_files), bagspath))
    # print (filtered_files, start_file, end_file)
    #
    return filtered_files


def is_valid_file(txtfile):
    reach_goal_flag = False
    collision_flag = False

    try:
        with codecs.open(txtfile, 'r', encoding='utf-8',
                         errors='ignore') as f:
            for line in f:

                if 'goal reached' in line:
                    reach_goal_flag = True
                    break

                if (('INININ' in line) or ("collision = 1" in line)) and reach_goal_flag == False:
                    collision_flag = True
                    print("colliison in " + txtfile)
                    break
    except Exception as e:
        print("Get error [{}] when opening txt file {}".format(e, txtfile))

    # Debugging code
    return True

    if reach_goal_flag is True and collision_flag is False:
        return True
    elif reach_goal_flag is False:
        print("goal not reached in " + txtfile)
        return False
    else:
        return False


def collect_txt_files(bagspath):
    txt_files = list([])
    for root, dirnames, filenames in os.walk(bagspath):
        if "_debug" not in root:
            # print("subfolder {} found".format(root))
            for filename in fnmatch.filter(filenames, '*.txt'):
                # append the absolute path for the file
                txt_files.append(os.path.join(root, filename))
    print("%d files found in %s" % (len(txt_files), bagspath))
    return txt_files


if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bagspath',
        type=str,
        default='BTSRL_driving_data/may16',
        help='Path to data file')
    parser.add_argument(
        '--peds_goal_path',
        type=str,
        default='../../Maps',
        help='Path to pedestrian goals')

    parser.add_argument(
        '--nped',
        type=int,
        default=0,
        help='Number of neighbouring peds to consider')
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Number of neighbouring peds to consider')
    parser.add_argument(
        '--end',
        type=int,
        default=100000,
        help='Number of neighbouring peds to consider')

    bagspath = parser.parse_args().bagspath
    peds_goal_path = parser.parse_args().peds_goal_path
    start_file = parser.parse_args().start
    end_file = parser.parse_args().end

    main(bagspath, parser.parse_args().nped, start_file, end_file, 0)



else:  # Code to be executed by importing the file
    # print ("not called from main")
    # map_list, plan_list, ped_list, pomdp_act_list, vel_list, is_valid = parse_bag_file(
    #     filename=
    #     '/home/aseem/PanPan/BTS_RL_NN/BTSRL_driving_data/may16/map_14_12_14_goal_19d9_0/14_12_14_goal_19d9_0_sample-54-2-1_2018-05-15-19-33-27.bag'
    # )
    # #
    # per_step_data_dict, map_dict = data_associate_time(
    #     map_list, plan_list, ped_list, pomdp_act_list, vel_list)
    # h5_data_dict = create_data(per_step_data_dict, map_dict)
    # dd.io.save(
    #     '/home/aseem/PanPan/BTS_RL_NN/BTSRL_driving_data/cross_west_south/sample-0.h5', h5_data_dict)
    pass
