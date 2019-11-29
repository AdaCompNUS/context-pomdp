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
import visualization

from geometry_msgs.msg import Twist
from msg_builder.msg import peds_car_info, ActionReward

import time
import copy

default_map_dim = 1024

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

    for timestamp in list(act_reward_dict.keys()):

        obs_ts_neareast = min(obs_dict, key=lambda x: abs(x - timestamp))
        local_obstacles = obs_dict[obs_ts_neareast]

        lane_ts_neareast = min(lane_dict, key=lambda x: abs(x - timestamp))
        local_lanes = lane_dict[lane_ts_neareast]

        hist_agents, hist_complete = \
            track_history(hist_ts, hist_agents, car_dict, ped_dict, timestamp)

        if not hist_complete:
            continue

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
    all_keys = list(combined_dict.keys())

    trim_start = len(all_keys) - 6  # trim two seconds of data
    for i in range(trim_start, len(all_keys)):
        del combined_dict[all_keys[i]]

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


def point_to_indices(x, y, origin, resolution, dim):
    indices = np.array(
        [(x - origin[0]) / resolution,
         (y - origin[1]) / resolution],
        dtype=np.float32)
    if np.any(indices < 0) or np.any(indices > (dim - 1)):
        return np.array([-1, -1], dtype=np.float32)

    return indices


# now create data in the required format here don't forget to add random data for ped goal beliefs and history too
# coz pedestrians are not constant, also add car dimensions for history image
'''coordinate system should be consistent; bottom left is origin, x is 2nd dim, y is 1st dim, x increases up, 
y increases up, theta increases in anti clockwise '''


def create_h5_data(data_dict,
                   map_dict,
                   down_sample_ratio=0.03125,
                   gamma=config.gamma):
    # !! down_sample_ratio should be 1/2^n, e.g., 1/2^5=0.03125 !!
    print("Processing data with %d source time steps..." % len(data_dict.keys()))

    dim, map_intensity, map_intensity_scale, map_ts, new_dim, origin, raw_map_array, resolution = \
        parse_map_data(down_sample_ratio, map_dict)

    # get data for other topics
    output_dict = OrderedDict()
    timestamps = list(data_dict.keys())

    # sample data points to h5
    # Change to sample a fixed number of points in the trajectory

    sample_idx, sample_length, sample_shuffled_idx = shuffle_and_sample_indices(timestamps)

    while not (len(sample_idx) == sample_length):
        old_output_dict_length = len(output_dict.keys())

        try:
            idx = next(sample_shuffled_idx)
        except Exception as e:
            error_handler(e)
            print("--Sampling warning details: already sampled {}, to sample {}, total data {}".format(
                len(sample_idx), sample_length, len(timestamps)))
            return dict(output_dict), False

        ts = timestamps[idx]

        create_dict_entry(idx, output_dict)

        map_array, hist_env_maps = \
            create_maps(map_dict, map_ts, raw_map_array)

        hist_cars, hist_exo_agents = get_history_agents(data_dict, ts)

        if map_array is None:
            origin = None

        agents_are_valid = process_exo_agents(hist_cars, hist_exo_agents, hist_env_maps, dim, resolution, map_intensity,
                                              map_intensity_scale, origin)

        if not agents_are_valid:
            return dict(output_dict), False  # report invalid peds

        if map_array is None:
            origin = select_null_map_origin(hist_cars, 0)

        process_maps(down_sample_ratio, idx, map_array, output_dict, hist_env_maps)

        process_car(idx, ts, output_dict, data_dict, hist_cars, dim, down_sample_ratio, resolution, origin)

        process_actions(data_dict, idx, output_dict, ts)

        process_obstacles(idx, ts, output_dict, data_dict, dim, down_sample_ratio, resolution, origin)

        process_lanes(idx, ts, output_dict, data_dict, dim, down_sample_ratio, resolution, origin)

        process_parametric_agents(idx, output_dict, hist_exo_agents, hist_cars)

        if len(output_dict.keys()) > old_output_dict_length:
            sample_idx.append(idx)

    process_values(data_dict, gamma, output_dict, sample_idx, timestamps)

    if len(output_dict.keys()) == 0:
        print("[Investigate] Bag results in no data!")
        pdb.set_trace()
    else:
        print("Creating data with %d time steps..." % len(output_dict.keys()))

    return dict(output_dict), True


def shuffle_and_sample_indices(timestamps):
    sample_idx = []
    sample_shuffled_idx = list(range(len(timestamps)))
    if learning_mode == 'rl':
        random.shuffle(sample_shuffled_idx)
        sample_shuffled_idx = iter(sample_shuffled_idx)
        sample_length = -1
    else:
        random.shuffle(sample_shuffled_idx)
        sample_shuffled_idx = iter(sample_shuffled_idx)
        sample_length = min(config.num_samples_per_traj, len(timestamps) / 6)
    return sample_idx, sample_length, sample_shuffled_idx


def process_exo_agents(hist_cars, hist_exo_agents, hist_env_maps, dim, resolution, map_intensity, map_intensity_scale,
                       origin=None):
    try:
        nearest_agent_ids = get_nearest_agent_ids(hist_exo_agents[0], hist_cars[0])  # return ped id
        valid_agent = 0

        for agent_no in range(len(nearest_agent_ids)):
            #
            agent_id = nearest_agent_ids[agent_no]

            hist_agent = get_exo_agent_history(agent_id, hist_exo_agents)

            origins = []
            for i in range(config.num_hist_channels):
                if origin is None:
                    origins.append(select_null_map_origin(hist_cars, i))
                else:
                    origins.append(origin)

            # calculate position in map image
            hist_image_space_agent, is_out_map = get_image_space_agent_history(hist_agent, origins, resolution, dim)

            # discard agents outside the map
            if is_out_map:
                # print("Agent {} outside of map, ".format(agent_id))
                continue
                #
            else:
                # Project the rest of agents in environment maps
                fill_agent_map(hist_image_space_agent, hist_env_maps, map_intensity, map_intensity_scale, dim)
                valid_agent += 1

        # print("{} peds used in frame env_map".format(valid_agent))
        if valid_agent == 0:
            print("Warning!!!!!!: no exo-agent appeared in the local map")

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return True


def process_ped(data_idx, ped_idx, output_dict, hist_positions,
                dim, down_sample_ratio):
    # k-nearest neighbours
    # parse ped goal, and fill history entries

    if data_idx is None:
        process_ped_inner(ped_idx, output_dict, hist_positions,
                          dim, down_sample_ratio)
    else:
        process_ped_inner(ped_idx, output_dict[data_idx], hist_positions,
                          dim, down_sample_ratio)


def process_ped_inner(ped_idx, output_dict_entry, hist_positions,
                      dim, down_sample_ratio):
    # k-nearest neighbours
    # parse ped goal, and fill history entries
    if config.num_hist_channels > 2:
        output_dict_entry['ped'][ped_idx] = construct_ped_data(dim, down_sample_ratio, hist_positions)
    else:
        output_dict_entry['ped'][ped_idx] = construct_ped_data(dim, down_sample_ratio, hist_positions)


def process_blank_ped(data_idx, ped_idx, output_dict):
    process_blank_ped_inner(output_dict[data_idx], ped_idx)


def process_blank_ped_inner(output_dict_entry, valid_ped):
    if config.num_hist_channels > 2:
        output_dict_entry['ped'][valid_ped] = construct_blank_ped_data()
    else:
        output_dict_entry['ped'][valid_ped] = construct_blank_ped_data()


def create_maps(map_dict, map_ts, raw_map_array):
    if len(map_dict.keys()) > 0:
        return create_maps_inner(map_dict[map_ts], raw_map_array)
    else:
        return create_null_maps()


def create_maps_inner(map_dict_entry, raw_map_array):
    map_array = np.array(raw_map_array, dtype=np.float32)
    hist_ped_maps = []
    for i in range(config.num_hist_channels):
        hist_ped_maps.append(np.zeros((map_dict_entry.info.height, map_dict_entry.info.width), dtype=np.float32))

    return map_array, hist_ped_maps


def create_null_maps():
    map_array = None
    hist_ped_maps = []
    for i in range(config.num_hist_channels):
        hist_ped_maps.append(np.zeros((default_map_dim, default_map_dim), dtype=np.float32))

    return map_array, hist_ped_maps


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
                print(e)
                pdb.set_trace()

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
    output_dict[idx]['acc_id'] = np.array(
        [float(data_dict[ts]['action_reward'].acceleration_id.data)], dtype=np.int32)
    output_dict[idx]['lane_change'] = np.array(
        [float(data_dict[ts]['action_reward'].lane_change.data)], dtype=np.int32)


def process_car(data_idx, ts, output_dict, data_dict, hist_cars, dim, down_sample_ratio, resolution, origin):
    process_car_inner(output_dict[data_idx], data_dict[ts], hist_cars, dim, down_sample_ratio,
                      origin, resolution)


def process_car_inner(output_dict_entry, data_dict_entry, hist_cars, dim, down_sample_ratio,
                      origin, resolution):
    try:
        print_time = False
        path = data_dict_entry['plan']
        # parse path data
        start_time = time.time()
        path_data = construct_path_data(path, origin, resolution, dim,
                                        down_sample_ratio)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Construct path time: " + str(elapsed_time) + " s")

        # parse car data
        start_time = time.time()

        origins = []
        for i in range(config.num_hist_channels):
            origins.append(select_null_map_origin(hist_cars, i))

        output_dict_entry['car'] = construct_car_data(origins, resolution, dim, down_sample_ratio,
                                                      hist_cars, path_data=path_data)

        # print('----------------------------------------')
        # print('len of output_dict_entry[car][hist] = {}'.format(len(output_dict_entry['car']['hist'])))
        # print(output_dict_entry['car']['hist'])
        # print('----------------------------------------')

        if print_time:
            elapsed_time = time.time() - start_time
            print("Construct car time: " + str(elapsed_time) + " s")

    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def process_maps(down_sample_ratio, idx, map_array, output_dict, hist_ped_maps):
    process_maps_inner(down_sample_ratio, map_array, output_dict[idx], hist_ped_maps)


def process_maps_inner(down_sample_ratio, map_array, output_dict_entry, hist_ped_maps):
    if map_array is not None:
        map_array = rescale_image(map_array, down_sample_ratio)
        map_array = np.array(map_array, dtype=np.int32)
        # print("rescaled map max", map_array.max())
        map_array = normalize(map_array)
        # print("normalized map max", map_array.max())

    hist_env_map = []
    for i in range(len(hist_ped_maps)):
        ped_array = rescale_image(hist_ped_maps[i], down_sample_ratio)
        # print("rescaled ped_array max", ped_array.max())
        ped_array = normalize(ped_array)
        # print("normalized ped_array max", ped_array.max())
        if map_array is not None:
            ped_array = np.maximum(map_array, ped_array)
        # print("merged ped_array max", np.maximum(map_array, ped_array).max())
        hist_env_map.append(ped_array)

    # combine the static map and the pedestrian map
    output_dict_entry['maps'] = hist_env_map

    if config.visualize_raw_data:
        visualization.visualized_exo_agent_data(hist_env_map, root='Data_processing/')


def process_obstacles(data_idx, ts, output_dict, data_dict, dim, down_sample_ratio, resolution, origin):
    process_obstacles_inner(output_dict[data_idx], data_dict[ts], dim, down_sample_ratio,
                      origin, resolution)


def process_obstacles_inner(output_dict_entry, data_dict_entry, dim, down_sample_ratio,
                      origin, resolution):
    obstacles = data_dict_entry['obstacles']
    output_dict_entry['obs'] = construct_obs_data(origin, resolution, dim, down_sample_ratio, obstacles)
    pass


def construct_obs_data(origin, resolution, dim, down_sample_ratio, obstacles):
    obs_points = get_obs_points(obstacles, origin, dim, down_sample_ratio, resolution)
    return obs_points


def get_obs_points(obstacles, origin, dim_high_res, down_sample_ratio, resolution):
    # create transformation matrix of point, multiply with 4 points in local coordinates to get global coordinates
    obs_points = None

    obs_edge_pixels = []
    for obs in obstacles:
        print_time = False
        start_time = time.time()
        image_space_obstacle = get_image_space_obstacle(obs, origin, resolution)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Obstacle prepare time: " + str(elapsed_time) + " s")

        # construct the image
        start_time = time.time()
        obs_edge_pixels = obs_edge_pixels + fill_polygon_edges(image_space_obstacle)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Fill obstacle time: " + str(elapsed_time) + " s")

    try:
        start_time = time.time()
        obs_points = get_pyramid_image_points(obs_edge_pixels, dim_high_res, down_sample_ratio,
                                              draw_image=True, draw_flag='obs')
        if print_time:
            elapsed_time = time.time() - start_time
            print("Obs pyramid time: " + str(elapsed_time) + " s")
    except Exception as e:
        error_handler(e)

    return obs_points


def process_lanes(data_idx, ts, output_dict, data_dict, dim, down_sample_ratio, resolution, origin):
    process_lanes_inner(output_dict[data_idx], data_dict[ts], dim, down_sample_ratio,
                      origin, resolution)


def process_lanes_inner(output_dict_entry, data_dict_entry, dim, down_sample_ratio,
                      origin, resolution):
    lanes = data_dict_entry['lanes']
    output_dict_entry['lane'] = construct_lane_data(origin, resolution, dim, down_sample_ratio, lanes)
    pass


def construct_lane_data(origin, resolution, dim, down_sample_ratio, lanes):
    lane_points = get_lane_points(lanes, origin, dim, down_sample_ratio, resolution)
    return lane_points


def get_lane_points(lanes, origin, dim_high_res, down_sample_ratio, resolution):
    # create transformation matrix of point, multiply with 4 points in local coordinates to get global coordinates
    lane_points = None

    lane_pixels = []
    for lane_seg in lanes:
        print_time = False
        start_time = time.time()
        image_space_lane = get_image_space_lane(lane_seg, origin, resolution)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Lane prepare time: " + str(elapsed_time) + " s")

        # construct the image
        start_time = time.time()
        lane_pixels = lane_pixels + fill_polygon_edges(image_space_lane)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Fill lane_seg time: " + str(elapsed_time) + " s")

    try:
        start_time = time.time()
        lane_points = get_pyramid_image_points(lane_pixels, dim_high_res, down_sample_ratio,
                                              draw_image=True, draw_flag='lane')
        if print_time:
            elapsed_time = time.time() - start_time
            print("Lane pyramid time: " + str(elapsed_time) + " s")
    except Exception as e:
        error_handler(e)

    return lane_points


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


def normalize(ped_array):
    if np.max(ped_array) > 0:
        ped_array = ped_array / np.max(ped_array)
    return ped_array


def get_map_indices(hist_agent, dim, origin, resolution):
    hist_positions = []
    is_out_map = True  # it will be true if all hist pos are outside of the map
    for i in range(len(hist_agent)):
        hist_positions.append(np.array([], dtype=np.float32))
        try:
            ped = hist_agent[i]

            if ped is not None:
                hist_positions[i] = point_to_indices(ped.ped_pos.x, ped.ped_pos.y, origin, resolution, dim)
                is_out_map = is_out_map and np.any(hist_positions[i] == -1)
            else:
                hist_positions[i] = np.array([None, None], dtype=np.float32)
        except Exception as e:
            error_handler(e)
            pdb.set_trace()

    return hist_positions, is_out_map


def get_polygons(hist_agent, dim, origin, resolution):
    hist_positions = []
    is_out_map = True  # it will be true if all hist pos are outside of the map
    for i in range(len(hist_agent)):
        hist_positions.append(np.array([], dtype=np.float32))
        try:
            ped = hist_agent[i]

            if ped is not None:
                hist_positions[i] = point_to_indices(ped.ped_pos.x, ped.ped_pos.y, origin, resolution, dim)
                is_out_map = is_out_map and np.any(hist_positions[i] == -1)
            else:
                hist_positions[i] = np.array([None, None], dtype=np.float32)
        except Exception as e:
            error_handler(e)
            pdb.set_trace()

    return hist_positions, is_out_map


def add_to_ped_map(hist_ped, hist_ped_maps, dim, map_intensity, map_intensity_scale, origin, resolution):
    for ts in range(len(hist_ped)):
        ped = hist_ped[ts]
        if ped is not None:
            ped_map_array = hist_ped_maps[ts]
            fill_ped_in_map(dim, map_intensity, map_intensity_scale, origin, ped, ped_map_array, resolution)
        else:
            pass
            # print("Ped hist {} empty".format(ts))


def fill_agent_map(hist_image_space_agent, hist_env_maps, map_intensity, map_intensity_scale, dim):
    ts = 0
    for agent in hist_image_space_agent:
        if agent is not None:
            env_map_array = hist_env_maps[ts]
            if not check_out_map(agent, dim):
                agent_edge_pixels = fill_polygon_edges(agent)
                fill_pixels_in_map(map_intensity, map_intensity_scale, agent_edge_pixels, env_map_array)
        else:
            pass
        ts += 1


def fill_ped_in_map(dim, map_intensity, map_intensity_scale, origin, ped, ped_map_array, resolution):
    position = point_to_indices(ped.ped_pos.x, ped.ped_pos.y, origin, resolution, dim)
    if np.all(position != -1):
        ped_map_array[int(round(position[1]))][int(round(position[0]))] = map_intensity * map_intensity_scale
    # return position


def fill_pixels_in_map(map_intensity, map_intensity_scale, positions, env_map_array):
    try:
        for position in positions:
            position = np.array(position, dtype=np.float32)
            if np.all(position != -1) and np.all(position >= 0) \
                    and np.all(position < env_map_array.shape[0]):
                env_map_array[int(round(position[1]))][int(round(position[0]))] = map_intensity * map_intensity_scale
            else:
                pass
                # print("point out of map bound: point ({},{}), map dim {}.".format(
                #     position[0], position[1], env_map_array.shape[0]))
    except Exception as e:
        error_handler(e)
        pdb.set_trace()


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
            pdb.set_trace()

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
        pdb.set_trace()

    return has_hist, hist_count


def get_history_agents(data_dict, ts):
    # agent_file.write("\n====== ts {} ======\n".format(ts))

    hist_cars, hist_exo_agents = \
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

    try:
        for i in range(config.num_hist_channels):

            # print("--------------------------------------------------------------------")
            # print(data_dict_entry[flag_hist][i])
            # print("--------------------------------------------------------------------")

            if data_dict_entry[flag_hist][i].peds is not None and data_dict_entry[flag_hist][i].car is not None:
                hist_exo_agents.append(make_agent_id_dict(data_dict_entry[flag_hist][i].peds))
                hist_cars.append(data_dict_entry[flag_hist][i].car)
            else:
                hist_exo_agents.append(None)
                hist_cars.append(None)
            # agent_file.write("{},{} ".format(hist_cars[i].car_pos.x, hist_cars[i].car_pos.y))

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return hist_cars, hist_exo_agents


def create_dict_entry(idx, output_dict):
    output_dict[idx] = {
        'maps': None,
        'ped': [dict({}) for x in range(0)],
        'car': {
            'goal': None,
            'hist': None,
            'car_state': None,
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

        dim, map_intensity, map_intensity_scale, new_dim, origin, raw_map_array, resolution = \
            parse_map_data_from_dict(down_sample_ratio, map_dict[map_ts])

        return dim, map_intensity, map_intensity_scale, map_ts, new_dim, origin, raw_map_array, resolution
    else:
        # create dummy map if no map data.
        map_ts = None
        raw_map_array = None
        origin = None
        dim, map_intensity, map_intensity_scale, new_dim, resolution = \
            create_null_map_data(down_sample_ratio)
        return dim, map_intensity, map_intensity_scale, map_ts, new_dim, origin, raw_map_array, resolution


def parse_map_data_from_dict(down_sample_ratio, map_dict_entry):
    # get map data
    raw_map_array = np.asarray(map_dict_entry.data).reshape(
        (map_dict_entry.info.height, map_dict_entry.info.width))
    resolution = map_dict_entry.info.resolution
    origin = map_dict_entry.info.origin.position.x, map_dict_entry.info.origin.position.y
    dim = int(map_dict_entry.info.height)  # square assumption
    # map_intensity = np.max(raw_map_array)

    map_intensity = np.max(raw_map_array)
    map_intensity_scale = 1500.0

    new_dim = int(dim * down_sample_ratio)
    return dim, map_intensity, map_intensity_scale, new_dim, origin, raw_map_array, resolution


def create_null_map_data(down_sample_ratio):
    resolution = 0.0390625
    dim = default_map_dim
    map_intensity = 1.0
    map_intensity_scale = 1500.0

    new_dim = int(dim * down_sample_ratio)
    return dim, map_intensity, map_intensity_scale, new_dim, resolution


def rescale_image(image,
                  down_sample_ratio=0.03125):
    image1 = image.copy()

    try:
        iters = int(math.log(1 / down_sample_ratio, 2))
        for i in range(iters):
            image1 = cv2.pyrDown(image1)

    except Exception as e:
        error_handler(e)

    return image1


def get_pyramid_image_points(points,
                             dim=default_map_dim,
                             down_sample_ratio=0.03125,
                             intensities=False,
                             draw_image=False, draw_flag='car'):
    format_point = None

    try:
        # !! input points are in Euclidean space (x,y), output points are in image space (row, column) !!
        arr = np.zeros((dim, dim), dtype=np.float32)
        fill_image_with_points(arr, dim, points, intensities)
        # down sample the image
        arr1 = rescale_image(arr, down_sample_ratio)

        if draw_image and config.visualize_raw_data:
            if 'car' in draw_flag:
                visualization.visualize_image(arr1, root='Data_processing/', subfolder='h5_car_image')
            elif 'lane' in draw_flag:
                visualization.visualize_image(arr1, root='Data_processing/', subfolder='h5_lane_image')
            elif 'obs' in draw_flag:
                visualization.visualize_image(arr1, root='Data_processing/', subfolder='h5_obs_image')

        format_point = extract_nonzero_points(arr1)
    except Exception as e:
        error_handler(e)
        pdb.set_trace()
    return format_point


def extract_nonzero_points(arr1):
    format_point = None
    try:
        indexes = np.where(arr1 != 0)
        format_point = np.array(
            [(x, y, arr1[x, y]) for x, y in zip(indexes[0], indexes[1])],
            dtype=np.float32)
    except Exception as e:
        error_handler(e)  # do nothing
    return format_point


def fill_image_with_points(arr, dim, points, intensities=False):
    intensity = 1.0
    for point in points:
        try:
            point = np.array(point, dtype=np.float32)
            if np.any(point > (dim - 1)) or np.any(point < 0):
                continue
            if intensities:
                arr[int(round(point[1])), int(round(point[0]))] = point[2]
            else:
                arr[int(round(point[1])), int(round(point[0]))] = intensity
        except Exception as e:
            error_handler(e)  # do nothing
            pass


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


def construct_ped_data(dim, down_sample_ratio, hist_positions):
    ped_output_dict = {'hist': [list([])]}
    print_time = True

    try:
        # create original scale array, downscale using pyramid scheme, get non zero indexes and values
        hist_positions_in_image = []

        for i in range(hist_positions):
            position = hist_positions[i]
            if position.size:
                start_time = time.time()
                hist_positions_in_image.append(get_pyramid_image_points(list([position]), dim=dim,
                                                                        down_sample_ratio=down_sample_ratio))
                if print_time:
                    elapsed_time = time.time() - start_time
                    print("ped pyramid time: " + str(elapsed_time) + " s")
            else:
                hist_positions_in_image.append(np.array([], dtype=np.float32))

        ped_output_dict['hist'] = hist_positions_in_image

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return ped_output_dict


def construct_blank_ped_data():
    ped_output_dict = {'goal': list([]), 'hist1': list([]), 'hist2': list([]), 'hist3': list([]), 'hist4': list([])}

    try:
        # create original scale array, downscale using pyramid scheme, get non zero indexes and values
        positions_in_image = np.array([], dtype=np.float32)
        prev_positions_in_image = np.array([], dtype=np.float32)

        # if config.num_hist_channels > 2:
        prev_positions2_in_image = np.array([], dtype=np.float32)
        prev_positions3_in_image = np.array([], dtype=np.float32)

        # get goal coordinates in meters
        goal_coords = np.array([], dtype=np.float32)

        try:
            assert (len(goal_coords))
            ped_output_dict['goal'] = goal_coords
        except Exception as e:
            error_handler(e)
            pdb.set_trace()

        ped_output_dict['hist1'] = positions_in_image
        ped_output_dict['hist2'] = prev_positions_in_image
        ped_output_dict['hist3'] = prev_positions2_in_image
        ped_output_dict['hist4'] = prev_positions3_in_image
    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return ped_output_dict


def construct_path_data(path, origin, resolution, dim, down_sample_ratio):
    path_output_dict = {'path': list([])}
    for pos in path.poses:
        pix_pos = np.array(
            [(pos.pose.position.x - origin[0]) / resolution,
             (pos.pose.position.y - origin[1]) / resolution],
            dtype=np.float32)
        if np.any(pix_pos < 0) or np.any(pix_pos > (dim - 1)):
            continue  # skip path points outside the map
        else:
            path_output_dict['path'].append(pix_pos)
    path_output_dict['path'] = np.array(
        path_output_dict['path'], dtype=np.float32)
    path_output_dict['path'] = get_pyramid_image_points(
        path_output_dict['path'], dim=dim, down_sample_ratio=down_sample_ratio)
    return path_output_dict


def construct_car_data(origins, resolution, dim, down_sample_ratio, hist_cars, path_data=None):
    car_output_dict = {'goal': list([]), 'hist': [], 'car_state': []}
    try:
        # 'goal' contains the rescaled path in pixel points
        car_output_dict['goal'] = path_data['path']

        if config.use_hist_channels:
            for i in range(len(hist_cars)):
                draw = False
                if i == 0:
                    draw = True

                car_points = get_car_points(hist_cars[i], origins[i], dim, down_sample_ratio, resolution, draw)
                car_output_dict['hist'].append(car_points)
        else:
            for i in range(len(hist_cars)):
                car_points = get_car_state_points(hist_cars[i], origins[i], dim, down_sample_ratio, resolution)
                car_output_dict['car_state'].append(car_points)

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return car_output_dict


def get_car_points(car, origin, dim_high_res, down_sample_ratio, resolution, draw_image=False):
    # create transformation matrix of point, multiply with 4 points in local coordinates to get global coordinates
    car_points = None

    if car is not None:
        print_time = False
        start_time = time.time()
        image_space_car = get_image_space_car(car, origin, resolution)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Car prepare time: " + str(elapsed_time) + " s")

        # construct the image
        start_time = time.time()
        car_edge_pixels = fill_polygon_edges(image_space_car)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Fill car time: " + str(elapsed_time) + " s")

        try:
            start_time = time.time()
            car_points = get_pyramid_image_points(car_edge_pixels, dim_high_res, down_sample_ratio,
                                                  draw_image=draw_image, draw_flag='car')
            if print_time:
                elapsed_time = time.time() - start_time
                print("Car pyramid time: " + str(elapsed_time) + " s")
        except Exception as e:
            error_handler(e)

    return car_points


def normalize_intensity(points, target_intensity):
    intensities = points[:,2]
    max_intensity = np.max(intensities)
    intensities = intensities / max_intensity * target_intensity
    points[:, 2] = intensities


def get_car_state_points(car, origin, dim_high_res, down_sample_ratio, resolution):
    # create transformation matrix of point, multiply with 4 points in local coordinates to get global coordinates
    car_state_points = None

    if car is not None:
        image_space_car_state = get_image_space_car_state(car, origin, resolution)
        # construct the image
        car_state_edge_pixels = fill_polygon_edges(image_space_car_state)
        # print("[bag2hdf] num points in car_state_edge_pixels: {}".format(len(car_state_edge_pixels)))
        try:
            car_state_points = get_pyramid_image_points(car_state_edge_pixels, dim_high_res, down_sample_ratio,
                                                  draw_flag='car_state')
            # print("[bag2hdf] num points in car_state_points: {}".format(len(car_state_points)))
            normalize_intensity(car_state_points, 2.0)
        except Exception as e:
            error_handler(e)

    return car_state_points


def get_image_points_local_pyramid(car_edge_points, dim_high_res, down_sample_ratio, print_time, start_time):
    # allocate high-res and low-res images
    image_high_res = np.zeros((dim_high_res, dim_high_res), dtype=np.float32)
    image_low_res = np.zeros((config.imsize, config.imsize), dtype=np.float32)
    if print_time:
        elapsed_time = time.time() - start_time
        print("Alloc image time: " + str(elapsed_time) + " s")
    start_time = time.time()
    fill_image_with_points(image_high_res, dim_high_res, car_edge_points)
    if print_time:
        elapsed_time = time.time() - start_time
        print("Fill image time: " + str(elapsed_time) + " s")
    start_time = time.time()
    car_points = get_car_points_in_low_res_image(car_edge_points, image_high_res, image_low_res, dim_high_res,
                                                 down_sample_ratio)
    return car_points, start_time


def get_car_points_in_low_res_image(car_edge_points, image_high_res, image_low_res, dim, down_sample_ratio):
    # find the corresponding range in the low-res image
    x_range_low_res, y_range_low_res = get_car_range_low_res(car_edge_points, down_sample_ratio)
    # find the high-res image range
    x_range_low_res_dialected, y_range_low_res_dialected = dialect_range_low_res(x_range_low_res, y_range_low_res,
                                                                                 down_sample_ratio)
    x_range_high_res, y_range_high_res = get_range_high_res(x_range_low_res_dialected, y_range_low_res_dialected,
                                                            down_sample_ratio)
    # copy the range into a small patch
    image_patch_high_res = image_high_res[x_range_high_res[0]:x_range_high_res[1],
                           y_range_high_res[0]:y_range_high_res[1]].copy()
    # down sample the small patch
    image_patch_low_res = rescale_image(image_patch_high_res, down_sample_ratio)
    # copy the patch to the low-res image
    image_low_res[x_range_low_res_dialected[0]:x_range_low_res_dialected[1],
    y_range_low_res_dialected[0]:y_range_low_res_dialected[1]] = image_patch_low_res
    car_points = extract_nonzero_points(image_low_res)
    return car_points


def get_image_space_car(car, origin, resolution):
    X = []
    for point in car.car_bbox.points:
        x = (point.x - origin[0]) / resolution
        y = (point.y - origin[1]) / resolution
        X.append([x, y, 1])

    return np.asarray(X)

def get_image_space_car_state(car, origin, resolution):
    X = []

    car_dir = (car.car_speed * math.cos(car.car_yaw), car.car_speed * math.sin(car.car_yaw))

    x = (car.car_pos.x - origin[0]) / resolution
    y = (car.car_pos.y - origin[1]) / resolution
    X.append([x, y, 1])
    x = (car.car_pos.x + car_dir[0] - origin[0]) / resolution
    y = (car.car_pos.y + car_dir[1] - origin[1]) / resolution
    X.append([x, y, 1])
    return np.asarray(X)

def get_image_space_obstacle(obs, origin, resolution):
    X = []
    for point in obs.points:
        x = (point.x - origin[0]) / resolution
        y = (point.y - origin[1]) / resolution
        X.append([x, y, 1])

    return np.asarray(X)


def get_image_space_lane(lane_segment, origin, resolution):
    X = []
    x = (lane_segment.start.x - origin[0]) / resolution
    y = (lane_segment.start.y - origin[1]) / resolution
    X.append([x, y, 1])
    x = (lane_segment.end.x - origin[0]) / resolution
    y = (lane_segment.end.y - origin[1]) / resolution
    X.append([x, y, 1])

    return np.asarray(X)


def select_null_map_origin(hist_cars, ts=0):
    # origin is the upper-left corner of the map
    cur_car = hist_cars[ts]
    return [cur_car.car_pos.x - 20.0, cur_car.car_pos.y - 20.0]


def get_image_space_agent_history(agent_history, origins, resolution, dim):
    try:
        # ped_pos, ped_id, ped_goal_id, ped_speed, ped_vel, bb_x, bb_y, heading
        is_out_map = True
        transformed_history = []
        i = 0
        for hist_agent in agent_history:
            if hist_agent is not None:
                theta = hist_agent.heading
                x = hist_agent.ped_pos.x - origins[i][0]
                y = hist_agent.ped_pos.y - origins[i][1]
                # transformation matrix: rotate by theta and translate with (x,y)
                T = np.asarray([[cos(theta), -sin(theta), x], [sin(theta),
                                                               cos(theta), y], [0, 0, 1]])

                # bb_x is width of agent, bb_y is forward of agents.
                # car vertices in its local frame
                x1, y1, x2, y2, x3, y3, x4, y4 = hist_agent.bb_y, hist_agent.bb_x, -hist_agent.bb_y, hist_agent.bb_x, \
                                                 -hist_agent.bb_y, -hist_agent.bb_x, hist_agent.bb_y, -hist_agent.bb_x

                # homogeneous coordinates
                X = np.asarray([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]])

                # rotate and translate the car
                X = np.array([T.dot(x.T) for x in X], dtype=np.float32)
                # scale the car
                X = X / resolution
                transformed_history.append(X)

                frame_agent_is_out_map = check_out_map(X, dim)

                # agent should be included if it enters the map ar any of the history time steps
                if frame_agent_is_out_map is False:
                    is_out_map = False
                else:
                    pass  # print("frame out of map, dist to origin {}".format(dist))
            else:
                transformed_history.append(None)

            i += 1

        return transformed_history, is_out_map
    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def check_out_map(X, dim):
    for pixel_point in X:
        if np.any(pixel_point < 0) or np.any(pixel_point >= dim):
            return True
    return False


def get_range_high_res(x_range_dialected, y_range_dialected, down_sample_ratio):
    upsample_scale = 1.0 / down_sample_ratio

    x_range_high_res = [int(x_range_dialected[0] * upsample_scale), int(x_range_dialected[1] * upsample_scale)]
    y_range_high_res = [int(y_range_dialected[0] * upsample_scale), int(y_range_dialected[1] * upsample_scale)]
    return x_range_high_res, y_range_high_res


def dialect_range_low_res(x_range_low_res, y_range_low_res, down_sample_ratio):
    rescale_times = int(math.log(1 / down_sample_ratio, 2))
    x_range_dialected = [x_range_low_res[0] - rescale_times, x_range_low_res[1] + rescale_times]
    y_range_dialected = [y_range_low_res[0] - rescale_times, y_range_low_res[1] + rescale_times]

    x_range_dialected = np.maximum(x_range_dialected, 0)
    y_range_dialected = np.maximum(y_range_dialected, 0)
    x_range_dialected = np.minimum(x_range_dialected, config.imsize)
    y_range_dialected = np.minimum(y_range_dialected, config.imsize)

    return x_range_dialected, y_range_dialected


def get_car_range_low_res(points, down_sample_ratio):
    i_range = [1000000000, 0]
    j_range = [1000000000, 0]
    try:
        for point in points:
            x = point[0]
            y = point[1]
            x_low_res = int(x * down_sample_ratio)
            y_low_res = int(y * down_sample_ratio)
            if j_range[0] > x_low_res:
                j_range[0] = x_low_res
            if i_range[0] > y_low_res:
                i_range[0] = y_low_res
            if j_range[1] < x_low_res + 1:
                j_range[1] = x_low_res + 1
            if i_range[1] < y_low_res + 1:
                i_range[1] = y_low_res + 1

    except Exception as e:
        error_handler(e)  # do nothing
    return i_range, j_range


def fill_inside_car(points):
    points = np.array(np.round(points), dtype=np.int32)
    r = [p[0] for p in points]
    c = [p[1] for p in points]
    # fill the polygon
    rr, cc = polygon(r, c)
    new_points = zip(rr, cc)
    return list(new_points)


def fill_polygon_edges(points):
    new_points = []
    try:
        points = np.array(np.round(points), dtype=np.int32)

        for i, p in enumerate(points):
            r0 = p[0]
            c0 = p[1]
            if i + 1 < len(points):
                r1 = points[i + 1][0]
                c1 = points[i + 1][1]
            else:
                r1 = points[0][0]
                c1 = points[0][1]

            rr, cc = line(r0, c0, r1, c1)
            new_points = new_points + list(zip(rr, cc))

    except Exception as e:
        error_handler(e)

    # fill the polygon
    return list(new_points)


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


def parse_car_goal_from_txt(txt_file):
    with codecs.open(txt_file, 'r', encoding='utf-8',
                     errors='ignore') as fdata:
        for line in fdata:
            if 'car goal:' in line:
                line_split = line.split(' ')
                # print(line)
                return float(line_split[2]), float(line_split[3])
    return None


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
        if is_valid_file(txt_file):
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
