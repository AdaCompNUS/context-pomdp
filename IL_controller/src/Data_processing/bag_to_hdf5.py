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
import random
import pdb
import math
import glob
from skimage.draw import polygon, line

from transforms import *

from geometry_msgs.msg import Twist
from IL_controller.msg import peds_believes, peds_car_info

import time
import copy


# topic_mode = "seperate"
topic_mode = "combined"

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
    global topic_mode

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
    prev_ped_dict = OrderedDict()
    car_dict = OrderedDict()
    prev_car_dict = OrderedDict()
    act_reward_dict = OrderedDict()
    # ped_belief_dict = OrderedDict()
    pomdp_act_dict = OrderedDict()
    vel_dict = OrderedDict()

    i = 0
    for topic, msg, timestamp in bag.read_messages():
        if topic == '/map':
            map_dict[timestamp.to_nsec()] = msg
        elif topic == '/plan':
            plan_dict[timestamp.to_nsec()] = msg
            topic_mode = "seperate"
        elif topic == '/peds_car_info':
            ped_dict[timestamp.to_nsec()] = msg
            topic_mode = "seperate"
        elif topic == '/pomdp_action_plot':
            pomdp_act_dict[timestamp.to_nsec()] = msg
            topic_mode = "seperate"
        elif topic == '/cmd_vel_to_unity':
            vel_dict[timestamp.to_nsec()] = msg
        # elif topic == '/peds_believes':
        #     ped_belief_dict[timestamp.to_nsec()] = msg
        #     topic_mode = "seperate"
        elif topic == '/il_data':
            plan_dict[timestamp.to_nsec()] = msg.plan
            # ped_belief_dict[timestamp.to_nsec()] = msg.believes
            ped_dict[timestamp.to_nsec()] = msg.cur_peds.peds
            prev_ped_dict[timestamp.to_nsec()] = msg.past_peds.peds
            car_dict[timestamp.to_nsec()] = msg.cur_car
            prev_car_dict[timestamp.to_nsec()] = msg.past_car
            act_reward_dict[timestamp.to_nsec()] = msg.action_reward
            topic_mode = "combined"

    bag.close()

    # at least 1 message should be there
    if topic_mode == "combined" and (len(list(map_dict.keys())) < 1 or len(plan_dict.keys()) < 1 or len(
            ped_dict.keys()) < 1 or len(prev_ped_dict.keys()) < 1 or len(
        car_dict.keys()) < 1 or len(prev_car_dict.keys()) < 1 or len(list(act_reward_dict.keys())) < 1):
        print("invalid bag file: incomplete topics" + filename)
        is_valid = False
    else:
        is_valid = True

    if topic_mode == "combined":
        return map_dict, plan_dict, ped_dict, prev_ped_dict, car_dict, prev_car_dict, act_reward_dict, is_valid
    else:
        print("Exception: topic mode {} not defined".format(topic_mode))
        pdb.set_trace()


def combine_topics_in_one_dict(map_dict, plan_dict, ped_dict, car_dict, act_reward_dict):
    # print("Associating time for data with %d source time steps..." % len(act_reward_dict.keys()))

    combined_dict = OrderedDict()

    hist_ts = np.zeros(config.num_hist_channels, dtype=np.int32)

    hist_agents = []
    for i in range(0, config.num_hist_channels):
        hist_agents.append(peds_car_info())

    for timestamp in list(act_reward_dict.keys()):

        hist_agents, hist_complete = \
            track_history(hist_ts, hist_agents, car_dict, ped_dict, timestamp)

        # print("hist_ts: {}".format(hist_ts))

        if not hist_complete:
            continue

        # print("hist_agents: {}".format(hist_agents))

        vel_data = Twist()
        vel_data.linear.x = None  # current velocity of the car
        vel_data.linear.y = act_reward_dict[timestamp].linear.z  # target velocity of the car
        vel_data.angular.x = act_reward_dict[timestamp].angular.x  # steering

        action_data = Twist()
        action_data.linear.x = act_reward_dict[timestamp].linear.x  # acc
        action_data.linear.y = act_reward_dict[timestamp].linear.y  # reward

        combined_dict[timestamp] = {
            'agents': copy.deepcopy(hist_agents),
            'plan': plan_dict[timestamp],
            'vel': vel_data,
            'action': action_data,
        }

    was_near_goal = Was_near_goal(combined_dict)

    if was_near_goal:
        for timestamp in combined_dict.keys():
            hist_agents = combined_dict[timestamp]['agents']
            # agent_file.write("\n****** ts {} ******\n".format(timestamp))
            # for agent in hist_agents:
            #     agent_file.write("{},{} ".format(agent.car.car_pos.x, agent.car.car_pos.y))


    was_near_goal = trim_data_after_reach_goal(combined_dict)
    # print("bag: num_keys, near_goal", len(list(combined_dict.keys())), was_near_goal)
    return combined_dict, map_dict, was_near_goal


def Was_near_goal(combined_dict):
    dist_thresh = 2
    reach_goal = False

    for timestamp in list(combined_dict.keys()):
        goal_dist = 1000000
        cur_plan = combined_dict[timestamp]['plan']
        if len(cur_plan.poses) <= 3:
            goal_dist = 0

        if goal_dist < dist_thresh:
            reach_goal = True
            break

    return reach_goal


def trim_data_after_reach_goal(combined_dict):
    # get goal coordinates
    start_ts = min(list(combined_dict.keys()))
    start_plan = combined_dict[start_ts]['plan']
    #
    goal_coord = get_goal(start_plan)

    # check_path_step_res(start_plan)

    # get time at which car is within 2 meters of goal
    trim_time = None
    dist_thresh = 2
    if topic_mode == "combined":
        dist_thresh = 2

    for timestamp in list(combined_dict.keys()):
        goal_dist = 1000000
        cur_plan = combined_dict[timestamp]['plan']
        if len(cur_plan.poses) <= 3:
            goal_dist = 0

        if goal_dist < dist_thresh:
            trim_time = timestamp
            break
    #
    # delete all data after trim time
    all_keys = list(combined_dict.keys())
    if trim_time is None:  # never reach goal
        was_near_goal = False
    else:
        # remove data points after reaching goal
        for i in range(all_keys.index(trim_time) + 1, len(all_keys)):
            del combined_dict[all_keys[i]]
        was_near_goal = True
    return was_near_goal


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

            if config.num_hist_channels == i+1:
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


# now create data in the required format here
# don't forget to add random data for ped goal beliefs and history too coz pedestrians are not constant, also add car dimensions for history image
'''coordinate system should be consistent; bottom left is origin, x is 2nd dim, y is 1st dim, x increases up, y increases up, theta increases in anti clockwise'''


def create_h5_data(data_dict,
                   map_dict,
                   downsample_ratio=0.03125,
                   gamma=config.gamma):
    # !! downsample_ratio should be 1/2^n, e.g., 1/2^5=0.03125 !!
    print("Processing data with %d source time steps..." % len(data_dict.keys()))

    global topic_mode

    dim, map_intensity, map_intensity_scale, map_ts, new_dim, origin, raw_map_array, resolution = \
        parse_map_data(downsample_ratio, map_dict)

    # get data for other topics
    output_dict = OrderedDict()
    timestamps = list(data_dict.keys())

    # sample data points to h5
    # Change to sample a fixed number of points in the trajectory

    sample_idx = []
    sample_shuffled_idx = list(range(len(timestamps)))

    if learning_mode == 'rl':
        random.shuffle(sample_shuffled_idx)
        sample_shuffled_idx = iter(sample_shuffled_idx)
        sample_length = -1
    else:
        random.shuffle(sample_shuffled_idx)
        sample_shuffled_idx = iter(sample_shuffled_idx)
        sample_length = config.num_samples_per_traj

    while not (len(sample_idx) == sample_length):
        old_output_dict_length = len(output_dict.keys())
        idx = next(sample_shuffled_idx)

        ts = timestamps[idx]

        create_dict_entry(idx, output_dict)

        map_array, hist_pedmaps = \
            create_maps(map_dict, map_ts, raw_map_array)

        hist_cars, hist_peds = get_history_agents(data_dict, ts)

        peds_are_valid = process_peds(idx, output_dict, hist_cars, hist_peds, hist_pedmaps,
                                      dim, origin, resolution, downsample_ratio, map_intensity,
                                      map_intensity_scale)

        if not peds_are_valid:
            return dict(output_dict), False  # report invalid peds

        process_maps(dim, downsample_ratio, idx, map_array, output_dict, hist_pedmaps)

        process_car(idx, ts, output_dict, data_dict, hist_cars, dim, downsample_ratio, origin, resolution)

        process_actions(data_dict, idx, output_dict, ts)

        process_cart_agents(idx, output_dict, hist_peds, hist_cars)

        if len(output_dict.keys()) > old_output_dict_length:
            sample_idx.append(idx)

    process_values(data_dict, gamma, output_dict, sample_idx, timestamps)

    if len(output_dict.keys()) == 0:
        print("[Investigate] Bag results in no data!")
        pdb.set_trace()
    else:
        print("Creating data with %d time steps..." % len(output_dict.keys()))

    return dict(output_dict), True


def process_peds(data_idx, output_dict, hist_cars, hist_peds, hist_pedmaps,
                 dim, origin, resolution, downsample_ratio, map_intensity,
                 map_intensity_scale):

    try:
        nearest_ped_ids = get_nearest_ped_ids(hist_peds[0], hist_cars[0])  # return ped id
        valid_ped = 0

        for ped_no in range(len(nearest_ped_ids)):
            #
            ped_id = nearest_ped_ids[ped_no]

            history_is_complete, existing_hist_count = history_complete(ped_id, hist_peds)

            # if not history_is_complete:
            #     print("Warning: ped {} hist not complete. {} available".format(ped_id, existing_hist_count))

            hist_ped = get_ped_history(ped_id, hist_peds)

            # calculate position in map image
            hist_positions, is_out_map = get_map_indices(hist_ped, dim, origin, resolution)

            # discard peds outside the map
            if is_out_map:
                continue
                #
            elif valid_ped < config.num_peds_in_NN:
                # process nearby valid ped
                process_ped(data_idx, valid_ped, output_dict, hist_positions, dim, downsample_ratio)
                valid_ped += 1
                #
            else:
                # Treat the rest of far-away peds as static obstacles
                add_to_ped_map(hist_ped, hist_pedmaps, dim, map_intensity, map_intensity_scale,
                               origin, resolution)

        if valid_ped < config.num_peds_in_NN:
            # return dict(output_dict), False  # not going to use this bag
            print("!!! No enough valid pedestrians for input. Using dummy history at idx %d: Num valid peds %d."
                  % (data_idx, valid_ped))
            for i in range(valid_ped, config.num_peds_in_NN):
                process_blank_ped(data_idx, i, output_dict)

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return True


def process_ped(data_idx, ped_idx, output_dict, hist_positions,
                dim, downsample_ratio):
    # k-nearest neighbours
    # parse ped goal, and fill history entries

    if data_idx == None:
        process_ped_inner(ped_idx, output_dict, hist_positions,
                          dim, downsample_ratio)
    else:
        process_ped_inner(ped_idx, output_dict[data_idx], hist_positions,
                          dim, downsample_ratio)


def process_ped_inner(ped_idx, output_dict_entry, hist_positions,
                      dim, downsample_ratio):
    # k-nearest neighbours
    # parse ped goal, and fill history entries
    if config.num_hist_channels > 2:
        output_dict_entry['ped'][ped_idx] = construct_ped_data(dim, downsample_ratio, hist_positions)
    else:
        output_dict_entry['ped'][ped_idx] = construct_ped_data(dim, downsample_ratio, hist_positions)


def process_blank_ped(data_idx, ped_idx, output_dict):
    process_blank_ped_inner(output_dict[data_idx], ped_idx)


def process_blank_ped_inner(output_dict_entry, valid_ped):
    if config.num_hist_channels > 2:
        output_dict_entry['ped'][valid_ped] = construct_blank_ped_data()
    else:
        output_dict_entry['ped'][valid_ped] = construct_blank_ped_data()


def get_agents(data_dict, ts):
    peds = make_pedid_dict(data_dict[ts]['agent'].peds)
    car = data_dict[ts]['agent'].car
    return car, peds


def create_maps(map_dict, map_ts, raw_map_array):
    return create_maps_inner(map_dict[map_ts], raw_map_array)


def create_maps_inner(map_dict_entry, raw_map_array):
    map_array = np.array(raw_map_array, dtype=np.float32)
    hist_ped_maps = []
    for i in range(config.num_hist_channels):
        hist_ped_maps.append(np.zeros((map_dict_entry.info.height, map_dict_entry.info.width), dtype=np.float32))

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
    for idx in range(len(timestamps)):
        ts = timestamps[idx]
        if config.reward_mode == 'data':
            reward_arr[idx] = data_dict[ts]['action'].linear.y
        elif config.reward_mode == 'func':
            steer_data = data_dict[ts]['vel'].angular.x
            prev_ts = ts
            if idx >= 1:
                prev_ts = timestamps[idx - 1]
            try:
                prev_steer_data = data_dict[prev_ts]['vel'].angular.x
            except Exception as e:
                print(e)
                pdb.set_trace()

            acc_data = data_dict[ts]['action'].linear.x
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
    output_dict[idx]['vel_steer'] = np.array(
        [data_dict[ts]['vel'].linear.x, data_dict[ts]['vel'].angular.x, data_dict[ts]['vel'].linear.y],
        dtype=np.float32)
    assert (config.label_linear == 0 and config.label_angular == 1)
    output_dict[idx]['acc_id'] = np.array(
        [data_dict[ts]['action'].linear.x], dtype=np.float32)


def process_car(data_idx, ts, output_dict, data_dict, hist_cars, dim, downsample_ratio,
                origin, resolution):
    process_car_inner(output_dict[data_idx], data_dict[ts], hist_cars, dim, downsample_ratio,
                      origin, resolution)


def process_car_inner(output_dict_entry, data_dict_entry, hist_cars, dim, downsample_ratio,
                      origin, resolution):
    try:
        print_time = False
        path = data_dict_entry['plan']
        # parse path data
        start_time = time.time()
        path_data = construct_path_data(path, origin, resolution, dim,
                                        downsample_ratio)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Construct path time: " + str(elapsed_time) + " s")

        # parse car data
        start_time = time.time()

        output_dict_entry['car'] = construct_car_data(origin, resolution, dim, downsample_ratio,
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


def process_maps(dim, downsample_ratio, idx, map_array, output_dict, hist_pedmaps):
    process_maps_inner(dim, downsample_ratio, map_array, output_dict[idx], hist_pedmaps)


def process_maps_inner(dim, downsample_ratio, map_array, output_dict_entry, hist_pedmaps):
    map_array = rescale_image(map_array, dim, downsample_ratio)
    map_array = np.array(map_array, dtype=np.int32)
    # print("rescaled map max", map_array.max())
    map_array = normalize(map_array)
    # print("normalized map max", map_array.max())

    hist_peds_map = []
    for i in range(len(hist_pedmaps)):
        ped_array = rescale_image(hist_pedmaps[i], dim, downsample_ratio)
        # print("rescaled ped_array max", ped_array.max())
        ped_array = normalize(ped_array)
        # print("normalized ped_array max", ped_array.max())
        ped_array = np.maximum(map_array, ped_array)
        # print("merged ped_array max", np.maximum(map_array, ped_array).max())
        hist_peds_map.append(ped_array)

    # combine the static map and the pedestrian map
    output_dict_entry['maps'] = hist_peds_map


def process_cart_agents(idx, output_dict, hist_peds, hist_cars):
    process_cart_agents_inner(output_dict[idx], hist_peds, hist_cars)


def process_cart_agents_inner(output_dict_entry, hist_peds, hist_cars):
    try:
        no_info_value = -100.0
        hist_cart_positions = []

        nearest_ped_ids = get_nearest_ped_ids(hist_peds[0], hist_cars[0])  # return ped id

        for ts in range(config.num_hist_channels):
            car = hist_cars[ts]
            hist_cart_positions.append(car.car_pos.x)
            hist_cart_positions.append(car.car_pos.y)

            valid_ped = 0

            for ped_no in range(len(nearest_ped_ids)):
                if valid_ped < config.num_peds_in_map:
                    ped_id = nearest_ped_ids[ped_no]
                    hist_ped = get_ped_history(ped_id, hist_peds)

                    ped = hist_ped[ts]
                    if ped is not None:
                        hist_cart_positions.append(ped.ped_pos.x)
                        hist_cart_positions.append(ped.ped_pos.y)
                    else:
                        hist_cart_positions.append(no_info_value)
                        hist_cart_positions.append(no_info_value)

                    valid_ped += 1

            if valid_ped < config.num_peds_in_map:
                for i in range(valid_ped, config.num_peds_in_map):
                    hist_cart_positions.append(no_info_value)
                    hist_cart_positions.append(no_info_value)

    except Exception as e:
        print(e)

    output_dict_entry['cart_agents'] = hist_cart_positions


def normalize(ped_array):
    if np.max(ped_array) > 0:
        ped_array = ped_array / np.max(ped_array)
    return ped_array


def get_map_indices(hist_ped, dim, origin, resolution):
    hist_positions = []
    is_out_map = True  # it will be true if all hist pos are outside of the map
    for i in range(len(hist_ped)):
        hist_positions.append(np.array([], dtype=np.float32))
        try:
            ped = hist_ped[i]

            if ped is not None:
                hist_positions[i] = point_to_indices(ped.ped_pos.x, ped.ped_pos.y, origin, resolution, dim)
                is_out_map = is_out_map and np.any(hist_positions[i] == -1)
            else:
                hist_positions[i] = np.array([None, None], dtype=np.float32)
        except Exception as e:
            error_handler(e)
            pdb.set_trace()

    return hist_positions, is_out_map


def add_to_ped_map(hist_ped, hist_pedmaps, dim, map_intensity, map_intensity_scale, origin, resolution):
    for ts in range(len(hist_ped)):
        ped = hist_ped[ts]
        if ped is not None:
            ped_map_array = hist_pedmaps[ts]
            add_in_map(dim, map_intensity, map_intensity_scale, origin, ped, ped_map_array, resolution)
        else:
            pass
            # print("Ped hist {} empty".format(ts))


def add_in_map(dim, map_intensity, map_intensity_scale, origin, ped, ped_map_array, resolution):
    position = point_to_indices(ped.ped_pos.x, ped.ped_pos.y, origin, resolution, dim)
    if np.all(position != -1):
        ped_map_array[int(round(position[1]))][int(round(position[0]))] = map_intensity * map_intensity_scale
    # return position


def get_ped_history(ped_id, hist_peds):
    ped_hist = []

    for i in range(len(hist_peds)):
        try:
            if hist_peds[i] is not None and ped_id in hist_peds[i].keys():
                ped_hist.append(hist_peds[i][ped_id])
            else:
                ped_hist.append(None)
        except Exception as e:
            error_handler(e)
            print("!!! Investigate: not able to extract info on ped {} "
                  "in one of history time step. Ped hist {}".format(ped_id, hist_peds[i]))
            pdb.set_trace()

    return ped_hist


def history_complete(ped_id, hist_peds):
    has_hist = True
    hist_count = 0

    try:
        # print('------------------------------------------------')
        # print(hist_peds)
        # print('------------------------------------------------')

        for i in range(1, len(hist_peds)):
            if hist_peds[i] is not None:
                has_hist = has_hist and (ped_id in hist_peds[i].keys())
                hist_count += int(ped_id in hist_peds[i].keys())
            else:
                has_hist = False
    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return has_hist, hist_count


def get_history_agents(data_dict, ts):
    # agent_file.write("\n====== ts {} ======\n".format(ts))

    hist_cars, hist_peds = \
        get_combined_history(data_dict[ts], 'agents')

    agent_file.flush()

    return hist_cars, hist_peds


def get_combined_history(data_dict_entry, flag_hist):
    hist_cars = []
    hist_peds = []

    try:
        for i in range(config.num_hist_channels):

            # print("--------------------------------------------------------------------")
            # print(data_dict_entry[flag_hist][i])
            # print("--------------------------------------------------------------------")

            if data_dict_entry[flag_hist][i].peds is not None and data_dict_entry[flag_hist][i].car is not None:
                hist_peds.append(make_pedid_dict(data_dict_entry[flag_hist][i].peds))
                hist_cars.append(data_dict_entry[flag_hist][i].car)
            else:
                hist_peds.append(None)
                hist_cars.append(None)
            # agent_file.write("{},{} ".format(hist_cars[i].car_pos.x, hist_cars[i].car_pos.y))

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return hist_cars, hist_peds


def create_dict_entry(idx, output_dict):
    output_dict[idx] = {
        'maps': None,
        'ped': [dict({}) for x in range(config.num_peds_in_NN)],
        'car': {
            'goal': None,
            'hist': None,
        },
        'cart_agents': None,
        'vel_steer': None,
        'acc_id': None,
        'reward': None,
        'value': None
    }


def delete_dict_entry(idx, output_dict):
    del output_dict[idx]


def parse_map_data(downsample_ratio, map_dict):
    # get map data
    map_ts = list(map_dict.keys())[0]

    dim, map_intensity, map_intensity_scale, new_dim, origin, raw_map_array, resolution = \
        parse_map_data_from_dict(downsample_ratio, map_dict[map_ts])

    return dim, map_intensity, map_intensity_scale, map_ts, new_dim, origin, raw_map_array, resolution


def parse_map_data_from_dict(downsample_ratio, map_dict_entry):
    # get map data
    raw_map_array = np.asarray(map_dict_entry.data).reshape(
        (map_dict_entry.info.height, map_dict_entry.info.width))
    resolution = map_dict_entry.info.resolution
    origin = map_dict_entry.info.origin.position.x, map_dict_entry.info.origin.position.y
    dim = int(map_dict_entry.info.height)  # square assumption
    # map_intensity = np.max(raw_map_array)

    map_intensity = np.max(raw_map_array)
    map_intensity_scale = 1500.0

    new_dim = int(dim * downsample_ratio)
    return dim, map_intensity, map_intensity_scale, new_dim, origin, raw_map_array, resolution


def rescale_image(image,
                  dim=1024,
                  downsample_ratio=0.03125):
    image1 = image.copy()

    try:
        for i in range(int(math.log(1 / downsample_ratio, 2))):
            image1 = cv2.pyrDown(image1)

    except Exception as e:
        error_handler(e)

    return image1


def get_pyramid_image_points(points,
                             dim=1024,
                             downsample_ratio=0.03125,
                             intensities=False):
    format_point = None
    try:
        # !! input points are in Euclidean space (x,y), output points are in image space (row, column) !!
        arr = np.zeros((dim, dim), dtype=np.float32)
        fill_image_with_points(arr, dim, points, intensities)
        # down sample the image
        arr1 = rescale_image(arr, dim, downsample_ratio)

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


def get_nearest_ped_ids(peds, car):
    dist_list = {}
    for idx, ped in peds.items():
        dist_list[idx] = dist(ped, car)
    dist_list = OrderedDict(sorted(dist_list.items(), key=lambda a: a[1]))
    return list(dist_list.keys())


def make_pedid_dict(ped_list):
    ped_dict = {}
    for ped in ped_list:
        ped_dict[ped.ped_id] = ped
    return ped_dict


def dist(ped, car):
    return np.sqrt((ped.ped_pos.x - car.car_pos.x) ** 2 +
                   (ped.ped_pos.y - car.car_pos.y) ** 2)


def construct_ped_data(dim, downsample_ratio, hist_positions):
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
                                                                        downsample_ratio=downsample_ratio))
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


def construct_path_data(path, origin, resolution, dim, downsample_ratio):
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
        path_output_dict['path'], dim=dim, downsample_ratio=downsample_ratio)
    return path_output_dict


def construct_car_data(origin, resolution, dim, downsample_ratio, hist_cars, path_data=None):
    car_output_dict = {'goal': list([]), 'hist': []}
    try:
        # 'goal' contains the rescaled path in pixel points
        car_output_dict['goal'] = path_data['path']

        # print('--------------------------------------')
        # print("==> path data with {} points".format(len(path_data['path'])))
        # print('--------------------------------------')

        print("construct_car_data: num hist cars {}".format(len(hist_cars)))

        for i in range(len(hist_cars)):

            # print('--------------------------------------')
            # print("===> construct_car_data hist {} car {}".format(i, hist_cars[i]))
            # print('--------------------------------------')

            car_points = get_car_points(hist_cars[i], origin, dim, downsample_ratio, resolution)

            # if car_points is not None:
            #     print("==> car data with {} points".format(len(car_points)))

            # if hist_cars[i]:
            #     # if len(car_points) == 0:
            #     print("===> construct_car_data hist {} car points {}".format(i, car_points))

            car_output_dict['hist'].append(car_points)

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

    return car_output_dict


def get_car_points(car, origin, dim_high_res, downsample_ratio, resolution):
    # create transformation matrix of point, multiply with 4 points in local coordinates to get global coordinates
    car_points = None

    if car is not None:
        print_time = False
        start_time = time.time()
        X = get_transformed_car(car, origin, resolution)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Car prepare time: " + str(elapsed_time) + " s")

        # construct the image
        start_time = time.time()
        # car_edge_points = list(fill_inside_car(X))
        car_edge_points = fill_car_edges(X)
        if print_time:
            elapsed_time = time.time() - start_time
            print("Fill car time: " + str(elapsed_time) + " s")

        try:
            start_time = time.time()
            # car_points, start_time = get_image_points_local_pyramid(car_edge_points, dim_high_res, downsample_ratio,
            #                                                         print_time, start_time)

            car_points = get_pyramid_image_points(car_edge_points, dim_high_res, downsample_ratio)

            if print_time:
                elapsed_time = time.time() - start_time
                print("Car pyramid time: " + str(elapsed_time) + " s")
        except Exception as e:
            error_handler(e)

        #####################################################################
        #  points = get_pyramid_image_points(points, dim, downsample_ratio)
        #####################################################################

    return car_points


def get_image_points_local_pyramid(car_edge_points, dim_high_res, downsample_ratio, print_time, start_time):
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
                                                 downsample_ratio)
    return car_points, start_time


def get_car_points_in_low_res_image(car_edge_points, image_high_res, image_low_res, dim, downsample_ratio):
    # find the corresponding range in the low-res image
    x_range_low_res, y_range_low_res = get_car_range_low_res(car_edge_points, downsample_ratio)
    # find the high-res image range
    x_range_low_res_dialected, y_range_low_res_dialected = dialect_range_low_res(x_range_low_res, y_range_low_res,
                                                                                 downsample_ratio)
    x_range_high_res, y_range_high_res = get_range_high_res(x_range_low_res_dialected, y_range_low_res_dialected,
                                                            downsample_ratio)
    # copy the range into a small patch
    image_patch_high_res = image_high_res[x_range_high_res[0]:x_range_high_res[1],
                           y_range_high_res[0]:y_range_high_res[1]].copy()
    # down sample the small patch
    image_patch_low_res = rescale_image(image_patch_high_res, dim, downsample_ratio)
    # copy the patch to the low-res image
    image_low_res[x_range_low_res_dialected[0]:x_range_low_res_dialected[1],
    y_range_low_res_dialected[0]:y_range_low_res_dialected[1]] = image_patch_low_res
    car_points = extract_nonzero_points(image_low_res)
    return car_points


def get_transformed_car(car, origin, resolution):
    theta = np.radians(car.car_yaw)
    x = car.car_pos.x - origin[0]
    y = car.car_pos.y - origin[1]
    # transformation matrix: rotate by theta and translate with (x,y)
    T = np.asarray([[cos(theta), -sin(theta), x], [sin(theta),
                                                   cos(theta), y], [0, 0, 1]])
    # car vertices in its local frame
    #  (-0.8, 0.95)---(3.6, 0.95)
    #  |                       |
    #  |                       |
    #  |                       |
    #  (-0.8, -0.95)--(3.6, 0.95)
    x1, y1, x2, y2, x3, y3, x4, y4 = 3.6, 0.95, -0.8, 0.95, -0.8, -0.95, 3.6, -0.95
    # homogeneous coordinates
    X = np.asarray([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]])
    # rotate and translate the car
    X = np.array([T.dot(x.T) for x in X], dtype=np.float32)
    # scale the car
    X = X / resolution
    return X


def get_range_high_res(x_range_dialected, y_range_dialected, downsample_ratio):
    upsample_scale = 1.0 / downsample_ratio
    x_range_high_res = [int(x_range_dialected[0] * upsample_scale), int(x_range_dialected[1] * upsample_scale)]
    y_range_high_res = [int(y_range_dialected[0] * upsample_scale), int(y_range_dialected[1] * upsample_scale)]
    return x_range_high_res, y_range_high_res


def dialect_range_low_res(x_range_low_res, y_range_low_res, downsample_ratio):
    rescale_times = int(math.log(1 / downsample_ratio, 2))
    x_range_dialected = [x_range_low_res[0] - rescale_times, x_range_low_res[1] + rescale_times]
    y_range_dialected = [y_range_low_res[0] - rescale_times, y_range_low_res[1] + rescale_times]

    x_range_dialected = np.maximum(x_range_dialected, 0)
    y_range_dialected = np.maximum(y_range_dialected, 0)
    x_range_dialected = np.minimum(x_range_dialected, config.imsize)
    y_range_dialected = np.minimum(y_range_dialected, config.imsize)

    return x_range_dialected, y_range_dialected


def get_car_range_low_res(points, downsample_ratio):
    i_range = [1000000000, 0]
    j_range = [1000000000, 0]
    try:
        for point in points:
            x = point[0]
            y = point[1]
            x_low_res = int(x * downsample_ratio)
            y_low_res = int(y * downsample_ratio)
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


def fill_car_edges(points):
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

def main(bagspath, peds_goal_path, nped, start_file, end_file, thread_id):

    global agent_file
    # Walk into subdirectories recursively and collect all txt files

    agent_file = open("agent_record.txt", 'w')

    # print("Initialize agent file {}".format(agent_file))

    config.num_peds_in_NN = nped
    topic_mode = "combined"

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
                ped_goal_list = parse_goals_using_h5_name(h5_name, peds_goal_path)
                #
                if topic_mode == "combined":
                    map_dict, plan_dict, ped_dict, prev_ped_dict, car_dict, prev_car_dict, \
                    act_reward_dict, topics_complete = parse_bag_file(filename=bag_file)

                    if topics_complete:  # all topics in the bag are non-empty

                        per_step_data_dict, map_dict, was_near_goal = combine_topics_in_one_dict(
                            map_dict, plan_dict, ped_dict, car_dict, act_reward_dict)
                        end_time = time.time()
                        print("Thread %d parsed: elapsed time %f s" % (thread_id, end_time - start_time))
                        if was_near_goal:
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
                                    "pedestrian data not used. Nearby pedestrians disappear during the drive "
                                    "(investigate)")
                                investigate_files.append(txt_file)
                                continue
                        else:  # was_near_goal == False
                            print("car did not reach near goal, not writing")
                            investigate_files.append(txt_file)
                            continue
                    else:  # topics_complete == False
                        print("doing nothing coz invalid bag file")
                        incomplete_file_counter += 1
                        continue
                else:
                    print("Exception: topic mode {} not defined".format(topic_mode))
                    pdb.set_trace()

    # print("number of filtered files %d" % len(filtered_files))
    print("no of files with imcomplete topics %d" % incomplete_file_counter)

    agent_file.close()


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
    with open(txtfile, 'r') as f:
        for line in f:

            # if config.fit_val:
            #     reach_goal_flag = True  # doesn't matter if the car reaches the goal for value fitting
            # else: #  if fitting action
            if 'goal reached' in line:
                reach_goal_flag = True
                break

            if (('INININ' in line) or ("collision = 1" in line)) and reach_goal_flag == False:
                collision_flag = True
                print("colliison in " + txtfile)
                break

    if reach_goal_flag == True and collision_flag == False:
        return True
    elif reach_goal_flag == False:
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
        default='BTSRL_driving_data/Maps',
        help='Path to pedestrian goals')

    parser.add_argument(
        '--nped',
        type=int,
        default=config.num_peds_in_NN,
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

    config.num_peds_in_NN = parser.parse_args().nped

    main(bagspath, peds_goal_path, parser.parse_args().nped, start_file, end_file, 0)



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
