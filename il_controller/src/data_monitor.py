import sys

sys.path.append('./Data_processing/')
# sys.path.append('./drive_topics/')

from Data_processing import global_params
config = global_params.config

import os

if config.pycharm_mode:
    import pyros_setup
    pyros_setup.configurable_import().configure('mysetup.cfg').activate()

import os, shutil

import time
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
from torch.distributions import Categorical
import ipdb as pdb
from dataset import *
from model_drive_net import *
from tensorboardX import SummaryWriter
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from Data_processing import bag_to_hdf5
from tqdm import tqdm
import random
import scipy.ndimage
import datetime

from transforms import *
from visualization import *

import threading

# plt.ioff()

valid_threshold = 1000

import rospy


from nav_msgs.msg import Path, Odometry
from nav_msgs.msg import OccupancyGrid
from il_controller.msg import peds_believes, peds_car_info, imitation_data
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()
ped_goal_list = None


class DataMonitor(data.Dataset):

    def __init__(self):
        self.lock = threading.Lock()

        self.valid_count = 0
        self.has_map = False
        self.has_hist = False
        self.has_plan = False
        self.has_beliefs = False
        self.has_data = False
        self.origin = None
        self.resolution = None
        self.dim = None
        self.new_dim = None
        self.map_intensity = None
        self.map_intensity_scale = None
        self.is_triggered = False
        self.data_is_alive = False
        # wait for data for 10 check_alive calls, if no data, exit program
        self.data_patience_clock = 0
        self.data_patience = 10
        self.num_agents = 1 + 0
        imsize = config.imsize
        # container to be updated directly by subscribers
        self.combined_dict = {
            'map': {},
            'hist': [None for x in range(config.num_hist_channels)],
            'plan': {}
        }
        # intermediate container for images
        self.output_dict = {
            'maps': None,
            'ped': [dict({}) for x in range(0)],
            'car': {
                'goal': None,
                'hist': None
            },
        }

        self.hist_ts = None
        self.belief_ts = None
        self.path_ts = None
        # final array for input images
        self.cur_data = {
            'true_steer': None,
            'true_acc': None,
            'true_vel': None,
            'nn_input': np.zeros(
                (1, self.num_agents, config.total_num_channels, imsize, imsize), dtype=np.float32)
        }

        # register callback functions
        rospy.Subscriber('/map', OccupancyGrid, self.receive_map_callback)

        rospy.Subscriber('/il_data', imitation_data, self.receive_il_data_callback, queue_size=1)
        rospy.Subscriber('/IL_steer_cmd', Float32, self.receive_il_steer_callback, queue_size=1)
        self.steering_topic_alive = True
        self.steering_is_triggered = False

        self.raw_map_array = None
        self.ped_map_array = []
        for i in range(0, config.num_hist_channels):
            self.ped_map_array.append(None)

        # self.agent_data = peds_car_info()
        # self.past_agent_data = peds_car_info()
        # self.past_agent_data2 = peds_car_info()
        # self.past_agent_data3 = peds_car_info()

        self.hist_agents = []
        for i in range(0, config.num_hist_channels):
            self.hist_agents.append(peds_car_info())

        self.true_steering = None
        self.true_acc = None
        self.true_vel = None

        self.update_steering = True

        self.update_data = True

        self.data_idx = 0

    # parse data from ros topics
    def receive_map_callback(self, data):
        self.lock.acquire()
        print("Receive map")
        if self.has_map:
            self.valid_count -= 1  # disable inference when updating data
        try:
            self.combined_dict['map'] = data

            status = self.convert_to_nn_input("map")
            if status:
                self.valid_count += 1
                if not self.has_map:
                    self.has_map = True
            else:
                self.has_map = False
        except Exception as e:
            print("Exception", e)
            self.has_map = False
        finally:
            if self.has_map:
                pass  # print("Map updated")
            else:
                print("Map skipped")
            if self.lock.locked():
                self.lock.release()  # release self.lock, no matter what

    def receive_il_data_callback(self, data):
        if not self.update_data:
            self.data_is_alive = True
            self.data_patience_clock = 0
            return
        self.lock.acquire()
        start_time = time.time()
        self.update_data = False
        print("Receive data")
        if self.has_data:
            self.valid_count -= 1  # disable inference when updating data
        try:

            self.parse_data_from_msg(data)

            if self.convert_to_nn_input("data"):
                self.valid_count += 1
                if not self.has_data:
                    self.has_data = True
            else:
                self.has_data = False
        except Exception as e:
            print("Exception", e)
            self.has_data = False
        finally:
            if self.has_data:
                elapsed_time = time.time() - start_time
                print("Data processing time: " + str(elapsed_time) + " s")
                # pass #print("Hist updated")
            else:
                print("Data package skipped")
            if self.lock.locked():
                self.lock.release()  # release self.lock, no matter what
            self.update_data = True

    def receive_il_steer_callback(self, data):
        if self.update_steering:
            # print("steering data call_back~~~~~~~~~~~~~~~~~~~~:", np.degrees(float(data.data)))
            self.true_steering = float(data.data)
            self.cur_data['true_steer'] = self.true_steering
            self.steering_topic_alive = True
            self.steering_is_triggered = True
        else:
            print("steering data call_back~~~~~~~~~~~ data skipped ~~~~~~~~~:")

    def parse_data_from_msg(self, data):
        start_time = time.time()

        self.parse_history(data)
        self.parse_plan(data)
        # self.parse_belief(data)
        self.parse_actions(data)

        elapsed_time = time.time() - start_time
        print("Topic parsing time: " + str(elapsed_time) + " s")

    def parse_actions(self, data):
        print('Parsing angle from il_data:', data.action_reward.angular.x)
        # self.true_steering = data.action_reward.angular.x
        self.true_acc = data.action_reward.linear.x
        self.true_vel = data.action_reward.linear.z

        # print("*******************************`Get action reward from data:", data.action_reward)

    def parse_belief(self, data):
        self.combined_dict['beliefs'] = data.believes

    def parse_plan(self, data):
        self.combined_dict['plan'] = data.plan

    def parse_history(self, data):
        try:
            for i in reversed(range(1, config.num_hist_channels)):
                self.hist_agents[i] = peds_car_info()
                if self.combined_dict['hist'][i-1] is not None:
                    self.hist_agents[i].car = self.hist_agents[i - 1].car
                    self.hist_agents[i].peds = self.hist_agents[i - 1].peds
                else:
                    self.hist_agents[i].car = None
                    self.hist_agents[i].peds = None

                self.combined_dict['hist'][i] = copy.deepcopy(self.hist_agents[i])

            self.hist_agents[0] = peds_car_info()
            self.hist_agents[0].car = data.cur_car
            self.hist_agents[0].peds = data.cur_peds.peds
            self.combined_dict['hist'][0] = copy.deepcopy(self.hist_agents[0])

            # for i in reversed(range(0, config.num_hist_channels)):
            #     if self.hist_agents[i]:
            #         print('=> hist car {}: {}'.format(i, self.hist_agents[i].car))

        except Exception as e:
            error_handler(e)
            pdb.set_trace()

    def test_terminal(self):
        terminal = False

        try:
            # get goal coordinates
            plan = self.combined_dict['plan']

            if config.car_goal[0] == -1 and config.car_goal[1] == -1: # no goal input from cmd_args
                goal_coord = bag_to_hdf5.get_goal(plan)
            else:
                goal_coord = config.car_goal

            # send terminal signal when car is within 1.2 meters of goal
            if bag_to_hdf5.euclid_dist(goal_coord, (
                    self.combined_dict['hist'][0].car.car_pos.x, self.combined_dict['hist'][0].car.car_pos.y)) < 1.3:
                terminal = True
                print("================= Goal reached ===================")
                return terminal
            else:
                print("================= Car: %f %f, goal: %f %f ===================" %
                      (self.combined_dict['hist'][0].car.car_pos.x, self.combined_dict['hist'][0].car.car_pos.y,
                       goal_coord[0], goal_coord[1]))
        except Exception as e:
            error_handler(e)
            pdb.set_trace()

        return terminal

    def data_valid(self):
        global valid_threshold
        if self.valid_count < valid_threshold:  # valid only when all 4 types of info exists: map, believes, hist, and plan
            print("Data not ready yet: valid_count: " + str(self.valid_count))
            return False
        else:
            return True

    def convert_to_nn_input(self, flag=None, downsample_ratio=0.03125):
        print("Converting " + flag + " info...")
        try:
            if flag == "map":
                self.dim, self.map_intensity, self.map_intensity_scale, self.new_dim, self.origin, self.raw_map_array, \
                self.resolution = bag_to_hdf5.parse_map_data_from_dict(downsample_ratio, self.combined_dict['map'])

            if flag == "hist" or flag == "beliefs" or flag == "plan" or flag == "data":

                if flag == "hist":
                    if not self.has_beliefs or not self.has_plan:
                        return False
                if flag == "plan":
                    if not self.has_beliefs:
                        return False
                if flag == "beliefs":
                    pass

                # reset patience clock
                self.data_is_alive = True
                self.is_triggered = True
                self.data_patience_clock = 0

                # start_time = time.time()
                map_array, self.ped_map_array = \
                    bag_to_hdf5.create_maps_inner(self.combined_dict['map'], self.raw_map_array)

                # elapsed_time = time.time() - start_time
                # print("Map allocation time: " + str(elapsed_time) + " s")

                hist_cars, hist_peds = \
                    bag_to_hdf5.get_bounded_history(self.combined_dict, 'hist')

                # for i in reversed(range(0, config.num_hist_channels)):
                #     if hist_cars[i]:
                #         print('==> combined_dict hist car {}: {}'.format(i, hist_cars[i]))

                self.data_idx += 1

                self.check_history_completeness(hist_cars)

                # start_time = time.time()
                peds_are_valid = bag_to_hdf5.process_exo_agents(hist_cars=hist_cars, hist_exo_agents=hist_peds,
                                                                hist_env_maps=self.ped_map_array, dim=self.dim,
                                                                resolution=self.resolution,
                                                                map_intensity=self.map_intensity,
                                                                map_intensity_scale=self.map_intensity_scale,
                                                                origin=self.origin)

                # elapsed_time = time.time() - start_time
                # print("Peds processing time: " + str(elapsed_time) + " s")

                if not peds_are_valid:
                    return False  # report invalid peds

                # start_time = time.time()
                bag_to_hdf5.process_maps_inner(downsample_ratio, map_array, self.output_dict,
                                               self.ped_map_array)
                # elapsed_time = time.time() - start_time
                # print("Map processing time: " + str(elapsed_time) + " s")

                # start_time = time.time()
                bag_to_hdf5.process_car_inner(self.output_dict, self.combined_dict, hist_cars,
                                              self.dim, downsample_ratio, self.origin, self.resolution)
                # elapsed_time = time.time() - start_time
                # print("Car processing time: " + str(elapsed_time) + " s")

            # put all info into images
            # start_time = time.time()
            self.data_to_images(flag)
            self.record_labels()
            # elapsed_time = time.time() - start_time
            # print("Image conversion time: " + str(elapsed_time) + " s")

            print(flag + " update done.")
            return True
        except Exception as e:
            error_handler(e)
            return False

    def check_history_completeness(self, hist_cars):
        history_is_complete = True
        for i in reversed(range(0, len(hist_cars))):
            if hist_cars[i] is None:
                history_is_complete = False

        return history_is_complete

    def record_labels(self):
        self.cur_data['true_steer'] = self.true_steering
        self.cur_data['true_acc'] = self.true_acc
        self.cur_data['true_vel'] = self.true_vel

    def data_to_images(self, flag):
        try:
            for i in range(self.num_agents):

                if flag == "map" or flag == "data":

                    if self.output_dict['maps'] is not None:
                        for c in range(0, config.num_hist_channels):
                            self.cur_data['nn_input'][0, i, config.channel_map[c]] = self.output_dict['maps'][c]
                    else:
                        for c in range(0, config.num_hist_channels):
                            self.cur_data['nn_input'][0, i, config.channel_map[c], ...] = 0

                if i >= 0:  # pedestrians
                    agent_flag = 'car'

                    if self.output_dict[agent_flag]['goal'] is None: ##########################
                        return

                    # if self.output_dict[agent_flag] is not None:
                    if flag == "hist" or flag == "data":
                        self.cur_data['nn_input'][0, i, config.channel_goal, ...] = 0
                        # print("==> goal channel to fill: {}, {} points".format(config.channel_goal,
                        #                                                        len(self.output_dict[agent_flag]['goal'])))
                        for point in self.output_dict[agent_flag]['goal']:
                            self.cur_data['nn_input'][0, i, config.channel_goal, int(
                                point[0]), int(point[1])] = point[2]

                        for c in range(0, config.num_hist_channels):

                            self.cur_data['nn_input'][0, i, config.channel_hist[c], ...] = 0

                            if self.output_dict[agent_flag]['hist'][c] is not None:
                                # if len(self.output_dict[agent_flag]['hist'][c]) == 0:
                                #     print("====> fill image: hist {} car points empty ".format(c))

                                for point in self.output_dict[agent_flag]['hist'][c]:
                                    self.cur_data['nn_input'][0, i, config.channel_hist[c], int(
                                        point[0]), int(point[1])] = point[2]
                            else:
                                pass
                                # print("====> fill image: hist {} car points None ".format(c))

        except Exception as e:
            error_handler(e)
            return


    def check_alive(self):
        if self.is_triggered:

            self.data_patience_clock += 1
            print("============================ data_lock %d, is_alive %d ======================" %
                  (self.data_patience_clock, self.data_is_alive))

            # data supply is missing
            if self.data_patience_clock >= self.data_patience:
                self.data_is_alive = False

            # if not self.steering_topic_alive:
            #     print("steering topic broken!!!!!!!!!!!!!!!!!!!!")
            #     return False

            return self.data_is_alive
        else:
            # still waiting for first data
            print('Waiting for triggering')
            return True


def load_goals(ped_goal_file):
    global ped_goal_list
    if ped_goal_file is None:
        print("no goal file found")
        assert False
        pass
    with open(ped_goal_file, 'r') as f:
        ped_goal_list = f.read().split('\n')
        ped_goal_list = [
            list(map(float,
                     x.strip().split(' '))) for x in ped_goal_list
        ]

    print("load_goals: ped_goal_list", ped_goal_list)
