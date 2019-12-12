import sys

sys.path.append('./Data_processing/')

from Data_processing import global_params
from Data_processing.global_params import print_long

config = global_params.config

import os


if config.pycharm_mode:
    import pyros_setup

    pyros_setup.configurable_import().configure('mysetup.cfg').activate()

from policy_value_network import *
from tensorboardX import SummaryWriter
import matplotlib

matplotlib.use('Agg')
from Data_processing import bag_to_hdf5
from visualization import *

hist_num_threshold = config.num_hist_channels

import rospy
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from msg_builder.msg import peds_car_info, imitation_data, Lanes
from std_msgs.msg import Float32
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


class DataMonitor(data.Dataset):

    def __init__(self):

        self.cur_hist_count = 0

        self.map_ready = False
        self.local_lanes_ready = False
        self.il_data_ready = False
        self.steering_ready = False

        self.coord_frame = None

        self.dim, self.map_intensity, self.map_intensity_scale, self.new_dim, self.resolution = \
            bag_to_hdf5.create_null_map_data()
        self.raw_map_array, self.hist_env_maps, self.hist_car_maps, self.lane_map, self.obs_map, self.goal_map \
            = bag_to_hdf5.create_null_maps()

        self.data_is_alive = False
        # wait for data for 10 check_alive calls, if no data, exit program
        self.data_patience_clock = 0
        self.data_patience = 50
        imsize = config.imsize
        # container to be updated directly by subscribers
        self.combined_dict = {
            'map': {},
            'agents_hist': [None for x in range(config.num_hist_channels)],
            'plan': {}
        }
        # intermediate container for images
        self.output_dict = {
            'maps': None,
            'ped': [dict({}) for x in range(0)],
            'lane': None,
            'car': {
                'goal': None,
                'hist': None,
                'semantic': None
            },
        }

        # final array for input images
        self.cur_data = {
            'true_steer_normalized': None,
            'true_acc': None,
            'true_vel': None,
            'true_lane': None,
            'nn_input': np.zeros(
                (1, 1, config.total_num_channels, imsize, imsize), dtype=global_config.data_type),
            'nn_semantic_input': None
        }

        # register callback functions
        rospy.Subscriber('/map', OccupancyGrid, self.receive_map_callback)
        rospy.Subscriber('/il_data', imitation_data, self.receive_il_data_callback, queue_size=1)
        rospy.Subscriber('/local_lanes', Lanes, self.receive_lane_callback, queue_size=1)
        rospy.Subscriber('/purepursuit_cmd_steer', Float32, self.receive_cmd_steer_callback, queue_size=1)

        self.hist_agents = []
        for i in range(0, config.num_hist_channels):
            self.hist_agents.append(peds_car_info())

        self.true_steering_norm = None
        self.true_acc = None
        self.true_vel = None
        self.true_lane = None

    # parse data from ros topics
    def receive_map_callback(self, data):
        self.map_ready = False
        return

        print_long("Receive map")
        try:
            self.combined_dict['map'] = data
            self.map_ready = self.convert_to_nn_input("map")
        except Exception as e:
            print("Exception", e)
            self.map_ready = False
        finally:
            if self.map_ready:
                pass
            else:
                print_long("Map skipped")

    def receive_il_data_callback(self, data):
        self.data_is_alive = True
        self.data_patience_clock = 0

        start_time = time.time()
        print_long("Receive data")
        try:
            self.parse_il_data_from_msg(data)
            self.il_data_ready, self.cur_hist_count = self.convert_to_nn_input("data")
        except Exception as e:
            print("Exception", e)
            self.il_data_ready = False
        finally:
            if self.il_data_ready:
                elapsed_time = time.time() - start_time
                print_long("Data processing time: " + str(elapsed_time) + " s")
            else:
                print_long("Data package skipped")

    def receive_lane_callback(self, data):
        print_long("Receive lane")
        try:
            self.combined_dict['lanes'] = data.lane_segments
            self.local_lanes_ready, _ = self.convert_to_nn_input("lanes")
        except Exception as e:
            print("Exception", e)
            self.local_lanes_ready = False
        finally:
            if self.local_lanes_ready:
                pass
            else:
                print_long("Lane skipped")

    def receive_cmd_steer_callback(self, data):
        # print_long("steering data call_back~~~~~~~~~~~~~~~~~~~~:", np.degrees(float(data.data)))
        self.true_steering_norm = float(data.data)
        self.cur_data['true_steer_normalized'] = self.true_steering_norm
        self.steering_ready = True

    def parse_il_data_from_msg(self, data):
        start_time = time.time()

        self.parse_history(data)
        self.parse_plan(data)
        self.parse_actions(data)

        elapsed_time = time.time() - start_time
        print_long("Topic parsing time: " + str(elapsed_time) + " s")

    def parse_actions(self, data):
        print('Parsing angle from il_data:', data.action_reward.steering_normalized)
        # self.true_steering = data.action_reward.steering_normalized
        self.true_acc = int(data.action_reward.acceleration_id.data)
        self.true_vel = float(data.action_reward.target_speed.data)
        self.true_lane = int(data.action_reward.lane_change.data)
        # print_long("*******************************`Get action reward from data:", data.action_reward)

    def parse_plan(self, data):
        self.combined_dict['plan'] = data.plan

    def parse_lanes(self, data):
        self.combined_dict['lanes'] = data.lane_segments

    def parse_history(self, data):
        try:
            for i in reversed(range(1, config.num_hist_channels)):
                self.hist_agents[i] = peds_car_info()
                if self.combined_dict['agents_hist'][i - 1] is not None:
                    self.hist_agents[i].car = self.hist_agents[i - 1].car
                    self.hist_agents[i].peds = self.hist_agents[i - 1].peds
                else:
                    self.hist_agents[i].car = None
                    self.hist_agents[i].peds = None

                self.combined_dict['agents_hist'][i] = copy.deepcopy(self.hist_agents[i])

            self.hist_agents[0] = peds_car_info()
            self.hist_agents[0].car = data.cur_car
            self.hist_agents[0].peds = data.cur_peds.peds
            self.combined_dict['agents_hist'][0] = copy.deepcopy(self.hist_agents[0])

        except Exception as e:
            error_handler(e)
            pdb.set_trace()

    def test_terminal(self):
        terminal = False

        try:
            # get goal coordinates
            plan = self.combined_dict['plan']

            if config.car_goal[0] == -1 and config.car_goal[1] == -1:  # no goal input from cmd_args
                goal_coord = bag_to_hdf5.get_goal(plan)
            else:
                goal_coord = config.car_goal

            # send terminal signal when car is within 1.2 meters of goal
            if bag_to_hdf5.euclid_dist(goal_coord, (
                    self.combined_dict['agents_hist'][0].car.car_pos.x,
                    self.combined_dict['agents_hist'][0].car.car_pos.y)) < 1.3:
                terminal = True
                print_long("================= Goal reached ===================")
                return terminal
            else:
                print_long("================= Car: %f %f, goal: %f %f ===================" %
                           (self.combined_dict['agents_hist'][0].car.car_pos.x,
                            self.combined_dict['agents_hist'][0].car.car_pos.y,
                            goal_coord[0], goal_coord[1]))
        except Exception as e:
            error_handler(e)
            pdb.set_trace()

        return terminal

    def data_valid(self):
        global hist_num_threshold
        if self.cur_hist_count < hist_num_threshold:  # valid only when all 4 types of info exists: map, believes, hist, and plan
            print_long("Data not ready yet: cur_hist_count: " + str(self.cur_hist_count))
            return False
        else:
            return True

    def convert_to_nn_input(self, flag=None, down_sample_ratio=config.default_ratio):
        print_long("Converting " + flag + " info...")
        cur_hist_count = 0
        try:
            if flag == "map":
                if self.map_ready:
                    self.dim, self.map_intensity, self.map_intensity_scale, self.new_dim, self.coord_frame, \
                    self.raw_map_array, self.resolution = \
                        bag_to_hdf5.parse_map_data_from_dict(down_sample_ratio, self.combined_dict['map'])

            elif flag == "data":
                # reset patience clock
                if self.map_ready:
                    self.raw_map_array, self.hist_env_maps, self.hist_car_maps, self.lane_map, self.obs_map, self.goal_map \
                        = bag_to_hdf5.create_maps_inner(self.combined_dict['map'], self.raw_map_array)
                else:
                    pass
                    # print('Creating null maps')
                    # self.raw_map_array, self.hist_env_maps, self.hist_car_maps, self.lane_map, self.obs_map, self.goal_map \
                    #     = bag_to_hdf5.create_null_maps()

                hist_cars, hist_peds, cur_hist_count = \
                    bag_to_hdf5.get_bounded_history(self.combined_dict, 'agents_hist')

                # if not self.check_history_completeness(hist_cars):
                #     return False
                if cur_hist_count == 0:
                    print_long("no history agents yet...")
                    return False, cur_hist_count

                bag_to_hdf5.clear_maps(self.hist_env_maps, self.hist_car_maps, self.lane_map, self.obs_map, self.goal_map)

                self.coord_frame = None
                agents_are_valid, elapsed_time = \
                    bag_to_hdf5.process_exo_agents(hist_cars=hist_cars, hist_exo_agents=hist_peds,
                                                   hist_env_maps=self.hist_env_maps, dim=self.dim,
                                                   resolution=self.resolution, coord_frame=self.coord_frame)
                print_long("Exo-agents processing time: " + str(elapsed_time) + " s")

                if not agents_are_valid:
                    return False, cur_hist_count  # report invalid peds

                self.coord_frame = None

                elapsed_time = bag_to_hdf5.process_maps_inner(down_sample_ratio, self.raw_map_array, self.output_dict,
                                                              self.hist_env_maps)
                print_long("Agent map processing time: " + str(elapsed_time) + " s")

                elapsed_time = bag_to_hdf5.process_car_inner(self.output_dict, self.combined_dict, hist_cars,
                                                             self.hist_car_maps, self.goal_map,
                                                             self.dim, down_sample_ratio, self.coord_frame,
                                                             self.resolution, mode='online')
                print_long("Ego car processing time: " + str(elapsed_time) + " s")

            elif flag == 'lanes':
                hist_cars, hist_peds, cur_hist_count = \
                    bag_to_hdf5.get_bounded_history(self.combined_dict, 'agents_hist')
                if cur_hist_count == 0:
                    print_long("no history agents yet...")
                    return False, cur_hist_count
                self.coord_frame = None
                elapsed_time = bag_to_hdf5.process_lanes_inner(self.output_dict, self.combined_dict, hist_cars[0],
                                                               self.lane_map, down_sample_ratio, self.coord_frame,
                                                               self.resolution, mode='online')
                print_long("Lane processing time: " + str(elapsed_time) + " s")

            elapsed_time = self.format_nn_input(flag)
            print_long("Input image formatting time: " + str(elapsed_time) + " s")
            self.record_labels()

            print_long(flag + " update done.")
            return True, cur_hist_count
        except Exception as e:
            error_handler(e)
            return False, cur_hist_count

    def check_history_completeness(self, hist_cars):
        history_is_complete = True
        for i in reversed(range(0, len(hist_cars))):
            if hist_cars[i] is None:
                history_is_complete = False

        return history_is_complete

    def record_labels(self):
        self.cur_data['true_steer_normalized'] = self.true_steering_norm
        self.cur_data['true_acc'] = self.true_acc
        self.cur_data['true_vel'] = self.true_vel
        self.cur_data['true_lane'] = self.true_lane

    def format_nn_input(self, flag):
        start = time.time()
        try:
            if flag == "lanes":
                # print_long('lane points {}'.format(self.output_dict['lane']))
                if self.output_dict['lane'] is not None:
                    self.cur_data['nn_input'][0, 0, config.channel_lane, ...] = self.output_dict['lane']
                else:
                    self.cur_data['nn_input'][0, 0, config.channel_lane, ...] = 0.0

            if flag == "data":
                if self.output_dict['maps'] is not None:
                    for c in range(0, config.num_hist_channels):
                        self.cur_data['nn_input'][0, 0, config.channel_map[c]] = self.output_dict['maps'][c]
                else:
                    for c in range(0, config.num_hist_channels):
                        self.cur_data['nn_input'][0, 0, config.channel_map[c], ...] = 0.0

                agent_flag = 'car'
                if config.use_goal_channel:
                    if self.output_dict[agent_flag]['goal'] is not None:
                        self.cur_data['nn_input'][0, 0, config.channel_goal, ...] = self.output_dict[agent_flag]['goal']
                    else:
                        self.cur_data['nn_input'][0, 0, config.channel_goal, ...] = 0.0

                if config.use_hist_channels:
                    for c in range(0, config.num_hist_channels):
                        if self.output_dict[agent_flag]['hist'][c] is not None:
                            self.cur_data['nn_input'][0, 0, config.channel_hist[c], ...] = \
                                self.output_dict[agent_flag]['hist'][c]
                        else:
                            self.cur_data['nn_input'][0, 0, config.channel_hist[c], ...] = 0.0

                if self.output_dict[agent_flag]['semantic'] is not None:
                    self.cur_data['nn_semantic_input'] = self.output_dict[agent_flag]['semantic']

        except Exception as e:
            error_handler(e)
        finally:
            end = time.time()
            return end - start

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
