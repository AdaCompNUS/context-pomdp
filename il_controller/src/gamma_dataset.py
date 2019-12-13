import sys

from torch.utils.data import Dataset
import deepdish as dd
from Data_processing.global_params import config
from dataset import set_encoders

from transforms import ang_transform_normalized_to_degree, Identity, Fliplr

sys.path.append('.')
import numpy as np


def reset_global_params_for_dataset(cmd_args):
    config.fit_acc = False
    config.fit_lane = False
    config.fit_action = False
    config.fit_all = False
    config.fit_val = False
    config.fit_vel = True
    config.fit_ang = True

    config.use_vel_head = True
    config.vel_max = 1.0
    config.imsize = 100
    cmd_args.imsize = config.imsize
    config.num_hist_channels = 2

    ''' Channel codes '''
    # channel 0-3, exo history 1-4
    config.channel_map = []
    for i in range(config.num_hist_channels):
        config.channel_map.append(i)
    # channel 4, lanes
    config.channel_lane = config.num_hist_channels
    if not config.use_goal_channel:
        config.gppn_input_end = config.channel_lane + 1
    # channel 5, goal path
    config.channel_goal = config.num_hist_channels + 1
    if config.use_goal_channel:
        config.gppn_input_end = config.channel_goal + 1
    # channel 6-9, ego history 1-4
    config.channel_hist = []
    for i in range(config.num_hist_channels):
        config.channel_hist.append(i + config.gppn_input_end)
    # total number of channels
    if config.use_hist_channels:
        config.total_num_channels = config.channel_hist[-1] + 1
    else:
        config.total_num_channels = config.gppn_input_end
    ''' Channel codes '''

    config.num_gppn_inputs = config.channel_hist[0]
    config.num_semantic_inputs = 2

    print("======== resetting global parameters for the gamma dataset =========")


class GammaDataset(Dataset):
    def __init__(self, filename, start, end):  # start and end are in terms of percentage
        self.data = dd.io.load(filename)
        self.start = int((start * self.data['state_frames'].shape[0]))
        self.end = int((end * self.data['state_frames'].shape[0]))
        self.data_len = self.end - self.start

        print('state_frames shape {}'.format(self.data['state_frames'][0].shape))
        self.state_frames_table = self.data['state_frames']
        print('state_prev_controls {}'.format(self.data['state_prev_controls'][0]))
        self.state_prev_controls_table = self.data['state_prev_controls']
        print('controls {}'.format(self.data['controls'][0]))
        self.controls_table = self.data['controls']

        self.encode_steer_from_degree, self.encode_acc_from_id, self.encode_vel_from_raw, self.encode_lane_from_int = \
            set_encoders()

        self.transform_list = [Identity(), Fliplr()]

    def __len__(self):
        return len(self.transform_list) * self.data_len

    def get_augmented_data(self, transform_idx, input_data, steer, lane=0):
        return self.transform_list[transform_idx](input_data, steer, lane)

    def __getitem__(self, idx):

        data_index = idx % self.data_len
        transform_idx = idx // self.data_len
        acc_id_label, v_label, lane_label = 0, 0.0, 0

        input_data = self.state_frames_table[self.start + data_index]
        semantic_data = self.state_prev_controls_table[self.start + data_index]
        vel_label = self.controls_table[self.start + data_index][0]
        steer_normalized = self.controls_table[self.start + data_index][0]

        steer_degree = ang_transform_normalized_to_degree(steer_normalized)
        input_data, steer_degree, _ = self.get_augmented_data(
            transform_idx, np.expand_dims(input_data, axis=0), steer_degree)

        steer_label = self.encode_steer_from_degree(steer_degree)  # this returns only the index of the non zero bin
        vel_label = self.encode_vel_from_raw(vel_label)

        # print('getting item {} {} {} {} {} {} {}'.format(
        #     input_data.shape, semantic_data, v_label, acc_id_label, steer_label, vel_label, lane_label))

        return input_data.astype(np.float32), \
               semantic_data, v_label, acc_id_label, steer_label, vel_label, lane_label


if __name__ == '__main__':
    print('Loading data...')
