import sys

from torch.utils.data import Dataset
import deepdish as dd
from Data_processing.global_params import config

sys.path.append('.')


class GammaDataset(Dataset):
    def __init__(self, filename, start, end):  # start and end are in terms of percentage
        self.data = dd.io.load(filename)
        self.start = int((start * self.data['state_frames'].shape[0]))
        self.end = int((end * self.data['state_frames'].shape[0]))

        print('state_frames shape {}'.format(self.data['state_frames'][0].shape))
        self.state_frames_table = self.data['state_frames']
        print('state_prev_controls {}'.format(self.data['state_prev_controls'][0]))
        self.state_prev_controls_table = self.data['state_prev_controls']
        print('controls {}'.format(self.data['controls'][0]))
        self.controls_table = self.data['controls']
        config.fit_acc = False
        config.fit_lane = False
        config.fit_action = False
        config.fit_all = False
        config.fit_val = False

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        input_data = self.state_frames_table[self.start + idx]
        semantic_data = self.state_prev_controls_table[self.start + idx]
        vel_label = self.controls_table[self.start + idx][0]
        steer_label = self.controls_table[self.start + idx][0]
        acc_id_label, v_label, lane_label = None, None, None

        return input_data, semantic_data, v_label, acc_id_label, steer_label, vel_label, lane_label


if __name__ == '__main__':
    print('Loading data...')
