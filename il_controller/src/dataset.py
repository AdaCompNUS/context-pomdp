## dataset loading
import time
import torch
import torch.utils.data as data
import numpy as np
import deepdish as dd
import ipdb as pdb
from transforms import *
from Data_processing import global_params
import random
import os
import copy

import cycler


config = global_params.config


# child class of pytorch data.Dataset
def set_encoders():
    if config.head_mode == "mdn":
        encode_steer_from_degree = MdnSteerEncoderDegree2Normalized()  # conversion from id to normalized steering
        encode_acc_from_id = MdnAccEncoderID2Normalized()  # conversion from id to normalized acceleration
        encode_vel_from_raw = MdnVelEncoderRaw2Normalized()  # conversion from id to normalized command velocity
    elif config.head_mode == "hybrid":
        encode_steer_from_degree = SteerEncoderDegreeToOnehot()  # one-hot vector of steering
        encode_acc_from_id = MdnAccEncoderID2Normalized()  # conversion from id to normalized acceleration
        encode_vel_from_raw = MdnVelEncoderRaw2Normalized()  # conversion from id to normalized command velocity
    else:
        encode_steer_from_degree = SteerEncoderDegreeToOnehot()  # one-hot vector of steering
        encode_acc_from_id = AccEncoderIDToOnehot()  # one-hot vector of acceleration
        encode_vel_from_raw = VelEncoderRaw2Onehot()  # one-hot vector of command velocity
    encode_lane_from_int = LaneEncoderIntToOnehot()
    return encode_steer_from_degree, encode_acc_from_id, encode_vel_from_raw, encode_lane_from_int


class IdxLooper:
    """ Index looper

        Instances: shuffled_traj_indices, shuffled_scene_indices

        Functionality: Sample indices in random orders, until the data in the scene/trajectory are exhausted.


        num_indices: number of indices
        indices: raw list of indices 0~num_indices
        visit_count: number of visit for each index
        max_visit_count: maximum visit count allowed for an index = the number of data in the scene/trajectory
        active_indices: indices whose visit_count haven't exceeded the max_visit_count (legal to be sampled).
        iter_pos: current iterator pos
    """

    def __init__(self, indices, num_indices, max_visit_count, aug_factor, content_flag):

        self.num_indices = num_indices
        random.shuffle(indices)
        self.indices = indices
        self.iter_pos = -1

        self.max_visit_count = [x * aug_factor for x in max_visit_count]
        self.visit_count = np.zeros(num_indices, dtype=np.int)

        self.content_flag = content_flag

        # TODO: maintain an active list of indices that haven't exceeded the max_visit_count.
        # TODO: init as a copy of indices
        self.active_indices = copy.deepcopy(self.indices)

    def __iter__(self):
        return self

    def next(self):
        cur_idx = None
        try:
            if len(self.active_indices) == 0:
                return None
                # TODO: remove a index if hitting max_visit_count
                # TODO: go on to sample the next index
                # TODO: return None only when active_indices is empty
            cur_idx = self.get_next()
            self.visit_count[cur_idx] += 1
            if self.visit_count[cur_idx] >= self.max_visit_count[cur_idx]:
                del self.active_indices[self.active_indices.index(cur_idx)]

        except Exception as e:
            print(e)

        return cur_idx

    def get_next(self):
        
        if self.iter_pos >= len(self.active_indices) - 1:
           random.shuffle(self.active_indices)
           self.iter_pos = -1
        self.iter_pos += 1

        # TODO: sample from active list only
        return self.active_indices[self.iter_pos]

    def get_count(self):
        return sum(self.visit_count)

    def reset(self):
        # TODO: call this when all data in the training set are used up (best do it in get_train_item )
        # TODO: hint: when shuffled_scene_indices returns None sample, it indicates that the training set is exhausted.
        self.visit_count = np.zeros(self.num_indices, dtype=np.int)
        self.iter_pos = -1
        random.shuffle(self.indices)

        # TODO: reset the active list as a copy of indices
        self.active_indices = copy.deepcopy(self.indices)


class IdxIterator:
    """ Index iterator

        Instances: shuffled_data_indices

        Functionality: Sample indices in random orders, until the data in the indices list are exhausted.
                       Each index can be only sampled once.


        num_indices: number of indices
        indices: raw list of indices 0~num_indices
        iter_pos: current iterator pos
    """
    def __init__(self, indices, num_indices):
        self.num_indices = num_indices
        self.indices = indices
        self.iter_pos = -1

    def __iter__(self):
        # self.pos = -1
        return self

    def next(self):
        if self.iter_pos >= self.num_indices - 1:
            return None  # end of iteration
        self.iter_pos += 1

        cur_data = self.indices[self.iter_pos]
        return cur_data

    def reset(self):
        # TODO: call this when all scenarios data are used up (best do it in get_train_item )

        random.shuffle(self.indices)
        self.iter_pos = -1


class DrivingData(data.Dataset):
    def __init__(self, filename=None, flag=None):
        self.data = None
        self.data_len = None

        self.flag = flag

        self.scene_num = 0  # used for train set
        self.accumulated_data_length = {}  # used for train set
        self.shuffled_scene_indices = None
        self.shuffled_traj_indices = {}  # used for train set, key: folder, value: [No. of traj, num of traj]
        self.shuffled_data_indices = {}
        self.trans_idx = {}
        self.transform_list = [Identity(), Rot(1), Rot(2), Rot(3), Flipud(), FlipRot(1), FlipRot(2), FlipRot(3)]

        if self.flag == 'Training' and config.sample_mode == 'hierarchical':
            self.load_train_dataset_from_h5(filename, flag)
        else:
            self.load_dataset_from_h5(filename, flag)

        self.encode_value = ValueEncoderRaw2Normalized()  # rescale value

        self.encode_input = InputEncoder()  # normalize input images

        self.encode_steer_from_degree, self.encode_acc_from_id, self.encode_vel_from_raw, self.encode_lane_from_int = \
            set_encoders()

    def load_dataset_from_h5(self, file, flag):
        try:
            start_time = time.time()
            self.data = dd.io.load(file)
            print("keys in loaded data set", self.data.keys())
            self.data_len = self.data['data'].shape[0]
            elapsed_time = time.time() - start_time
            # ouput information about the data set
            print(flag + " h5 file loaded in %.1f seconds." % elapsed_time),
            print("%d data points: " % self.data_len),
            image_dimx = self.data['data'].shape[3]
            image_dimy = self.data['data'].shape[4]
            assert image_dimx == image_dimy  # only support square images
            if flag == "Training":
                config.imsize = image_dimx
            else:  # test and validation set images should match training images in dimensions
                assert config.imsize == image_dimx
            print("input images have dim %d * %d." % (image_dimx, image_dimy))
        except Exception as e:
            error_handler(e)

    def load_train_dataset_from_h5(self, file_name_tag, flag):
        start_time = time.time()
        self.data_len = {}
        self.data = []

        root = config.train_set_path
        all_files = os.listdir(root)
        for each_file in all_files:
            if each_file.find(file_name_tag + '_') > -1:
                self.scene_num += 1

        augmentation_factor = 1
        if config.augment_data:
            augmentation_factor = len(self.transform_list)

        for scene in range(self.scene_num):

            filename = os.path.join(root, file_name_tag) + '_' + str(scene) + '.h5'
            data = dd.io.load(filename)

            num_trajectories = data['traj_num']  # a int, num of traj in the .h5 file
            traj_len_list = data['points_num']  # a list, num of data points in each traj within the .h5 file

            self.data.append(data)
            self.data_len[str(scene)] = sum(traj_len_list)

            self.accumulated_data_length[str(scene)] = [0]
            for pos in range(1, num_trajectories + 1):
                traj = pos -1
                accu_data_length = self.get_start_data_pos(scene, pos - 1) + traj_len_list[traj]
                self.accumulated_data_length[str(scene)].append(accu_data_length)

            self.shuffled_data_indices[str(scene)] = []
            for traj in range(0, num_trajectories):
                data_ids = list(range(traj_len_list[traj] * augmentation_factor))
                random.shuffle(data_ids)
                idx_iter = IdxIterator(data_ids, len(data_ids))
                self.shuffled_data_indices[str(scene)].append(idx_iter)

            traj_ids = list(range(num_trajectories))
            # random.shuffle(traj_ids)
            idx_iter = IdxLooper(traj_ids, len(traj_ids), list(traj_len_list), augmentation_factor,
                                 'trajectories')
            self.shuffled_traj_indices[str(scene)] = idx_iter

        scene_ids = list(range(self.scene_num))
        scene_max_visit_count = []
        for scene_idx in scene_ids:
            scene_max_visit_count.append(self.data_len[str(scene_idx)])
        # random.shuffle(scene_ids)
        idx_iter = IdxLooper(scene_ids, len(scene_ids), scene_max_visit_count, augmentation_factor, 'scenes')
        self.shuffled_scene_indices = idx_iter

        elapsed_time = time.time() - start_time
        # ouput information about the data set
        print(str(self.scene_num) + ' ' + flag + " h5 files loaded in %.1f seconds." % elapsed_time)
        print("%d data points: " % sum(self.data_len.values()))
        image_dimx = self.data[0]['data'].shape[3]
        image_dimy = self.data[0]['data'].shape[4]
        assert image_dimx == image_dimy
        config.imsize = image_dimx
        print("input images have dim %d * %d." % (image_dimx, image_dimy))

    def get_datalen(self, scene):
        return self.accumulated_data_length[str(scene)][-1]

    def get_augmented_data(self, transform_idx, input_data, steer):
        return self.transform_list[transform_idx](input_data, steer)

    def get_item(self, global_index):

        data_index = global_index % self.data_len
        transform_idx = global_index // self.data_len
        if not config.augment_data:  # no data augmentation
            transform_idx = 0  # Identity transformation
        # take the steering in radians, convert to degrees
        steer_normalized = self.data['ang_normalized_labels'][data_index]
        steer_degree = ang_transform_normalized_to_degree(steer_normalized)
        input_data = self.data['data'][data_index]

        # validate_map(input_data, "DataSet __getitem__")
        input_data, steer_degree = self.get_augmented_data(transform_idx, input_data, steer_degree)
        input_data = self.encode_input(input_data)

        steer_label = self.encode_steer_from_degree(steer_degree)  # this returns only the index of the non zero bin
        # print("raw data {}, degree {}, bin_idx {}".format(steer_normalized[0], steer_degree[0], steer_label), flush=True)
        acc_id_label = self.encode_acc_from_id(self.data['acc_id_labels'][data_index])
        vel = 0.0
        if config.fit_vel or config.fit_action or config.fit_all:
            vel = self.data['vel_labels'][data_index]
        vel_label = self.encode_vel_from_raw(vel)

        lane = self.data['lane_labels'][data_index]
        lane_label = self.encode_lane_from_int(lane)

        v_label = self.encode_value(self.data['v_labels'][data_index])

        return input_data, v_label, acc_id_label, steer_label, vel_label, lane_label

    def get_train_item(self, global_index):

        scene_index = next(iter(self.shuffled_scene_indices))

        if scene_index is None:
            self.reset_data_indices_recursive()
            scene_index = next(iter(self.shuffled_scene_indices))

        scene_str = str(scene_index)

        traj_index = next(iter(self.shuffled_traj_indices[scene_str]))

        data_index_in_traj = next(iter(self.shuffled_data_indices[scene_str][traj_index]))

        transform_idx = 0  # default for no augmentation
        if config.augment_data:  # data augmentation
            transform_idx = data_index_in_traj % len(self.transform_list)
            data_index_in_traj = data_index_in_traj // len(self.transform_list)

        data_index_in_dataset = self.get_start_data_pos(scene_index, traj_index) + data_index_in_traj

        steer_normalized = self.data[scene_index]['ang_normalized_labels'][data_index_in_dataset]
        steer_degree = ang_transform_normalized_to_degree(steer_normalized)

        input_data = self.data[scene_index]['data'][data_index_in_dataset]

        input_data, steer_degree = self.get_augmented_data(transform_idx, input_data, steer_degree)

        input_data = self.encode_input(input_data)

        steer_label = self.encode_steer_from_degree(steer_degree)  # this returns only the index of the non zero bin

        acc_label = self.encode_acc_from_id(self.data[scene_index]['acc_id_labels'][data_index_in_dataset])

        vel = 0.0
        if config.fit_vel or config.fit_action or config.fit_all:
            vel = self.data[scene_index]['vel_labels'][data_index_in_dataset]

        vel_label = self.encode_vel_from_raw(vel)

        lane = self.data[scene_index]['lane_labels'][data_index_in_dataset]
        lane_label = self.encode_lane_from_int(lane)

        v_label = self.encode_value(self.data[scene_index]['v_labels'][data_index_in_dataset])

        return input_data, v_label, acc_label, steer_label, vel_label, lane_label

    def reset_data_indices_recursive(self):
        print('\nTraining Data Set Exaulsted, Reset Now ...\n')
        self.shuffled_scene_indices.reset()
        for scene in range(self.shuffled_scene_indices.num_indices):
            self.shuffled_traj_indices[str(scene)].reset()
            for traj in range(self.shuffled_traj_indices[str(scene)].num_indices):
                self.shuffled_data_indices[str(scene)][traj].reset()
        print('\nTraining Data Set is Reset!\n')

    def get_start_data_pos(self, scene_idx, traj_index):
        return self.accumulated_data_length[str(scene_idx)][traj_index]

    def __getitem__(self, global_index):
        if self.flag == 'Training' and config.sample_mode == 'hierarchical':
            return self.get_train_item(global_index)
        else:
            return self.get_item(global_index)

    def __len__(self):
        if config.augment_data:
            if self.flag == 'Training' and config.sample_mode == 'hierarchical':
                return len(self.transform_list) * sum(self.data_len.values())
            else:
                return len(self.transform_list) * self.data_len
        else:
            if self.flag == 'Training' and config.sample_mode == 'hierarchical':
                return sum(self.data_len.values())
            else:
                return self.data_len
