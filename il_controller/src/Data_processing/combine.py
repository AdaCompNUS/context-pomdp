from global_params import config

if config.pycharm_mode:
    import pyros_setup

    pyros_setup.configurable_import().configure('mysetup.cfg').activate()

import sys

sys.path.append('./')

print(sys.path)

import deepdish as dd
import fnmatch
import os
import argparse
from transforms import *
from visualization import *
import global_params
import h5py
import time
import multiprocessing

from multiprocessing import Process, Queue

import gc

config = global_params.config
imsize = config.imsize
num_agents = 1 + 0
# LARGE_NO = 300000
LARGE_NO = 3000
populate_images = PopulateImages()

num_threads = 1


def INPUT(out_str):
    if sys.version_info > (3, 0):
        return input(out_str)
    else:
        return raw_input(out_str)


def save(files, filename, flag=-1):
    '''
    Args:
    flag: -1:                   means list of val and test files
          [0, num scenearios):  means dict of train set and

    Return:

    '''
    # create file handle

    if flag > -1:
        if sys.version_info[0] > 2:
            os.makedirs(config.train_set_path, exist_ok=True)
        else:
            if not os.path.exists(config.train_set_path):
                os.makedirs(config.train_set_path)

        split_pos = filename.find('.h5')
        if split_pos >= 0:
            filename = filename[0: filename.find('.h5')]
        filename = config.train_set_path + '/' + filename + '_' + str(flag) + '.h5'

    if file_exist(filename):
        return

    counter = int(0)

    with h5py.File(filename, 'a') as f:
        # open extendible dataset
        acc_id_labels_array, ang_normed_labels_array, data_array, cart_data_array, \
        v_labels_array, vel_labels_array, lane_labels_array = \
            allocate_containters()
        # dont discretize angular labels
        print("total files %d" % len(files))

        traj_num = 0
        points_num = []

        visualization_done = False
        for file in files:
            if "train" not in file and "val" not in file and "test" not in file:

                d, skip_file = load_file(file)

                if skip_file:
                    continue  # skip the file
                traj_num += 1
                old_counter = counter
                points_num.append(len(d.keys()))
                step_in_file = 0
                for key in d.keys():  # key is the original index in separate h5 files
                    data, cart_data, value, acc_id, ang_normalized, vel, lane = populate_images(d[key], False)
                    put_images_in_dataset(acc_id, acc_id_labels_array, ang_normalized, ang_normed_labels_array, counter,
                                          data_array, data, cart_data, cart_data_array,
                                          v_labels_array, value, vel, vel_labels_array, lane, lane_labels_array)

                    file_base_name = os.path.basename(file)
                    file_base_name = os.path.splitext(file_base_name)[0][29:]
                    visualization_done = visualize_images(step_in_file, data, visualization_done,
                                                          file_flag=file_base_name)
                    visualization_done = visualize_cart_images(step_in_file, cart_data, data, visualization_done,
                                                               file_flag=file_base_name)

                    step_in_file += 1
                    counter += 1
                if counter > old_counter:
                    print("counter value increased to %d" % counter)
                else:
                    print("counter unchanged: num_keys=%d" % len(d.keys()))
                    # pdb.set_trace()
                    continue  # return

        print("Saving data...")
        time.sleep(3)

        save_dataset_to_h5(acc_id_labels_array, ang_normed_labels_array, counter, data_array, cart_data_array, f,
                           v_labels_array, vel_labels_array, lane_labels_array, traj_num, points_num)

    print("Collection memory garbages")
    del data_array, cart_data_array, acc_id_labels_array, vel_labels_array, ang_normed_labels_array, \
        v_labels_array, lane_labels_array
    gc.collect()

    print("%s saved with %d data points" % (filename, counter))


def save_dataset_to_h5(acc_id_labels_array, ang_normalized_labels_array, counter, data_array, cart_data_array, f,
                       v_labels_array, vel_labels_array, lane_labels_array, traj_num, points_num):
    try:
        print("data shape: {}".format(data_array[0].shape))
        f.create_dataset('data', maxshape=(None, num_agents, config.total_num_channels, imsize, imsize),
                         data=data_array[0:counter], dtype='f4')

        print("Saving cart data...")
        print("data shape: {}".format(cart_data_array[0].shape))
        f.create_dataset('cart_data', maxshape=(None, 2 * (1 + config.num_agents_in_map) * config.num_hist_channels),
                         data=cart_data_array[0:counter], dtype='f4')

        print("Saving labels...")
        f.create_dataset('v_labels', maxshape=(None, 1), data=v_labels_array[0:counter], dtype='f4')
        f.create_dataset('acc_id_labels', maxshape=(None, 1), data=acc_id_labels_array[0:counter],
                         dtype='f4')
        f.create_dataset('ang_normalized_labels', maxshape=(None, 1), data=ang_normalized_labels_array[0:counter],
                         dtype='f4')
        f.create_dataset('vel_labels', maxshape=(None, 1), data=vel_labels_array[0:counter],
                         dtype='f4')
        f.create_dataset('lane_labels', maxshape=(None, 1), data=lane_labels_array[0:counter],
                         dtype='f4')

        if config.sample_mode == 'hierarchical':
            f.create_dataset('traj_num', data=traj_num, dtype='int32')
            f.create_dataset('points_num', data=points_num, dtype='int32')
    except Exception as e:
        print("Investigate table")
        error_handler(e)
        pdb.set_trace()


new_file_flags = []
max_vis_traj = 1


def visualize_images(counter, data, visualization_done, file_flag=''):
    if not visualization_done:
        visualize(data, 'combine/' + file_flag + '_' + str(counter), root="Data_processing/")

        if file_flag not in new_file_flags:
            new_file_flags.append(file_flag)
        if len(new_file_flags) >= max_vis_traj:
            visualization_done = True
    return visualization_done


def visualize_cart_images(counter, cart_data, data, visualization_done, file_flag=''):
    if not visualization_done:
        visualize_both_agent_inputs(cart_data, data, 'combine/' + file_flag + '_cart_' + str(counter),
                                    root="Data_processing/")

        if file_flag not in new_file_flags:
            new_file_flags.append(file_flag)
        if len(new_file_flags) >= max_vis_traj:
            visualization_done = True
    return visualization_done


def put_images_in_dataset(acc_id, acc_id_labels_array, ang_normalized, ang_normed_labels_array, counter, data_array,
                          data, cart_data, cart_data_array, v_labels_array,
                          value, vel, vel_labels_array, lane, lane_labels_array):
    try:
        data_array[counter] = data
        # cart_data_array[counter] = cart_data
        v_labels_array[counter][0] = value
        acc_id_labels_array[counter][0] = acc_id
        ang_normed_labels_array[counter][0] = ang_normalized
        vel_labels_array[counter][0] = vel
        lane_labels_array[counter][0] = lane

    except Exception as e:
        print("Investigate file")
        error_handler(e)
        pdb.set_trace()


def allocate_containters():
    data_array = np.zeros((LARGE_NO, num_agents, config.total_num_channels, imsize, imsize), dtype=np.float32)
    cart_data_array = np.zeros((LARGE_NO, 2 * (1 + config.num_agents_in_map) * config.num_hist_channels),
                               dtype=np.float32)

    v_labels_array = np.zeros((LARGE_NO, 1), dtype=np.float32)
    acc_id_labels_array = np.zeros((LARGE_NO, 1), dtype=np.float32)
    ang_normalized_labels_array = np.zeros((LARGE_NO, 1), dtype=np.float32)
    vel_labels_array = np.zeros((LARGE_NO, 1), dtype=np.float32)
    lane_labels_array = np.zeros((LARGE_NO, 1), dtype=np.float32)

    return acc_id_labels_array, ang_normalized_labels_array, data_array, cart_data_array, \
           v_labels_array, vel_labels_array, lane_labels_array


def load_file(file):
    skip_file = False
    d = None
    print("reading file " + file)
    try:
        d = dd.io.load(file)
    except Exception as e:
        skip_file = True
        print("Unable to load " + file)
        print("Exception: ", e)
        choice = INPUT('Delete file? ')
        if choice == 'y':
            os.remove(file)
            print(file + " removed")
        else:
            print(file + " skipped")
    return d, skip_file


class Combiner(Process):
    def __init__(self, id, queue):
        Process.__init__(self)
        self.queue = queue
        self.id = id

    def run(self):
        while True:
            files, h5_name = self.queue.get()

            if files:
                save(files, h5_name)

            if self.queue.empty():
                break

        print("Thread %d finished saving %s" % (self.id, h5_name))


def parse_cmd_args():
    global config
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bagspath', type=str,
                        default='./jul22/', help='Path to data file')
    parser.add_argument('--outfile', type=str,
                        default='train.h5', help='Path to data file')
    parser.add_argument('--outfolder',
                        type=str,
                        default=None,
                        help='path for the set of training h5 files (for hierarchical mode only)')
    parser.add_argument('--outpath',
                        type=str,
                        default=None,
                        help='path to store combined data sets')
    parser.add_argument('--samplemode',
                        type=str,
                        default=config.sample_mode,
                        help='sampling mode')
    parser.add_argument(
        '--nped',
        type=int,
        default=0,
        help='Number of neighbouring peds to consider')
    bagspath = parser.parse_args().bagspath
    train_filename = parser.parse_args().outfile
    config.sample_mode = parser.parse_args().samplemode
    config.train_set_path = parser.parse_args().outfolder
    config.out_path = parser.parse_args().outpath

    if config.train_set_path is not None:
        config.sample_mode = 'hierarchical'
        print("training set folder", config.train_set_path)

    print("===================== cmd_args ======================")
    print("bagspath: ", bagspath)
    print("train_filename: ", train_filename)
    print("0: ", 0)
    print("config.sample_mode: ", config.sample_mode)
    print("config.train_set_path: ", config.train_set_path)
    print("===================== cmd_args ======================")

    return bagspath, train_filename


def collect_h5_files(bag_root_path, suffix='.h5'):
    # walk into subdirectories recursively and collect all .h5 files

    if config.sample_mode == 'hierarchical':
        '''
        Args:
        bag_root_path: path of the folder containg all scenarios folder
        suffix:        default: '.h5'
    
        Return:
        dict_h5:       a dict, key: sceneario folder path, value: full path of the files with suffix
        dict_h5_num:   a dict, key: sceneario folder path, value: num of the files with suffix
        '''
        dict_h5 = {}
        dict_h5_num = {}

        for root, dirs, files in os.walk(bag_root_path):
            name_list = []
            for filename in fnmatch.filter(files, '*' + suffix):
                name_list.append(os.path.join(root, filename))

            dict_h5[root] = name_list
            dict_h5_num[root] = len(name_list)

        if bag_root_path in dict_h5.keys():
            del dict_h5[bag_root_path]
        if bag_root_path in dict_h5_num.keys():
            del dict_h5_num[bag_root_path]
        return dict_h5, dict_h5_num

    else:
        files = list([])
        for root, dirnames, filenames in os.walk(bagspath):
            for filename in fnmatch.filter(filenames, '*.h5'):
                files.append(os.path.join(root, filename))

        print("{} h5 files found in {}".format(str(len(files)), bagspath))
        return files


def split_h5_files(files=[], file_dict=None, file_num_dict=None):
    if config.sample_mode == 'hierarchical':
        '''
            Args:
            file_dict:      a dict, key: sceneario folder path, value: full path of the files
            file_num_dict:  a dict, key: sceneario folder path, value: num of the files

            Return:
            file_dict:      test set, a dict, key: sceneario folder path, value: full path of the files
            val_files:      val  set, a list, element: full path of the file
            test_files:     test set, a list, element: full path of the file

            '''

        print("Data split: train %.0f%%, validation %.0f%%, test %.0f%%" % (
            config.train_split * 100, config.val_split * 100, config.test_split * 100))
        h5_num = sum(file_num_dict.values())
        scene_num = len(file_dict)
        test_size = round(h5_num * config.test_split)
        test_size_scene = int(test_size // scene_num)
        val_size = round(h5_num * config.val_split)
        val_size_scene = int(val_size // scene_num)

        val_files = []
        test_files = []

        for scene_folder in file_dict.keys():
            random.shuffle(file_dict[scene_folder])
            test_files += file_dict[scene_folder][0: test_size_scene]
            val_files += file_dict[scene_folder][test_size_scene: val_size_scene]
            file_num_dict[scene_folder] -= (test_size_scene + val_size_scene)
            del file_dict[scene_folder][0: test_size_scene + val_size_scene]
        return file_dict, test_files, val_files

    else:
        print("Data split: train %.0f%%, validation %.0f%%, test %.0f%%" % (
            config.train_split * 100, config.val_split * 100, config.test_split * 100))
        train_files = files[0:int(round(config.train_split * len(files)))]
        test_files = []
        val_files = []
        if (config.test_split > 0.0):
            test_files = files[int(round(config.train_split * len(files))):int(
                round((config.train_split + config.test_split) * len(files)))]
        if (config.val_split > 0.0):
            val_files = files[int(round((config.train_split + config.test_split) * len(files))):]
        return train_files, test_files, val_files


def file_exist(filename):
    if os.path.isfile(filename):
        choice = INPUT(filename + ' exists. Delete file?')
        print("You have chosen %s for %s" % (choice, filename))

        if choice == 'y':
            os.remove(filename)
            return False
        else:
            print(filename + " unchanged")
            return True  # skip the file
    else:
        return False


def combine_files_into_datasets_parallel(train_files, test_files, val_files):
    # use threads to combine training, validation, and test thread simultaneously
    queue = Queue()
    workers = []
    i = 0
    for x in range(num_threads):
        worker = Combiner(i, queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
        workers.append(worker)
        i += 1
    if not file_exist(train_filename):
        queue.put((train_files, train_filename))
    val_filename = 'val.h5'
    if not file_exist(val_filename):
        queue.put((val_files, val_filename))
    test_filename = 'test.h5'
    if len(test_files) > 0 and not file_exist(test_filename):
        queue.put((test_files, test_filename))

    # Causes the main thread to wait for the queue to finish processing all the
    for worker in workers:
        worker.join()


def combine_files_into_datasets_serial(train_files, test_files, val_files):
    val_filename = 'val.h5'
    test_filename = 'test.h5'

    if config.out_path:
        global train_filename
        train_filename = config.out_path + '/' + train_filename
        val_filename = config.out_path + '/' + val_filename
        test_filename = config.out_path + '/' + test_filename
        print("ouput file names:", train_filename, val_filename, test_filename)

    if config.sample_mode == 'hierarchical':
        train_dict = train_files
        scene_count = int(0)
        for key in train_dict.keys():
            save(train_dict[key], train_filename, scene_count)
            scene_count += 1
    else:
        if not file_exist(train_filename):
            save(train_files, train_filename)

    gc.collect()  # to free up the memory usage
    # del gc.garbage[:]

    if not file_exist(val_filename):
        save(val_files, val_filename)
    if test_files and len(test_files) > 0 and not file_exist(test_filename):
        save(test_files, test_filename)

    print("All data sets saved")


if __name__ == "__main__":
    bagspath, train_filename = parse_cmd_args()
    clear_png_files("Data_processing/", subfolder='visualize', remove_flag='')

    if config.sample_mode == 'hierarchical':
        dict_h5, dict_h5_num = collect_h5_files(bagspath, suffix='.h5')
        train_dict, test_files, val_files = split_h5_files(file_dict=dict_h5, file_num_dict=dict_h5_num)
        combine_files_into_datasets_serial(train_dict, test_files, val_files)

    else:
        files = collect_h5_files(bagspath)
        random.shuffle(files)
        train_files, test_files, val_files = split_h5_files(files=files)
        combine_files_into_datasets_serial(train_files, test_files, val_files)

    print("Done")
