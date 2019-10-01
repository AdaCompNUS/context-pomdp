# from Data_processing import global_params
from global_params import config

if config.pycharm_mode:
    import pyros_setup
    pyros_setup.configurable_import().configure('mysetup.cfg').activate()

import sys

sys.path.append('./Data_processing/')
sys.path.append('../')

import glob
import deepdish as dd
# import ipdb as pdb
import fnmatch
import os
import argparse
import random
import ipdb as pdb
import h5py

from Data_processing import bag_to_hdf5
import global_params, bag_to_hdf5
from global_params import config

config = global_params.config

import multiprocessing

from multiprocessing import Process, Queue

import time

peds_goal_path = None
num_threads = int(multiprocessing.cpu_count() * 1.0)
# num_threads = 1

class Parser(Process):
    def __init__(self, folder_group, id):
        Process.__init__(self)
        self.queue = queue
        self.id = id

    def run(self):
        while True:
            folder, start, end = self.queue.get()
            # print (self, self.is_alive())
            # Get the work from the queue and expand the tuple
            bag_to_hdf5.main(folder, peds_goal_path, config.num_peds_in_NN, start, end, self.id)

            print("Thread %d finished work %s (%d to %d)" % (self.id, folder, start, end))

            if self.queue.empty():
                break

        print("Thread %d finished work" % self.id)


def count_txt_files(bagspath):
    txt_file_count = 0
    for root, dirnames, filenames in os.walk(bagspath):
        if "_debug" not in root:
            # print("subfolder {} found".format(root))
            for filename in fnmatch.filter(filenames, '*.txt'):
                # append the absolute path for the file
                txt_file_count += 1
    print("%d files found in %s" % (txt_file_count, bagspath))
    return txt_file_count


if __name__ == "__main__":
    # parse command line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bagspath',
        type=str,
        default='test_parse',
        help='Path to data file')
    parser.add_argument(
        '--peds_goal_path',
        type=str,
        default='Maps',
        help='Path to pedestrian goals')

    parser.add_argument(
        '--nped',
        type=int,
        default=config.num_peds_in_NN,
        help='Number of neighbouring peds to consider')

    parser.add_argument(
        '--nsample',
        type=int,
        default=config.num_samples_per_traj,
        help='Number of data points to sample from a trajectory')

    bagspath = parser.parse_args().bagspath
    peds_goal_path = parser.parse_args().peds_goal_path
    config.num_peds_in_NN = parser.parse_args().nped
    config.num_samples_per_traj = parser.parse_args().nsample

    folders = list([])

    file_count = 0

    for subfolder in os.listdir(bagspath):
        folder = os.path.join(bagspath, subfolder)
        print("found folder %s" % folder)

        if os.path.isdir(folder):
            folders.append(folder)
            file_count = max(count_txt_files(folder), file_count)

    print("Number of folders: {}".format(len(folders)))


    queue = Queue()

    i = 0
    # Create 8 worker 
    workers = []
    for x in range(num_threads):
        worker = Parser(queue, i)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
        workers.append(worker)
        i += 1

    # Put the tasks into the queue as a tuple
    chunk = 100
    for i in range(0, file_count//chunk + 1):
        for folder in folders:
            queue.put((folder, i * chunk, (i + 1) * chunk))
            print('Queueing %s from %d to %d' % (folder, i * chunk, (i + 1) * chunk))
            # time.sleep(3)
        # break

    # Causes the main thread to wait for the queue to finish processing all the 
    for worker in workers:
        worker.join()

    print("============================== after join =================================")
