from argparse import Namespace
import torch
import sys, os, time
from inspect import getframeinfo

init_time = time.time()
import traceback


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)
    print('Call-stack:')
    traceback.print_stack()
    sys.stdout.flush()
    exit(-1)
    # sys.exit(str(e))


def print_long(msg):
    frameinfo = getframeinfo(sys._getframe(1))
    print('[{}:{}:{}] {}. ts {:.1f}'.format(os.path.basename(frameinfo.filename), frameinfo.function, frameinfo.lineno,
                                        msg, time.time()-init_time), flush=True)


config = Namespace()
# velocity labels used in data['vel_steer']
config.pycharm_mode = False


''' Hardware settings '''
num_gpus = torch.cuda.device_count()
print("=> num of gpus = {}".format(num_gpus))
config.GPU_devices = None
if num_gpus == 1:
    config.GPU_devices = [0]
elif num_gpus == 2:
    config.GPU_devices = [0, 1]
    config.num_data_loaders = 1
elif num_gpus == 3:
    config.GPU_devices = [0, 1, 2]
elif num_gpus == 4:
    config.GPU_devices = [0, 1, 2, 3]
config.num_data_loaders = num_gpus*4
''' Hardware settings '''


''' Action model settings '''
config.steering_resolution = 5.0  # no unit
config.max_steering_degree = 35.0 
config.num_steering_bins = 2 * int(round(config.max_steering_degree / config.steering_resolution))

config.num_acc_bins = 3
config.acc_resolution = 2.0 / config.num_acc_bins
config.max_acc = 3.0  # in degrees

config.num_vel_bins = 6
config.vel_max = 8.0

config.num_lane_bins = 3
''' Action model settings '''


''' Sampling settings '''
import numpy as np
downscale_count = 4
config.default_ratio = 1.0 / pow(2, downscale_count)  # down sample 3 times
config.num_samples_per_traj = 200
config.min_samples_gap = 1
config.num_agents_in_map = 20
config.buffer_mode = 'full'  # full or replay
config.sample_mode = 'random'  # hierarchical or random
config.data_type = np.uint8  # np.uint8, np.float32
config.data_balancing = True
''' Sampling settings '''


''' Dataset settings '''
config.train_set_path = ''
config.train_split = 0.8  # 0.7
config.val_split = 0.2  # 0.25
config.test_split = 0.00  # 0.05
''' Dataset settings '''


''' Data label settings '''
config.value_normalizer = 10.0
config.label_linear = 0
# config.label_angular = 1
config.label_cmdvel = 1
config.steer_normalized_limit = 0.7
''' Data label settings '''


''' Channel codes '''
config.num_hist_channels = 4
config.use_goal_channel = False
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
config.use_hist_channels = False
config.channel_hist = []
for i in range(config.num_hist_channels):
    config.channel_hist.append(i + config.gppn_input_end)
# total number of channels
if config.use_hist_channels:
    config.total_num_channels = config.channel_hist[-1] + 1
else:
    config.total_num_channels = config.gppn_input_end
''' Channel codes '''


''' Training settings '''
# toggle data augmentation
config.augment_data = True
config.use_leaky_relu = True
config.leaky_factor = 0.01
# toggle dropout
config.do_dropout = False
config.do_prob = 0.5
# L2 regularization factor
config.l2_reg_weight = 0  # 1e-3
# toggle batch norm
config.disable_bn_in_resnet = False
config.norm_mode = 'instance'
config.track_running_stats = True
config.label_smoothing = False
# set default num epochs
config.num_epochs = 50
# adapt learning rates
config.enable_scheduler = True
config.use_adaptive_rl = False
config.adaptive_lr_stride = 100
config.adaptive_lr_decay = 0.2  # 0.1
# choose which loss to use or not
config.fit_ang = False
config.fit_acc = False
config.fit_vel = False
config.fit_val = False
config.fit_action = False
config.fit_all = True
''' Training settings '''


''' NN settings '''
config.default_map_dim = 1024
config.imsize = int(config.default_map_dim / pow(2, downscale_count))
config.image_half_size_meters = 20.0
config.lstm_mode = 'gppn'  # 'gppn' or 'convlstm'
config.vanilla_resnet = False
# GPPN params
config.num_gppn_inputs = config.channel_hist[0]
config.num_gppn_hidden_channels = 2  # 50
config.num_gppn_iterations = 5  # 10
config.gppn_kernelsize = 7  # 9
config.num_acc_bins_in_gppn = 1  # this is a latent number
config.gppn_out_channels = config.num_steering_bins * config.num_acc_bins_in_gppn
# resnet params
config.Num_resnet_layers = 3
config.num_resnet_output = 256 * 4
config.resblock_in_layers = [1, 1, 1, 2]
config.resnet_width = 32
# heads params
config.head_mode = "categorical"  # "categorical", "hybrid", "mdn"
config.use_vel_head = False
config.sigma_smoothing = 0.03
config.num_guassians_in_heads = 5
''' NN settings '''


''' Loss settings '''
config.reward_mode = 'data'
config.gamma = 1.0
config.val_scale = 1.0  # 10.0
config.ang_scale = 1.0  # 10.0
config.acc_scale = 1.0
config.vel_scale = 1.0
config.lane_scale = 1.0
''' Loss settings '''


''' Debugging settings '''
# toggle data visualization
config.visualize_train_data = True
config.visualize_val_data = False
config.visualize_inter_data = False
config.visualize_raw_data = False
# toggle debug printing
config.print_preds = False
config.draw_prediction_records = False
''' Debugging settings '''


''' Controller settings '''
config.control_freq = 10.0  # 10.0
config.acc_slow_down = 1
config.car_goal = [-1, -1]  # dummy goal
''' Controller settings '''
