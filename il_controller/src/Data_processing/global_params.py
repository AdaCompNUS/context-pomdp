from argparse import Namespace

import torch

config = Namespace()
# velocity labels used in data['vel_steer']
config.pycharm_mode = False

config.label_linear = 0
config.label_angular = 1
config.label_cmdvel = 2

# global car parameters
config.steering_resolution = 5.0  # in degrees
config.max_steering = 35.0  # in degrees
config.num_steering_bins = 2 * int(round(config.max_steering / config.steering_resolution))

config.num_acc_bins = 14
config.acc_resolution = 2.0 / config.num_acc_bins
config.max_acc = 1.5  # in degrees

config.num_vel_bins = 6
config.vel_max = 1.5

config.num_samples_per_traj = 20

# traning data structure labels for dim 0 (except the batch dim)
config.ped_mode = "new_res"
config.lstm_mode = 'gppn'  # 'gppn' or 'convlstm'

# traning data structure labels for dim 1 (except the batch dim)

config.channel_map = 0  # hist maps from 0 to 8
config.channel_goal = 4  # 9
config.channel_hist1 = 5  # 10  # current step
config.num_hist_channels = 4  # 9
config.num_channels = config.channel_hist1 + config.num_hist_channels

if config.ped_mode == "separate":
    config.num_agents_in_NN = 5

    config.channel_map = 0
    config.channel_goal = 1
    config.channel_hist1 = 2  # current step
    config.channel_hist2 = 3  # previous steps
    config.channel_hist3 = 4  # previous steps
    config.channel_hist4 = 5  # previous steps
    config.num_hist_channels = 4
    config.num_channels = 2 + config.num_hist_channels
elif config.ped_mode == "combined" or config.ped_mode == "new_res":
    config.num_agents_in_NN = 0
    config.num_agents_in_map = 20
    config.num_hist_channels = 4

    # config.channel_goal = 0
    #
    # config.channel_map = []
    # for i in range(config.num_hist_channels):
    #     config.channel_map.append(i+1)
    # map_last = config.channel_map[-1] + 1
    #
    # config.channel_hist = []
    # for i in range(config.num_hist_channels):
    #     config.channel_hist.append(i + map_last)

    # exo history 1-4, channel 0-3
    config.channel_map = []
    for i in range(config.num_hist_channels):
        config.channel_map.append(i)

    config.channel_lane = config.num_hist_channels       # channel 4
    config.channel_goal = config.num_hist_channels + 1   # channel 5

    # ego history 1-4, channel 6-9
    config.channel_hist = []
    for i in range(config.num_hist_channels):
        config.channel_hist.append(i + config.channel_goal + 1)

    config.num_channels = 2 + 2 * config.num_hist_channels

# size of downsampled map
config.imsize = 32

# label related settings
config.value_normalizer = 10.0

# cross validation setting
config.train_split = 1.0 # 0.7
config.val_split = 0.0 # 0.25
config.test_split = 0.00 # 0.05

# training settings

num_gpus = torch.cuda.device_count()

print("=> num of gpus = {}".format(num_gpus))

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


# toggle data augmentation
config.augment_data = True
# toggle dropout
config.do_dropout = False
config.do_prob = 0.5

# toggle L2 regularization
config.l2_reg_weight = 0 #1e-3  # 1e-3 #1e-4
# toggle batch norm
config.disable_bn_in_resnet = False
config.track_running_stats = True
config.label_smoothing = False
config.enable_scheduler = True

# set default num epochs
config.num_epochs = 50

config.vanilla_resnet = False
# VIN params
config.default_li = config.channel_hist1
config.default_lh = 50  # 16
config.num_iterations = 5  # 10
config.GPPN_kernelsize = 7  # 9
config.num_acc_bins_in_GPPN = 1  # this is a latent number
config.vin_out_channels = config.num_steering_bins * config.num_acc_bins_in_GPPN

# resnet params
config.Num_resnet_layers = 3
config.num_resnet_output = 256 * 4
config.resblock_in_layers = [1, 1, 1, 2]
config.resnet_width = 32

### debugging settings
# toggle data visualization
config.visualize_train_data = True
config.visualize_val_data = False
config.visualize_inter_data = False

# toggle debug printing
config.print_preds = False

config.fit_ang = False
config.fit_acc = False
config.fit_vel = False
config.fit_val = False
config.fit_action = False
config.fit_all = True

config.use_vel_head = False

config.num_guassians_in_heads = 5
config.head_mode = "hybrid"  # "categorical", "hybrid", "mdn"


## MDN settings
config.sigma_smoothing = 0.01


## DriveController settings
config.control_freq = 10.0  # 10.0
config.acc_slow_down = 1
config.draw_prediction_records = False
config.car_goal = [-1, -1]  # dummy goal


## replay buffer mode
config.buffer_mode = 'full'  # full or replay
config.sample_mode = 'random'  # hierarchical or random

config.train_set_path = ''

## reward
config.reward_mode = 'func'
config.gamma = 1.0
config.val_scale = 1.0 # 10.0
config.ang_scale = 2.0 # 10.0

#### not used any more
# adaptive learning rate setting 
config.use_adaptive_rl = False
config.adaptive_lr_stride = 100
config.adaptive_lr_decay = 0.2  # 0.1
#### not used any more
