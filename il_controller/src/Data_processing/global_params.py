from argparse import Namespace
import torch

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
config.steering_resolution = 1.0/7.0  # no unit
config.max_steering = 1.0  # no unit
config.num_steering_bins = 2 * int(round(config.max_steering / config.steering_resolution))

config.num_acc_bins = 3
config.acc_resolution = 2.0 / config.num_acc_bins
config.max_acc = 1.5  # in degrees

config.num_vel_bins = 6
config.vel_max = 1.5
''' Action model settings '''


''' Sampling settings '''
config.num_samples_per_traj = 20
config.num_agents_in_map = 20
config.buffer_mode = 'full'  # full or replay
config.sample_mode = 'random'  # hierarchical or random
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
config.label_angular = 1
config.label_cmdvel = 2
''' Data label settings '''


''' Channel codes '''
config.num_hist_channels = 4
# channel 0-3, exo history 1-4
config.channel_map = []
for i in range(config.num_hist_channels):
    config.channel_map.append(i)
# channel 4, lanes
config.channel_lane = config.num_hist_channels
# channel 5, goal path
config.channel_goal = config.num_hist_channels + 1
# channel 6-9, ego history 1-4
config.use_hist_channels = False
config.channel_hist = []
for i in range(config.num_hist_channels):
    config.channel_hist.append(i + config.channel_goal + 1)
# total number of channels
if config.use_hist_channels:
    config.total_num_channels = 2 + 2 * config.num_hist_channels
else:
    config.total_num_channels = 2 + config.num_hist_channels
''' Channel codes '''


''' Training settings '''
# toggle data augmentation
config.augment_data = True
# toggle dropout
config.do_dropout = False
config.do_prob = 0.5
# L2 regularization factor
config.l2_reg_weight = 0  # 1e-3
# toggle batch norm
config.disable_bn_in_resnet = False
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
config.imsize = 32
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
config.resblock_in_layers = [2, 2, 1, 2]
config.resnet_width = 32
# heads params
config.head_mode = "categorical"  # "categorical", "hybrid", "mdn"
config.use_vel_head = False
config.sigma_smoothing = 0.03
config.num_guassians_in_heads = 5
''' NN settings '''


''' Loss settings '''
config.reward_mode = 'func'
config.gamma = 1.0
config.val_scale = 1.0  # 10.0
config.ang_scale = 2.0  # 10.0
''' Loss settings '''


''' Debugging settings '''
# toggle data visualization
config.visualize_train_data = True
config.visualize_val_data = False
config.visualize_inter_data = True
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
