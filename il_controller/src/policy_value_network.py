import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Components import GPPN, resnet_modified, mdn, ConvLSTM
import argparse
from Data_processing import global_params
from Components.nan_police import *
import sys

global_config = global_params.config
''' Input Tensor is of the form ==> BatchSize * Ped * map/goal/hist * Width * height
'''


def get_input_slicing():
    ped_start = min(global_config.channel_map)
    ped_end = max(global_config.channel_map) + 1
    gppn_end = max(ped_end, global_config.channel_goal + 1, global_config.channel_lane + 1)
    hist_start = min(global_config.channel_hist)
    hist_end = max(global_config.channel_hist) + 1
    return ped_start, ped_end, gppn_end, hist_start, hist_end


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)


class ActionHead(nn.Module):

    def __init__(self, inplanes=64, imsize=16, num_classes=global_config.num_steering_bins):
        super(ActionHead, self).__init__()
        outplanes = 4

        self.drop_o_2d = nn.Dropout2d(p=global_config.do_prob)
        self.conv = conv1x1(inplanes, outplanes, stride=1)
        if not global_config.disable_bn_in_resnet:
            self.bn = nn.BatchNorm2d(outplanes, track_running_stats=global_config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features=imsize * imsize * outplanes,
                            out_features=num_classes,
                            bias=True)

    def forward(self, x):
        if global_config.do_dropout:
            x = self.drop_o_2d(x)
        out = self.conv(x)
        if not global_config.disable_bn_in_resnet:
            out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class LargeActionHead(nn.Module):

    def __init__(self, inplanes=64, imsize=16, num_classes=global_config.num_steering_bins):
        super(LargeActionHead, self).__init__()
        outplanes = 4

        self.drop_o_2d = nn.Dropout2d(p=global_config.do_prob)
        self.conv = conv1x1(inplanes, outplanes, stride=1)
        if not global_config.disable_bn_in_resnet:
            self.bn = nn.BatchNorm2d(outplanes, track_running_stats=global_config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features=imsize * imsize * outplanes,
                            out_features=128,
                            bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=128,
                            out_features=num_classes,
                            bias=True)

    def forward(self, x):
        if global_config.do_dropout:
            x = self.drop_o_2d(x)
        out = self.conv(x)
        if not global_config.disable_bn_in_resnet:
            out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu1(out)
        out = self.fc1(out)
        return out


class ActionMdnHead(nn.Module):

    def __init__(self, inplanes=64, imsize=16, num_modes=global_config.num_guassians_in_heads):
        super(ActionMdnHead, self).__init__()

        outplanes = 4
        self.conv = conv1x1(inplanes, outplanes, stride=1)
        if not global_config.disable_bn_in_resnet:
            self.bn = nn.BatchNorm2d(outplanes, track_running_stats=global_config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.mdn = mdn.MDN(in_features=imsize * imsize * outplanes, out_features=1,
                           num_gaussians=num_modes)

    def forward(self, x):
        x = self.conv(x)
        if not global_config.disable_bn_in_resnet:
            x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        pi, sigma, mu = self.mdn(x)
        return pi, sigma, mu


class ValueHead(nn.Module):

    def __init__(self, inplanes=64, imsize=16):
        super(ValueHead, self).__init__()
        outplanes = 4
        self.drop_o_2d = nn.Dropout2d()
        self.conv = conv1x1(inplanes, outplanes, stride=1)
        if not global_config.disable_bn_in_resnet:
            self.bn = nn.BatchNorm2d(outplanes, track_running_stats=global_config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features=imsize * imsize * outplanes,
                            out_features=1,
                            bias=True)

    def forward(self, x):
        if global_config.do_dropout:
            x = self.drop_o_2d(x)
        out = self.conv(x)
        if not global_config.disable_bn_in_resnet:
            out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class PolicyValueNet(nn.Module):

    def __init__(self, config):
        super(PolicyValueNet, self).__init__()
        self.fc_modules_size = 0
        self.num_steering_bins = global_config.num_steering_bins
        self.num_vel_bins = global_config.num_vel_bins
        self.num_acc_bins = global_config.num_acc_bins

        self.car_gppn, self.car_lstm, self.map_lstm = None, None, None
        if global_config.vanilla_resnet:
            self.input_channels_resnet = global_config.total_num_channels
        else:
            self.car_gppn = GPPN.GPPN(config, l_i=global_config.num_gppn_inputs,
                                      out_planes=global_config.gppn_out_channels)
            self.gppn_size = get_module_size(self.car_gppn)
            if global_config.use_hist_channels:
                self.input_channels_resnet = global_config.gppn_out_channels + global_config.num_hist_channels
            else:
                self.input_channels_resnet = global_config.gppn_out_channels

        # 3 channels include the Value image and the 2 hist layers (idx 0 is value image)

        self.pre_resnet = None
        if not global_config.vanilla_resnet:
            self.pre_resnet = nn.Sequential(
                resnet_modified.BasicBlock(inplanes=self.input_channels_resnet, planes=self.input_channels_resnet),
                resnet_modified.BasicBlock(inplanes=self.input_channels_resnet, planes=self.input_channels_resnet),
            )

        self.resnet = resnet_modified.ResNetModified(block=resnet_modified.BasicBlock,
                                                     layers=global_config.resblock_in_layers,
                                                     ip_channels=self.input_channels_resnet)
        self.resnet_size = get_module_size(self.resnet)

        self.drop_o = nn.Dropout()
        self.drop_o_2d = nn.Dropout2d()

        self.value_head = ValueHead(inplanes=self.resnet.num_out_features)
        self.fc_modules_size += get_module_size(self.value_head)

        if global_config.head_mode == "mdn":
            self.acc_head = ActionMdnHead(inplanes=self.resnet.num_out_features)
            self.ang_head = ActionMdnHead(inplanes=self.resnet.num_out_features, num_modes=10)

            if global_config.use_vel_head:
                self.vel_head = ActionMdnHead(inplanes=self.resnet.num_out_features)
        elif global_config.head_mode == "hybrid":
            self.acc_head = ActionMdnHead(inplanes=self.resnet.num_out_features)
            if global_config.use_vel_head:
                self.vel_head = ActionMdnHead(inplanes=self.resnet.num_out_features)
            self.ang_head = ActionHead(inplanes=self.resnet.num_out_features,
                                       num_classes=self.num_steering_bins)
        else:
            self.acc_head = LargeActionHead(inplanes=self.resnet.num_out_features,
                                       num_classes=self.num_acc_bins)
            self.ang_head = ActionHead(inplanes=self.resnet.num_out_features,
                                       num_classes=self.num_steering_bins)
            if global_config.use_vel_head:
                self.vel_head = ActionHead(inplanes=self.resnet.num_out_features,
                                           num_classes=self.num_vel_bins)

        self.fc_modules_size += get_module_size(self.acc_head)
        self.fc_modules_size += get_module_size(self.ang_head)
        if global_config.use_vel_head:
            self.fc_modules_size += get_module_size(self.vel_head)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                # m.bias.data.normal_(0, 1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTMCell):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        param.data.zero_()
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=np.sqrt(2))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freezed_modules = []
        self.freeze_parameters()

        print_model_size(self)

        self.ped_start, self.ped_end, self.gppn_end, self.hist_start, self.hist_end \
            = get_input_slicing()

    def freeze_parameters(self):
        for module in self.freezed_modules:
            if module:
                for param in module.parameters():
                    param.requires_grad = False

    def print_grad_state(self):
        for module in self.freezed_modules:
            if module:
                for name, param in module.named_parameters():
                    print("{}:{}".format(name, param.requires_grad), end=',')

    def check_grad_state(self):
        for module in self.freezed_modules:
            if module:
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        print("{}:{}".format(name, param.requires_grad), end=',')

    def forward(self, X, config):
        # Input Tensor is of the form ==> batch_size * 1 * map/goal/hist * Width * height
        # switches batch and agent dim in order to iterate over the agent dim
        self.check_grad_state()

        batch_size = X.size(0)
        car_input = X.contiguous()

        reshape_car_input = car_input.view(batch_size, global_config.total_num_channels, config.imsize,
                                           config.imsize).contiguous()

        car_gppn_input = reshape_car_input[:, 0:self.gppn_end, :, :]  # with 4 ped
        # maps, lane, and goal channel

        car_hist_data = None
        if global_config.use_hist_channels:
            car_hist_data = reshape_car_input[:, self.hist_start:self.hist_end, :, :]  # with 4 hist channels

        # Container for the output value images
        # 3 channels include the Value image and the 2 hist layers (idx 0 is value image)
        # adding 1 extra dummy dimension because VIN produces single channel output

        if global_config.vanilla_resnet:
            car_values = car_gppn_input
        else:
            car_values = self.car_gppn(car_gppn_input)

        if global_config.use_hist_channels:
            car_features = torch.cat((car_values, car_hist_data), 1).contiguous()  # stack the channels
        else:
            car_features = car_values

        if global_config.vanilla_resnet:
            car_res_features = car_features
        else:
            car_res_features = self.pre_resnet(car_features)

        if global_config.do_dropout and self.input_channels_resnet > 100 \
                and global_config.head_mode == 'categorical':
            car_res_features = self.drop_o_2d(car_res_features)

        res_images = self.resnet(car_res_features)

        if global_config.do_dropout and global_config.head_mode == 'categorical':
            res_images = self.drop_o_2d(res_images)

        if global_config.head_mode == "mdn":
            value, acc_pi, acc_sigma, acc_mu, ang_pi, ang_mu, ang_sigma = None, None, None, None, None, None, None
            if global_config.fit_all or global_config.fit_val:
                value = self.value_head(res_images)
            if global_config.fit_all or global_config.fit_action or global_config.fit_acc:
                acc_pi, acc_sigma, acc_mu = self.acc_head(res_images)
            if global_config.fit_all or global_config.fit_action or global_config.fit_ang:
                ang_pi, ang_sigma, ang_mu = self.ang_head(res_images)
            if global_config.use_vel_head:
                vel_pi, vel_sigma, vel_mu = self.vel_head(res_images)
            else:
                vel_pi, vel_sigma, vel_mu = None, None, None
            return value, acc_pi, acc_mu, acc_sigma, ang_pi, ang_mu, ang_sigma, vel_pi, vel_mu, vel_sigma, \
                   car_values[0], res_images[0]
        elif global_config.head_mode == "hybrid":
            value, acc_pi, acc_sigma, acc_mu, ang = None, None, None, None, None
            if global_config.fit_all or global_config.fit_val:
                value = self.value_head(res_images)
            if global_config.fit_all or global_config.fit_action or global_config.fit_acc:
                acc_pi, acc_sigma, acc_mu = self.acc_head(res_images)
            if global_config.fit_all or global_config.fit_action or global_config.fit_ang:
                ang = self.ang_head(res_images)
            if global_config.use_vel_head:
                vel_pi, vel_sigma, vel_mu = self.vel_head(res_images)
            else:
                vel_pi, vel_sigma, vel_mu = None, None, None
            return value, acc_pi, acc_mu, acc_sigma, ang, vel_pi, vel_mu, vel_sigma, \
                   car_values[0], res_images[0]
        else:
            if global_config.use_vel_head:
                vel_out = self.vel_head(res_images)
            else:
                vel_out = None
            return self.value_head(res_images), self.acc_head(res_images), self.ang_head(res_images), vel_out, \
                   car_values[0], res_images[0]


def print_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    resnet_size = 0
    if model.resnet:
        resnet_size = get_module_size(model.resnet)

    if model.pre_resnet:
        resnet_size += get_module_size(model.pre_resnet)

    gppn_size = 0

    if model.car_gppn:
        gppn_size += get_module_size(model.car_gppn)

    head_size = 0
    if model.ang_head:
        head_size += get_module_size(model.ang_head)
    if model.acc_head:
        head_size += get_module_size(model.acc_head)
    if model.value_head:
        head_size += get_module_size(model.value_head)

    print("\n=========================== Model statistics ==========================")
    print("No. trainable parameters in model: %d" % params)
    print("No. parameters in heads: %d" % head_size)
    print("No. parameters in resnet: %d" % resnet_size)
    # if not global_config.vanilla_resnet:
    print("No. parameters in GPPN: %d" % gppn_size)
    print("=========================== Model statistics ==========================\n")


def get_module_size(module):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


if __name__ == '__main__':
    # Automatic swith of GPU mode if available
    # use_GPU = torch.cuda.is_available()
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        default=None, help='Path to train file')
    parser.add_argument('--val', type=str,
                        default=None, help='Path to val file')
    parser.add_argument('--imsize',
                        type=int,
                        default=32,
                        help='Size of image')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,
                        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--epochs',
                        type=int,
                        default=global_config.num_epochs,
                        help='Number of epochs to train')
    parser.add_argument('--start_epoch',
                        type=int,
                        default=0,
                        help='start epoch for training')
    parser.add_argument('--k',
                        type=int,
                        default=global_config.num_gppn_iterations,  # 10
                        help='Number of Value Iterations')
    parser.add_argument('--f',
                        type=int,
                        default=global_config.gppn_kernelsize,
                        help='Number of Value Iterations')
    parser.add_argument('--l_i',
                        type=int,
                        default=global_config.num_gppn_inputs,
                        help='Number of channels in input layer')
    parser.add_argument('--l_h',
                        type=int,
                        default=global_config.num_gppn_hidden_channels,  # 150
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q',
                        type=int,
                        default=10,
                        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--no_action',
                        type=int,
                        default=4,
                        help='Number of actions')
    parser.add_argument('--modelfile',
                        type=str,
                        default="drive_net_params",
                        help='Name of model file to be saved')
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Name of model file to be loaded for resuming training')
    parser.add_argument('--no_vin',
                        type=int,
                        default=int(global_config.vanilla_resnet),
                        help='Number of pedistrians')
    parser.add_argument('--fit',
                        type=str,
                        default='action',
                        help='Number of pedistrians')

    config = parser.parse_args()

    global_config.vanilla_resnet = config.no_vin

    debug_net = PolicyValueNet(config)

    print_model_size(debug_net)

    X = Variable(torch.randn([config.batch_size, 0 +
                              1, 9, config.imsize, config.imsize]))
    val, steer, ang, vel, car_values, res_image = debug_net.forward(X, config)
