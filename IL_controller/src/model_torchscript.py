import numpy as np
import torch
import torch.nn as nn
from PyTorch_to_C import GPPN, resnet_modified, mdn, global_params
# from Data_processing import global_params
import argparse

# import global_params


PYTORCH_JIT = 1

global_config = global_params.config
''' Input Tensor is of the form ==> BatchSize * Ped * map/goal/hist * Width * height
'''

imsize_eg_input_32 = 32  # example input
imsize_eg_input_16 = 16  # example input

head_mode_dict = {}
head_mode_dict['mdn'] = 0
head_mode_dict['hybrid'] = 1
head_mode_dict['catergorical'] = 2
head_mode = head_mode_dict[global_config.head_mode]

no_ped = global_config.num_peds_in_NN  # 0
no_car = 1  # default value in train.py
resblock_in_layers = global_config.resblock_in_layers  # [2, 2, 2, 2]
disable_bn_in_resnet = global_config.disable_bn_in_resnet  # False
num_steering_bins = global_config.num_steering_bins  # 14
imsize = 32  # default

GPPN_l_i = global_config.default_li  # 5
GPPN_l_h = 100
GPPN_k = global_config.num_iterations  # 5
GPPN_f = global_config.GPPN_kernelsize  # 7
output_channels_VIN = 32
vanilla_resnet = global_config.vanilla_resnet  # False

resnet_width = global_config.resnet_width  # 32


def set_globals():
    global GPPN_l_i, GPPN_l_h, GPPN_k, GPPN_f, output_channels_VIN, vanilla_resnet, resnet_width
    GPPN_l_i = global_config.default_li  # 5
    GPPN_l_h = global_config.default_lh  # 50
    GPPN_k = global_config.num_iterations  # 5
    GPPN_f = global_config.GPPN_kernelsize  # 7
    output_channels_VIN = global_config.vin_out_channels
    if global_config.vanilla_resnet:
        output_channels_VIN = global_config.channel_hist1 + global_config.num_hist_channels  # 5 + 4 = 9
    else:
        output_channels_VIN = global_config.vin_out_channels + global_config.num_hist_channels

    vanilla_resnet = global_config.vanilla_resnet  # False
    resnet_width = global_config.resnet_width  # 32


class ActionHead(torch.jit.ScriptModule):

    def __init__(self, inplanes=64, imsize=16, num_classes=global_config.num_steering_bins):
        super(ActionHead, self).__init__()
        outplanes = 4

        # self.drop_o_2d = torch.jit.trace(nn.Dropout2d(),
        #     torch.randn(1, inplanes, imsize_eg_input_16, imsize_eg_input_16))
        self.conv = torch.jit.trace(nn.Conv2d(inplanes, outplanes, kernel_size=1,
                                              stride=1, bias=False),
                                    torch.randn(1, inplanes, imsize_eg_input_16, imsize_eg_input_16))

        # if self.disable_bn_in_resnet == 0:
        self.bn = torch.jit.trace(nn.BatchNorm2d(outplanes,
                                                 track_running_stats=global_config.track_running_stats),
                                  torch.randn(1, outplanes, imsize_eg_input_16, imsize_eg_input_16))
        self.relu = torch.jit.trace(nn.ReLU(inplace=True),
                                    torch.randn(1, outplanes, imsize_eg_input_16, imsize_eg_input_16))
        self.fc = torch.jit.trace(nn.Linear(in_features=imsize * imsize * outplanes,
                                            out_features=num_classes,
                                            bias=True),
                                  torch.randn(1, imsize * imsize * outplanes))

    @torch.jit.script_method
    def forward(self, x):
        # x = nn.Dropout2d(x)
        out = self.conv(x)
        # if self.disable_bn_in_resnet == 0:
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ActionMdnHead(torch.jit.ScriptModule):

    def __init__(self, inplanes=64, imsize=16):
        super(ActionMdnHead, self).__init__()

        outplanes = 4
        self.conv = torch.jit.trace(nn.Conv2d(inplanes, outplanes, stride=1,
                                              kernel_size=1, bias=False),
                                    torch.randn(1, inplanes, imsize_eg_input_16, imsize_eg_input_16))

        # if self.disable_bn_in_resnet == 0:
        self.bn = torch.jit.trace(nn.BatchNorm2d(outplanes,
                                                 track_running_stats=global_config.track_running_stats),
                                  torch.randn(1, outplanes, imsize_eg_input_16, imsize_eg_input_16))
        # self.relu = nn.ReLU(inplace=True)
        self.relu = torch.jit.trace(nn.ReLU(inplace=True),
                                    torch.randn(1, outplanes, imsize_eg_input_16, imsize_eg_input_16))

        self.mdn = mdn.MDN(in_features=imsize * imsize * outplanes, out_features=1,
                           num_gaussians=global_config.num_guassians_in_heads)

    @torch.jit.script_method
    def forward(self, x):
        x = self.conv(x)
        # if self.disable_bn_in_resnet == 0:
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        pi, sigma, mu = self.mdn(x)
        return pi, sigma, mu


class ValueHead(torch.jit.ScriptModule):

    def __init__(self, inplanes=64, imsize=16):
        super(ValueHead, self).__init__()
        outplanes = 4

        # self.drop_o_2d = torch.jit.trace(nn.Dropout2d(), 
        #     torch.randn(1, inplanes, imsize_eg_input_16, imsize_eg_input_16))

        self.conv = torch.jit.trace(nn.Conv2d(inplanes, outplanes, kernel_size=1,
                                              stride=1, bias=False),
                                    torch.randn(1, inplanes, imsize_eg_input_16, imsize_eg_input_16))
        # if self.disable_bn_in_resnet == 0:
        self.bn = torch.jit.trace(nn.BatchNorm2d(outplanes,
                                                 track_running_stats=global_config.track_running_stats),
                                  torch.randn(1, outplanes, imsize_eg_input_16, imsize_eg_input_16))
        # self.relu = nn.ReLU(inplace=True)
        self.relu = torch.jit.trace(nn.ReLU(inplace=True),
                                    torch.randn(1, outplanes, imsize_eg_input_16, imsize_eg_input_16))

        self.fc = torch.jit.trace(nn.Linear(in_features=imsize * imsize * outplanes,
                                            out_features=1, bias=True), torch.randn(1, imsize * imsize * outplanes))

    @torch.jit.script_method
    def forward(self, x):
        # x = nn.Dropout2d(x)
        out = self.conv(x)
        # if self.disable_bn_in_resnet == 0:
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        out = self.fc(out)
        return out


class DriveNetModiRes(torch.jit.ScriptModule):
    __constants__ = ['fc_modules_size', 'num_steering_bins', 'num_vel_bins',
                     'num_acc_bins', 'output_channels_VIN', 'no_input_resnet',
                     'num_resnet_out_features', 'no_car', 'no_ped', 'imsize',
                     'num_channels', 'channel_hist1', 'num_hist_channels',
                     'resblock_in_layers', 'batchsize']

    def __init__(self, batchsize):
        super(DriveNetModiRes, self).__init__()
        self.fc_modules_size = 0
        self.num_steering_bins = num_steering_bins
        self.num_vel_bins = global_config.num_vel_bins
        self.num_acc_bins = global_config.num_acc_bins
        self.no_ped = no_ped
        self.no_car = no_car
        self.imsize = imsize
        self.head_mode = head_mode
        self.num_channels = global_config.num_channels
        self.channel_hist1 = global_config.channel_hist1
        self.num_hist_channels = global_config.num_hist_channels
        self.resblock_in_layers = resblock_in_layers
        self.batchsize = batchsize

        # if self.vanilla_resnet == 0:
        self.car_VIN = GPPN.GPPN(l_i=GPPN_l_i, l_h=GPPN_l_h, k=GPPN_k, f=GPPN_f)
        self.output_channels_VIN = output_channels_VIN # 5 + 4
        # else:
        # self.output_channels_VIN = global_config.channel_hist1 + global_config.num_hist_channels

        # 3 channels include the Value image and the 2 hist layers (idx 0 is value image)
        self.no_input_resnet = self.output_channels_VIN * (self.no_ped + self.no_car)

        self.resnet = resnet_modified.ResNetModified(layers=self.resblock_in_layers,
                                                     ip_channels=self.no_input_resnet,
                                                     resnet_width_local=resnet_width)

        self.num_resnet_out_features = resnet_width

        # TODO: =================
        # if self.vanilla_resnet == 0:
        self.car_resnet = torch.jit.trace(nn.Sequential(
            resnet_modified.BasicBlock(self.output_channels_VIN, self.output_channels_VIN),
            resnet_modified.BasicBlock(self.output_channels_VIN, self.output_channels_VIN)),
            torch.randn(1, self.output_channels_VIN, imsize_eg_input_32, imsize_eg_input_32))

        self.value_head = ValueHead(inplanes=self.num_resnet_out_features)

        # TODO: =================
        # if self.head_mode == 0:
        #     self.acc_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
        #     self.ang_head = ActionMdnHead(inplanes=self.num_resnet_out_features)

        #     if self.use_vel_head != 0:
        #         self.vel_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
        # elif self.head_mode == 1:
        self.acc_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
        #     if self.use_vel_head != 0:
        #         self.vel_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
        self.ang_head = ActionHead(inplanes=self.num_resnet_out_features,
                                   num_classes=self.num_steering_bins)
        # else:
        # self.acc_head = ActionHead(inplanes=self.num_resnet_out_features,
        #                                 num_classes=self.num_acc_bins)
        # self.ang_head = ActionHead(inplanes=self.num_resnet_out_features,
        #                                 num_classes=self.num_steering_bins)
        # if self.use_vel_head != 0:
        #     self.vel_head = ActionHead(inplanes=self.num_resnet_out_features,
        #     num_classes=self.num_vel_bins)

    @torch.jit.script_method
    def forward(self, X):
        # def forward(self, X, batch_size = X.size(0), config):
        # Input Tensor is of the form ==> batch_size * Agent * map/goal/hist * Width * height
        # switches batch and agent dim in order to iterate over the agent dim
        # reshape_X = X.permute(1, 0, 2, 3, 4)

        num_agents = self.no_ped + self.no_car
        # car_input = X[:, self.no_ped:num_agents, :, :, :].contiguous()
        car_input = X.contiguous()

        # reshape_car_input = car_input.view(self.batchsize * self.no_car, self.num_channels, self.imsize,
        # self.imsize).contiguous()
        reshape_car_input = car_input

        car_vin_input = reshape_car_input[:, 0:self.channel_hist1, :, :]  # with 4 ped maps and goal channel

        car_hist_data = reshape_car_input[:, self.channel_hist1:, :, :]  # with 4 hist channels

        # TODO: ================
        # if self.vanilla_resnet != 0:
        #     car_values = car_vin_input
        # else:
        car_values = self.car_VIN(X=car_vin_input)

        car_features = torch.cat((car_values, car_hist_data), 1).contiguous()  # stack the channels

        # TODO: ================
        # if self.vanilla_resnet != 0:
        #     car_res_features = car_features
        # else:
        car_res_features = self.car_resnet(car_features)

        # car_res_features_recover = car_res_features.view(self.batchsize, self.no_car, self.output_channels_VIN,
        # self.imsize, self.imsize).contiguous()

        car_res_features_recover = car_res_features.contiguous()

        res_features = car_res_features_recover.contiguous()

        # res_features_reshape = res_features.view(self.batchsize, num_agents * self.output_channels_VIN, self.imsize,
        #                                          self.imsize).contiguous()
        res_features_reshape = res_features

        res_image = self.resnet(res_features_reshape)

        # TODO: ================
        # if self.head_mode == 0:
        #     value = self.value_head(res_image)
        #     acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
        #     ang_pi, ang_sigma, ang_mu = self.acc_head(res_image)
        #     if self.use_vel_head != 0:
        #         vel_pi, vel_sigma, vel_mu = self.vel_head(res_image)
        #     else:
        #         vel_pi, vel_sigma, vel_mu = None, None, None
        #     return value, acc_pi, acc_mu, acc_sigma, ang_pi, ang_mu, ang_sigma, vel_pi, vel_mu, vel_sigma, \
        #            car_values[0], res_image[0]
        # elif self.head_mode == 1:
        value = self.value_head(res_image)
        acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
        ang = self.ang_head(res_image)
        #     if global_config.use_vel_head:
        #         vel_pi, vel_sigma, vel_mu = self.vel_head(res_image)
        #     else:
        vel_pi, vel_sigma, vel_mu = None, None, None
        return value, acc_pi, acc_mu, acc_sigma, ang, \
               car_values[0], res_image[0]
        # else:
        #     if self.use_vel_head != 0:
        #         vel_out = self.vel_head(res_image)
        #     else:
        # vel_out = torch.Tensor([0])
        # return self.value_head(res_image), self.acc_head(res_image), self.ang_head(res_image), self.ang_head(res_image), \
        # car_values[0], res_image[0]
        # return self.value_head(res_image)


class DriveNetModiResMdn(torch.jit.ScriptModule):
    __constants__ = ['fc_modules_size', 'num_steering_bins', 'num_vel_bins',
                     'num_acc_bins', 'output_channels_VIN', 'no_input_resnet',
                     'num_resnet_out_features', 'no_car', 'no_ped', 'imsize',
                     'num_channels', 'channel_hist1', 'num_hist_channels',
                     'resblock_in_layers', 'batchsize']

    def __init__(self, batchsize):
        super(DriveNetModiResMdn, self).__init__()
        self.fc_modules_size = 0
        self.num_steering_bins = num_steering_bins
        self.num_vel_bins = global_config.num_vel_bins
        self.num_acc_bins = global_config.num_acc_bins
        self.no_ped = no_ped
        self.no_car = no_car
        self.imsize = imsize
        self.head_mode = head_mode
        self.num_channels = global_config.num_channels
        self.channel_hist1 = global_config.channel_hist1
        self.num_hist_channels = global_config.num_hist_channels
        self.resblock_in_layers = resblock_in_layers
        self.batchsize = batchsize

        self.car_VIN = GPPN.GPPN(l_i=GPPN_l_i, l_h=GPPN_l_h, k=GPPN_k, f=GPPN_f)
        self.output_channels_VIN = output_channels_VIN # 5 + 4

        # 3 channels include the Value image and the 2 hist layers (idx 0 is value image)
        self.no_input_resnet = self.output_channels_VIN * (self.no_ped + self.no_car)

        self.resnet = resnet_modified.ResNetModified(layers=self.resblock_in_layers,
                                                     ip_channels=self.no_input_resnet,
                                                     resnet_width_local=resnet_width)

        self.num_resnet_out_features = resnet_width

        # TODO: =================
        # if self.vanilla_resnet == 0:
        self.car_resnet = torch.jit.trace(nn.Sequential(
            resnet_modified.BasicBlock(self.output_channels_VIN, self.output_channels_VIN),
            resnet_modified.BasicBlock(self.output_channels_VIN, self.output_channels_VIN)),
            torch.randn(1, self.output_channels_VIN, imsize_eg_input_32, imsize_eg_input_32))

        self.value_head = ValueHead(inplanes=self.num_resnet_out_features)

        # TODO: =================
        self.acc_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
        self.ang_head = ActionMdnHead(inplanes=self.num_resnet_out_features)

    @torch.jit.script_method
    def forward(self, X):
        # def forward(self, X, batch_size = X.size(0), config):
        # Input Tensor is of the form ==> batch_size * Agent * map/goal/hist * Width * height
        # switches batch and agent dim in order to iterate over the agent dim
        # reshape_X = X.permute(1, 0, 2, 3, 4)

        car_input = X.contiguous()

        reshape_car_input = car_input

        car_vin_input = reshape_car_input[:, 0:self.channel_hist1, :, :]  # with 4 ped maps and goal channel

        car_hist_data = reshape_car_input[:, self.channel_hist1:, :, :]  # with 4 hist channels

        car_values = self.car_VIN(X=car_vin_input)

        car_features = torch.cat((car_values, car_hist_data), 1).contiguous()  # stack the channels

        car_res_features = self.car_resnet(car_features)

        car_res_features_recover = car_res_features.contiguous()

        res_features = car_res_features_recover.contiguous()

        res_features_reshape = res_features

        res_image = self.resnet(res_features_reshape)

        value = self.value_head(res_image)
        acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
        ang_pi, ang_sigma, ang_mu = self.ang_head(res_image)

        return value, acc_pi, acc_mu, acc_sigma, ang_pi, ang_mu, ang_sigma, \
               car_values[0], res_image[0]


class DriveNetVal(torch.jit.ScriptModule):
    __constants__ = ['fc_modules_size', 'num_steering_bins', 'num_vel_bins',
                     'num_acc_bins', 'output_channels_VIN', 'no_input_resnet',
                     'num_resnet_out_features', 'no_car', 'no_ped', 'imsize',
                     'num_channels', 'channel_hist1', 'num_hist_channels',
                     'resblock_in_layers', 'batchsize']

    def __init__(self, batchsize):
        super(DriveNetVal, self).__init__()
        self.fc_modules_size = 0
        self.num_steering_bins = num_steering_bins
        self.num_vel_bins = global_config.num_vel_bins
        self.num_acc_bins = global_config.num_acc_bins
        self.no_ped = no_ped
        self.no_car = no_car
        self.imsize = imsize
        self.head_mode = head_mode
        self.num_channels = global_config.num_channels
        self.channel_hist1 = global_config.channel_hist1
        self.num_hist_channels = global_config.num_hist_channels
        self.resblock_in_layers = resblock_in_layers
        self.batchsize = batchsize

        # if self.vanilla_resnet == 0:
        # self.car_VIN = GPPN.GPPN(l_i=GPPN_l_i, l_h=GPPN_l_h, k=GPPN_k, f=GPPN_f)
        self.output_channels_VIN = output_channels_VIN  # 5 + 4
        # else:
        # self.output_channels_VIN = global_config.channel_hist1 + global_config.num_hist_channels

        # 3 channels include the Value image and the 2 hist layers (idx 0 is value image)
        self.no_input_resnet = self.output_channels_VIN * (self.no_ped + self.no_car)

        self.resnet = resnet_modified.ResNet_2L(layers=self.resblock_in_layers,
                                                ip_channels=self.no_input_resnet,
                                                resnet_width_local=resnet_width)

        self.num_resnet_out_features = resnet_width

        self.value_head = ValueHead(inplanes=self.num_resnet_out_features)

        self.acc_head = ActionMdnHead(inplanes=self.num_resnet_out_features)

        if global_config.head_mode is "hybrid":
            self.ang_head = ActionHead(inplanes=self.num_resnet_out_features,
                                   num_classes=self.num_steering_bins)
        elif global_config.head_mode is "mdn":
            self.ang_head = ActionMdnHead(inplanes=self.num_resnet_out_features)

    @torch.jit.script_method
    def forward(self, X):
        # def forward(self, X, batch_size = X.size(0), config):
        # Input Tensor is of the form ==> batch_size * Agent * map/goal/hist * Width * height
        # switches batch and agent dim in order to iterate over the agent dim
        # reshape_X = X.permute(1, 0, 2, 3, 4)

        car_input = X.contiguous()

        res_image = self.resnet(car_input)

        value = self.value_head(res_image)

        return value


'''
for param_tensor in checkpoint['state_dict']:
    print(param_tensor, '\t', checkpoint['state_dict'][param_tensor].size())
'''
