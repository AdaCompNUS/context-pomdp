import torch
import socketio, eventlet
import eventlet.wsgi
from flask import Flask
import os
import sys
import rospy
from query_nn.srv import TensorData

import time

# sio = socketio.Server()
# app = Flask(__name__)


model_device_id = 0
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")


import sys

from collections import OrderedDict

home = os.path.expanduser('~')


nn_folder = os.path.join(home, "workspace/catkin_ws/src/il_controller/src/BTS_RL_NN/")
if not os.path.isdir(nn_folder):
    nn_folder = os.path.join(home, "catkin_ws/src/il_controller/src/BTS_RL_NN/")


test_model_name = os.path.join(home, "workspace/catkin_ws/src/il_controller/src/BTS_RL_NN/trained_models/hybrid_unicorn.pth")
if not os.path.isfile(test_model_name):
    test_model_name = os.path.join(home, "catkin_ws/src/il_controller/src/BTS_RL_NN/trained_models/hybrid_unicorn.pth")

sys.path.append(os.path.join(nn_folder, 'Data_processing'))
sys.path.append(nn_folder)


print(sys.path)

import numpy as np
import math

import matplotlib

matplotlib.use('Agg')
# from data_monitor import *
from train import forward_pass, load_settings_from_model
from train import parse_cmd_args, update_global_config

from Components import GPPN, resnet_modified, mdn

# from dataset import set_encoders
# from Components.mdn import sample_mdn, sample_mdn_ml
# from model_drive_net_modi_res import DriveNetModiRes
# from model_drive_net_0_ped import DriveNetZeroPed
from Data_processing import global_params
import torch
import torch.nn as nn
config = global_params.config
global_config = config

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def get_module_size(module):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

class ActionHead(nn.Module):

    def __init__(self, inplanes=64, imsize=16, num_classes=global_config.num_steering_bins):
        super(ActionHead, self).__init__()
        outplanes = 4

        self.drop_o_2d = nn.Dropout2d()
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


class ActionMdnHead(nn.Module):

    def __init__(self, inplanes=64, imsize=16):
        super(ActionMdnHead, self).__init__()

        outplanes = 4
        self.conv = conv1x1(inplanes, outplanes, stride=1)
        if not global_config.disable_bn_in_resnet:
            self.bn = nn.BatchNorm2d(outplanes, track_running_stats=global_config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.mdn = mdn.MDN(in_features=imsize * imsize * outplanes, out_features=1,
                           num_gaussians=global_config.num_guassians_in_heads)

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
        # outplanes = 2
        self.drop_o_2d = nn.Dropout2d()
        self.conv = conv1x1(inplanes, outplanes, stride=1)
        if not global_config.disable_bn_in_resnet:
            self.bn = nn.BatchNorm2d(outplanes, track_running_stats=global_config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(in_features=imsize * imsize * outplanes,
        #                      out_features=256,
        #                      bias=True)
        # self.fc2 = nn.Linear(in_features=256,
        #                      out_features=1,
        #                      bias=True)
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
        # out = self.fc1(out)
        # out = self.fc2(out)
        out = self.fc(out)
        return out



class DriveNetModiRes(nn.Module):

    def __init__(self, config):
        super(DriveNetModiRes, self).__init__()
        self.fc_modules_size = 0
        self.num_steering_bins = global_config.num_steering_bins
        self.num_vel_bins = global_config.num_vel_bins
        self.num_acc_bins = global_config.num_acc_bins

        self.car_VIN = None
        if not global_config.vanilla_resnet:
            self.car_VIN = GPPN.GPPN(config)
            self.output_channels_VIN = self.car_VIN.output_channels + global_config.num_hist_channels
        else:
            self.output_channels_VIN = global_config.channel_hist1 + global_config.num_hist_channels
            global_config.vin_out_channels = global_config.channel_hist1

        # 3 channels include the Value image and the 2 hist layers (idx 0 is value image)
        self.no_input_resnet = self.output_channels_VIN * (config.no_ped + config.no_car)

        self.resnet = resnet_modified.ResNetModified(block=resnet_modified.BasicBlock,
                                                     layers=global_config.resblock_in_layers,
                                                     ip_channels=self.no_input_resnet)

        self.num_resnet_out_features = self.resnet.num_outfeatures

        self.car_resnet = None
        if not global_config.vanilla_resnet:
            self.car_resnet = nn.Sequential(
                resnet_modified.BasicBlock(self.output_channels_VIN, self.output_channels_VIN),
                resnet_modified.BasicBlock(self.output_channels_VIN, self.output_channels_VIN),
            )

        self.drop_o = nn.Dropout()
        self.drop_o_2d = nn.Dropout2d()

        self.value_head = ValueHead(inplanes=self.num_resnet_out_features)

        if global_config.head_mode == "mdn":
            self.acc_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
            self.ang_head = ActionMdnHead(inplanes=self.num_resnet_out_features)

            if global_config.use_vel_head:
                self.vel_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
        elif global_config.head_mode == "hybrid":
            self.acc_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
            if global_config.use_vel_head:
                self.vel_head = ActionMdnHead(inplanes=self.num_resnet_out_features)
            self.ang_head = ActionHead(inplanes=self.num_resnet_out_features,
                                       num_classes=self.num_steering_bins)
        else:
            self.acc_head = ActionHead(inplanes=self.num_resnet_out_features,
                                       num_classes=self.num_acc_bins)
            self.ang_head = ActionHead(inplanes=self.num_resnet_out_features,
                                       num_classes=self.num_steering_bins)
            if global_config.use_vel_head:
                self.vel_head = ActionHead(inplanes=self.num_resnet_out_features,
                                           num_classes=self.num_vel_bins)

        # self.fc_modules_size += get_module_size(self.value_head)
        self.fc_modules_size += get_module_size(self.acc_head)
        self.fc_modules_size += get_module_size(self.ang_head)
        if global_config.use_vel_head:
            self.fc_modules_size += get_module_size(self.vel_head)

        self.resnet_size = get_module_size(self.resnet)

        if not global_config.vanilla_resnet:
            self.gppn_size = get_module_size(self.car_VIN)

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

        # # init categorical heads weights to be near zero
        #
        # if global_config.head_mode == 'hybrid' or global_config.head_mode == 'categorical':
        #     for m in self.ang_head.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.xavier_uniform_(m.weight, gain=np.sqrt(0.00002))
        #             m.bias.data.normal_(0, 0.0001)
        # if global_config.head_mode == 'categorical':
        #     for m in self.acc_head.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.xavier_uniform_(m.weight, gain=np.sqrt(0.00002))
        #             m.bias.data.normal_(0, 0.0001)

    def forward(self, X, config):
        # Input Tensor is of the form ==> batch_size * Agent * map/goal/hist * Width * height
        # switches batch and agent dim in order to iterate over the agent dim
        # reshape_X = X.permute(1, 0, 2, 3, 4)
        batch_size = X.size(0)
        num_agents = config.no_ped + config.no_car
        car_input = X[:, config.no_ped:num_agents, :, :, :].contiguous()

        reshape_car_input = car_input.view(batch_size * config.no_car, global_config.num_channels, config.imsize,
                                           config.imsize).contiguous()

        car_vin_input = reshape_car_input[:, 0:global_config.channel_hist1, :, :]  # with 4 ped maps and goal channel

        car_hist_data = reshape_car_input[:, global_config.channel_hist1:, :, :]  # with 4 hist channels

        # Container for the output value images
        # 3 channels include the Value image and the 2 hist layers (idx 0 is value image)
        # adding 1 extra dummy dimension because VIN produces single channel output

        if global_config.vanilla_resnet:
            car_values = car_vin_input
        else:
            car_values = self.car_VIN(car_vin_input, config)

        car_features = torch.cat((car_values, car_hist_data), 1).contiguous()  # stack the channels

        if global_config.vanilla_resnet:
            car_res_features = car_features
        else:
            car_res_features = self.car_resnet(car_features)

        car_res_features_recover = car_res_features.view(batch_size, config.no_car, self.output_channels_VIN,
                                                         config.imsize, config.imsize).contiguous()

        res_features = car_res_features_recover.contiguous()

        res_features_reshape = res_features.view(batch_size, num_agents * self.output_channels_VIN, config.imsize,
                                                 config.imsize).contiguous()

        if global_config.do_dropout and num_agents * self.output_channels_VIN > 100 \
                and global_config.head_mode == 'categorical':
            res_features_reshape = self.drop_o_2d(res_features_reshape)

        res_image = self.resnet(res_features_reshape)

        if global_config.do_dropout and global_config.head_mode == 'categorical':
            res_image = self.drop_o_2d(res_image)

        if global_config.head_mode == "mdn":
            # value = self.value_head(res_image)
            acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
            ang_pi, ang_sigma, ang_mu = self.ang_head(res_image)
            if global_config.use_vel_head:
                vel_pi, vel_sigma, vel_mu = self.vel_head(res_image)
            else:
                vel_pi, vel_sigma, vel_mu = None, None, None
            return None, acc_pi, acc_mu, acc_sigma, ang_pi, ang_mu, ang_sigma, vel_pi, vel_mu, vel_sigma, \
                   car_values[0], res_image[0]
        elif global_config.head_mode == "hybrid":
            # value = self.value_head(res_image)
            acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
            ang = self.ang_head(res_image)
            if global_config.use_vel_head:
                vel_pi, vel_sigma, vel_mu = self.vel_head(res_image)
            else:
                vel_pi, vel_sigma, vel_mu = None, None, None
            return None, acc_pi, acc_mu, acc_sigma, ang, vel_pi, vel_mu, vel_sigma, \
                   car_values[0], res_image[0]
        else:
            if global_config.use_vel_head:
                vel_out = self.vel_head(res_image)
            else:
                vel_out = None
            return None, self.acc_head(res_image), self.ang_head(res_image), vel_out, \
                   car_values[0], res_image[0]



def on_data_ready(data):
    print("Received data ")
    bs = data.batchsize
    mode = data.mode

    start = time.time()

    if mode == 'all':
        import pdb
        try:
            print("all mode")

            print("bs=", bs)

            print("view")
            pt_tensor_from_list = torch.FloatTensor(data.tensor).view(bs,9,32,32).unsqueeze(1).view(bs,1,9,32,32)

            print("cuda")
            input = pt_tensor_from_list.to(device)

            print("forward")
            _, acc_pi, acc_mu, acc_sigma, \
            ang, _, _ , _, _ , _= net.forward(input, cmd_args)

            print("check output")

            # value = value.cpu().view(-1).detach().numpy().tolist()
            acc_pi = acc_pi.cpu().view(-1).detach().numpy().tolist()
            acc_mu = acc_mu.cpu().view(-1).detach().numpy().tolist()
            acc_sigma = acc_sigma.cpu().view(-1).detach().numpy().tolist()
            ang = ang.cpu().view(-1).detach().numpy().tolist()

            value = np.zeros(len(acc_pi), dtype=float)

        except Exception as e:
            print(e)

            pdb.set_trace()
    else:
        print("only support all mode")

    print(mode+" model forward time: " + str(time.time()-start) +  's')

    return value, acc_pi, \
            acc_mu, acc_sigma, ang


def print_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("No. parameters in model: %d", params)


if __name__ == '__main__':
    model_mode = ''


    # try:
    #   drive_net = torch.jit.load(test_model_name).cuda(device)
    #   model_mode='jit'
    # except Exception as e:
    #   pass

    net = None

    if model_mode is '':
        # drive_net = torch.load(test_model_name).to(device)
        cmd_args = parse_cmd_args()

        update_global_config(cmd_args)

        config.track_running_stats = True
        config.head_mode = "hybrid"

        print("=> loading checkpoint '{}'".format(cmd_args.modelfile))
        checkpoint = torch.load(os.path.join(nn_folder, cmd_args.modelfile))

        try:
            # pass
            load_settings_from_model(checkpoint, global_config,
                                cmd_args)
        except Exception as e:
            print(e)

        net = DriveNetModiRes(cmd_args)

        print_model_size(net)
        net = nn.DataParallel(net, device_ids=[0]).to(device)  # device_ids= config.GPU_devices

        net.load_state_dict(checkpoint['state_dict'])
        print("=> model at epoch {}"
              .format(checkpoint['epoch']))

        model_mode='torch'

        from torch.autograd import Variable

        X = Variable(torch.randn([128, cmd_args.no_ped +
                          cmd_args.no_car, 9, cmd_args.imsize, cmd_args.imsize]))
        value, acc_pi, acc_mu, acc_sigma, \
        ang, _, _, _,_,_  = net.forward(X, cmd_args)


    print("drive_net model mode:", model_mode)

    rospy.init_node('nn_query_node')
    s = rospy.Service('query', TensorData, on_data_ready)
    print ("Ready to query nn.")
    rospy.spin()
