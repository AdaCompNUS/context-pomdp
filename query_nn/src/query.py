import numpy as np

import torch, socketio, eventlet, eventlet.wsgi
from flask import Flask
import os, sys, rospy
from msg_builder.srv import TensorData
import time

from collections import OrderedDict

model_device_id = 0
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")


home = os.path.expanduser('~')


nn_folder = os.path.join(home, "catkin_ws/src/il_controller/src/")
if not os.path.isdir(nn_folder):
    raise Exception('nn folder does not exist: {}'.format(nn_folder))

sys.path.append(os.path.join(nn_folder, 'Data_processing'))
sys.path.append(nn_folder)


import math

import matplotlib
matplotlib.use('Agg')

from Data_processing import global_params
import torch.nn as nn
config = global_params.config
global_config = config

def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)


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

            print("reshape data of length {}".format(len(data.tensor)))
            pt_tensor_from_list = torch.FloatTensor(data.tensor).view(bs,6,32,32).unsqueeze(1).view(bs,1,6,32,32)

            print("cuda")
            input = pt_tensor_from_list.to(device)

            print("forward")
            _, acc, \
            ang, _, _ , _ = net.forward(input, cmd_args)

            print("check output")

            acc = acc.cpu().view(-1).detach().numpy().tolist()
            ang = ang.cpu().view(-1).detach().numpy().tolist()

            value = np.zeros(len(acc), dtype=float)

        except Exception as e:
            error_handler(e)
            raise Exception('model query error')
    else:
        print("only support all mode")

    print(mode+" model forward time: " + str(time.time()-start) +  's')

    return value, acc, ang


def print_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("No. parameters in model: %d", params)


if __name__ == '__main__':
    print("main entry")

    print('5.2')
    import train

    print('5.1')
    import policy_value_network

    model_mode = ''

    net = None

    if model_mode is '':
        cmd_args = train.parse_cmd_args()

        train.update_global_config(cmd_args)
        print("configuration done")

        config.track_running_stats = True
        # config.head_mode = "hybrid"

        print("=> loading checkpoint '{}'".format(cmd_args.modelfile))
        checkpoint = torch.load(os.path.join(nn_folder, cmd_args.modelfile))

        try:
            # pass
            train.load_settings_from_model(checkpoint, global_config,
                                cmd_args)
        except Exception as e:
            print(e)

        net = policy_value_network.PolicyValueNet(cmd_args)

        print_model_size(net)
        net = nn.DataParallel(net, device_ids=[0]).to(device)  # device_ids= config.GPU_devices

        net.load_state_dict(checkpoint['state_dict'])
        print("=> model at epoch {}"
              .format(checkpoint['epoch']))

        model_mode='torch'

        from torch.autograd import Variable

        X = Variable(torch.randn([128, 0 +
                          1, 6, cmd_args.imsize, cmd_args.imsize]))
        value, acc, \
        ang, _, _, _  = net.forward(X, cmd_args)


    print("drive_net model mode:", model_mode)

    rospy.init_node('nn_query_node')
    s = rospy.Service('query', TensorData, on_data_ready)
    print ("Ready to query nn.")
    rospy.spin()
