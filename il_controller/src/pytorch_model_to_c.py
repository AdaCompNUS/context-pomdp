import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

compare_with_normal_version = False

from Components import GPPN, resnet_modified, mdn
import argparse
import model_torchscript
from Components.nan_police import *
from dataset import *
from PyTorch_to_C import global_params

global_config = global_params.config
''' Input Tensor is of the form ==> BatchSize * Ped * map/goal/hist * Width * height
'''


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class ActionHead(nn.Module):

    def __init__(self, inplanes=64, imsize=16, num_classes=global_config.num_steering_bins):
        super(ActionHead, self).__init__()
        outplanes = 4

        # self.drop_o_2d = nn.Dropout2d()
        self.conv = conv1x1(inplanes, outplanes, stride=1)
        if not global_config.disable_bn_in_resnet:
            self.bn = nn.BatchNorm2d(outplanes, track_running_stats=global_config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features=imsize * imsize * outplanes,
                            out_features=num_classes,
                            bias=True)

    def forward(self, x):

        # if global_config.do_dropout:
        #     x = self.drop_o_2d(x)

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
        # self.drop_o_2d = nn.Dropout2d()
        self.conv = conv1x1(inplanes, outplanes, stride=1)
        if not global_config.disable_bn_in_resnet:
            self.bn = nn.BatchNorm2d(outplanes, track_running_stats=global_config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.mdn = mdn.MDN(in_features=imsize * imsize * outplanes, out_features=1,
                           num_gaussians=global_config.num_guassians_in_heads)

    def forward(self, x):

        # if global_config.do_dropout:
        #     x = self.drop_o_2d(x)

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
        # self.drop_o_2d = nn.Dropout2d()
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

        # if global_config.do_dropout:
        #     x = self.drop_o_2d(x)

        out = self.conv(x)
        if not global_config.disable_bn_in_resnet:
            out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.fc2(out)
        out = self.fc(out)
        return out


class PolicyValueNet(nn.Module):

    def __init__(self, config):
        super(PolicyValueNet, self).__init__()
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

        self.fc_modules_size += get_module_size(self.value_head)
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

        # if global_config.do_dropout and num_agents * self.output_channels_VIN > 100 \
        #       and global_config.head_mode == 'categorical':
        #    # res_features_reshape = self.drop_o_2d(res_features_reshape)

        res_image = self.resnet(res_features_reshape)

        #if global_config.do_dropout and global_config.head_mode == 'categorical':
        #    # res_image = self.drop_o_2d(res_image)

        if global_config.head_mode == "mdn":
            value = self.value_head(res_image)
            acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
            ang_pi, ang_sigma, ang_mu = self.ang_head(res_image)
            if global_config.use_vel_head:
                vel_pi, vel_sigma, vel_mu = self.vel_head(res_image)
            else:
                vel_pi, vel_sigma, vel_mu = None, None, None
            return value, acc_pi, acc_mu, acc_sigma, ang_pi, ang_mu, ang_sigma, vel_pi, vel_mu, vel_sigma, \
                   car_values[0], res_image[0]
        elif global_config.head_mode == "hybrid":
            value = self.value_head(res_image)
            acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
            ang = self.ang_head(res_image)
            if global_config.use_vel_head:
                vel_pi, vel_sigma, vel_mu = self.vel_head(res_image)
            else:
                vel_pi, vel_sigma, vel_mu = None, None, None
            return value, acc_pi, acc_mu, acc_sigma, ang, vel_pi, vel_mu, vel_sigma, \
                   car_values[0], res_image[0]
        else:
            if global_config.use_vel_head:
                vel_out = self.vel_head(res_image)
            else:
                vel_out = None
            return self.value_head(res_image), self.acc_head(res_image), self.ang_head(res_image), vel_out, \
                   car_values[0], res_image[0]


def print_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("No. parameters in model: %d" % params)
    print("No. parameters in heads: %d" % model.fc_modules_size)
    print("No. parameters in resnet: %d" % model.resnet_size)
    if not global_config.vanilla_resnet:
        print("No. parameters in GPPN: %d" % model.gppn_size)


def get_module_size(module):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


model_device_id = global_config.GPU_devices[0]
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")

from train import set_fit_mode_bools

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
                        default=global_config.num_iterations,  # 10
                        help='Number of Value Iterations')
    parser.add_argument('--f',
                        type=int,
                        default=global_config.GPPN_kernelsize,
                        help='Number of Value Iterations')
    parser.add_argument('--l_i',
                        type=int,
                        default=global_config.default_li,
                        help='Number of channels in input layer')
    parser.add_argument('--l_h',
                        type=int,
                        default=global_config.default_lh,  # 150
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q',
                        type=int,
                        default=10,
                        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--no_ped',
                        type=int,
                        default=0,  # global_config.num_peds_in_NN,
                        help='Number of pedistrians')
    parser.add_argument('--no_car',
                        type=int,
                        default=1,
                        help='Number of cars')
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
    parser.add_argument('--w', 
                        type = int, 
                        default = global_config.resnet_width, 
                        help = 'ResNet Width')
    parser.add_argument('--vinout', 
                        type = int, 
                        default = global_config.vin_out_channels, 
                        help = 'ResNet Width')
    parser.add_argument('--trained_model', 
                        type = str, 
                        default = '',
                        help = 'Saved trained model file')
    parser.add_argument('--saving_name',
                        type=str,
                        default="torchscript_version.pt",
                        help='Name of model file to be saved')

    cmd_args = parser.parse_args()

    global_config.num_peds_in_NN = cmd_args.no_ped
    global_config.vanilla_resnet = cmd_args.no_vin

    global_config.default_li = cmd_args.l_i
    global_config.default_lh = cmd_args.l_h
    global_config.GPPN_kernelsize = cmd_args.f
    global_config.resnet_width = cmd_args.w
    global_config.vin_out_channels = cmd_args.vinout

    set_fit_mode_bools(cmd_args)

    global_config.Num_resnet_layers = 2

    model_torchscript.set_globals()

    print("=> num_peds_in_NN ", global_config.num_peds_in_NN)
    print("=> vanilla_resnet ", global_config.vanilla_resnet)
    print("=> default_li ", global_config.default_li)
    print("=> default_lh ", global_config.default_lh)
    print("=> GPPN_kernelsize ", global_config.GPPN_kernelsize)
    print("=> resnet_width ", global_config.resnet_width)
    print("=> vin_out_channels ", global_config.vin_out_channels)
    print("=> Num_resnet_layers ", global_config.Num_resnet_layers)
    print("=> fit ", cmd_args.fit)

    # print(global_config)

    debug_net = None
    if compare_with_normal_version:
        debug_net = PolicyValueNet(cmd_args)

    # print_model_size(debug_net)

    TorchScript_Module = None
    if cmd_args.fit == 'all':
        if global_config.head_mode is "hybrid":
            TorchScript_Module = model_torchscript.PolicyValueNet(cmd_args.batch_size)
        elif global_config.head_mode is "mdn":
            TorchScript_Module = model_torchscript.PolicyValueNetMdn(cmd_args.batch_size)

    if cmd_args.fit == 'val':
        TorchScript_Module = model_torchscript.DriveNetVal(cmd_args.batch_size)

    # checkpoint = torch.load('drive_net_params_2018_10_23_11_05_57_t_train_v_val_vanilla_False_mode_new_res_k_5_lh_100_nped_0_vin_o_28_nres_3_w_64_bn_True_layers_2222_lr_0.0001_ada_True_l2_0.001_bs_128_aug_True_do_False_bn_stat_False_sm_labels_False.pth')
    # checkpoint = torch.load('drive_net_params_2018_12_05_06_55_05_t_train_v_val_vanilla_False_mode_new_res_k_5_lh_100_nped_0_vin_o_28_nres_3_w_64_bn_True_layers_1112_lr_0.0001_ada_True_l2_0.001_bs_128_aug_True_do_True_bn_stat_False_sm_labels_False.pth')
    checkpoint = torch.load(cmd_args.trained_model)

    print("check point loaded")

    if compare_with_normal_version:
        iter1 = iter(debug_net.state_dict())
        iter2 = iter(TorchScript_Module.state_dict())
        iter3 = iter(checkpoint['state_dict'])

        print('\nOriginal Net Layers\nTorchScript Net Layers\nLoaded Net Layers\n')
        for param_tensor in debug_net.state_dict():
            print(next(iter1) + '\n' + next(iter2) + '\n' + next(iter3))
            print(debug_net.state_dict()[param_tensor].size())
            print(TorchScript_Module.state_dict()[param_tensor].size())
            print(checkpoint['state_dict']['module.' + param_tensor].size())
    
    dict_for_params_match = {}
    for param_tensor in checkpoint['state_dict'].keys():
        dict_for_params_match[param_tensor[7:]] = checkpoint['state_dict'][param_tensor]

    if compare_with_normal_version:
        # debug_net.load_state_dict(dict_for_params_match)
        debug_net = nn.DataParallel(debug_net, device_ids=[0]).to(device)  # device_ids= config.GPU_devices
        debug_net.load_state_dict(checkpoint['state_dict'])
        # debug_net = debug_net.cuda(device)

    TorchScript_Module.load_state_dict(dict_for_params_match)
    TorchScript_Module = TorchScript_Module.cuda(device)  # device_ids= config.GPU_devices

    # TorchScript_Module = torch.jit.load("torchscript_version.pt").cuda(device)

    print("parameters loaded")

    if compare_with_normal_version:
        for p1, p2 in zip(debug_net.parameters(), TorchScript_Module.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                raise Exception('network parameters are not matching!')

    test_input = torch.randn([cmd_args.batch_size, cmd_args.no_ped + cmd_args.no_car, 9, cmd_args.imsize, cmd_args.imsize]).to(device)

    test_input.cpu()

    print("test input size: " , test_input.size())

    value_debug, acc_pi_debug, acc_mu_debug, acc_sigma_debug = None, None, None, None
    value_torch, acc_pi_torch, acc_mu_torch, acc_sigma_torch = None, None, None, None
    if True:

        # debug_net.eval()
        # with torch.no_grad():
        #
        #     for i in range(0, 10):
        #         test_input = torch.randn(
        #             [cmd_args.batch_size, cmd_args.no_ped + cmd_args.no_car, 9, cmd_args.imsize, cmd_args.imsize]).to(
        #             device)
        #
        #         start = time.time()
        #
        #         value_debug, acc_pi_debug, acc_mu_debug, acc_sigma_debug, \
        #             _, _, _, _, _, _ = debug_net.forward(test_input, cmd_args)
        #
        #         value_debug.cpu()
        #         print("pytorch version forward time: " + str(time.time()-start) + 's')


        print ("start time testing")
        TorchScript_Module.eval()

        with torch.no_grad():
            for i in range(0, 10):
                start = time.time()
                test_input = torch.randn(
                    [cmd_args.batch_size, 9, cmd_args.imsize, cmd_args.imsize]).to(
                    device)

                if cmd_args.fit == 'all':
                    if global_config.head_mode is "hybrid":
                        value_torch, acc_pi_torch, acc_mu_torch, acc_sigma_torch, \
                            _, _, _ = TorchScript_Module.forward(test_input)
                    elif global_config.head_mode is "mdn":
                        value_torch, acc_pi_torch, acc_mu_torch, acc_sigma_torch, \
                        ang_pi_torch, ang_mu_torch, ang_sigma_torch, \
                        _, _ = TorchScript_Module.forward(test_input)
                elif cmd_args.fit == 'val':
                    value_torch = TorchScript_Module.forward(test_input)
                    value_torch.cpu()

                end = time.time()
                print("jit version forward time: " + str(end - start) + 's')


    # if compare_with_normal_version:
    #     print('\nTesting MSE between origian network with construted TorchScript network:')
    #     print('\n debug train vs torch_script train:')
    #     print('value MSE:', torch.sum((value_debug - value_torch) ** 2))
    #     print('acc_pi MSE: ', torch.sum((acc_pi_debug - acc_pi_torch) ** 2))
    #     print('acc_mu MSE:', torch.sum((acc_mu_debug - acc_mu_torch) ** 2))
    #     print('acc_sigma MSE:', torch.sum((acc_sigma_debug - acc_sigma_torch) ** 2))
    #
    # if compare_with_normal_version:
    #     print('value:')
    #     print(value_torch.detach().numpy())
    #     print('acc_pi:')
    #     print(acc_pi_torch.detach().numpy())
    #     print('acc_mu:')
    #     print(acc_mu_torch.detach().numpy())
    #     print('acc_sigma:')
    #     print(acc_sigma_torch.detach().numpy())
    #
    TorchScript_Module.save(cmd_args.saving_name)

    print("model saved")
    

# cmd: 
# python pytorch_model_to_c.py --batch_size 1 --lr 0.0001 --train train.h5 --val val.h5 --no_vin 0 --l_h 100 --vinout 28 --w 64 --fit action
