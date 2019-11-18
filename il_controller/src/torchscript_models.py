import torch
import torch.nn as nn
from PyTorch_to_C import GPPN, resnet_modified, mdn
from Data_processing import global_params
from policy_value_network import get_input_slicing


PYTORCH_JIT = 1

global_config = global_params.config
''' Input Tensor is of the form ==> BatchSize * Ped * map/goal/hist * Width * height
'''

input_imsize = 32  # example input
res_output_imsize = 16  # example input

head_mode_dict = {}
head_mode_dict['mdn'] = 0
head_mode_dict['hybrid'] = 1
head_mode_dict['catergorical'] = 2
head_mode = head_mode_dict[global_config.head_mode]

gppn_l_i, gppn_l_h, gppn_k, gppn_f, input_channels_resnet, vanilla_resnet, resnet_width = \
    None, None, None, None, None, None, None

resblock_in_layers, disable_bn_in_resnet, num_steering_bins, imsize = \
    None, None, None, None


def set_globals():
    global gppn_l_i, gppn_l_h, gppn_k, gppn_f, input_channels_resnet, vanilla_resnet, resnet_width
    global resblock_in_layers, disable_bn_in_resnet, num_steering_bins, imsize
    gppn_l_i = global_config.num_gppn_inputs
    gppn_l_h = global_config.num_gppn_hidden_channels
    gppn_k = global_config.num_gppn_iterations
    gppn_f = global_config.gppn_kernelsize
    if global_config.vanilla_resnet:
        input_channels_resnet = global_config.total_num_channels
    else:
        if global_config.use_hist_channels:
            input_channels_resnet = global_config.gppn_out_channels + global_config.num_hist_channels
        else:
            input_channels_resnet = global_config.gppn_out_channels

    vanilla_resnet = global_config.vanilla_resnet  # False
    resnet_width = global_config.resnet_width  # 32

    resblock_in_layers = global_config.resblock_in_layers  # [2, 2, 2, 2]
    disable_bn_in_resnet = global_config.disable_bn_in_resnet  # False
    num_steering_bins = global_config.num_steering_bins  # 14
    imsize = 32  # default


class ActionHead(torch.jit.ScriptModule):

    def __init__(self, inplanes=64, imsize=16, num_classes=global_config.num_steering_bins):
        super(ActionHead, self).__init__()
        outplanes = 4
        self.conv = torch.jit.trace(nn.Conv2d(inplanes, outplanes, kernel_size=1,
                                              stride=1, padding=0, bias=False),
                                    torch.randn(1, inplanes, res_output_imsize, res_output_imsize))

        self.bn = torch.jit.trace(nn.BatchNorm2d(outplanes,
                                                 track_running_stats=global_config.track_running_stats),
                                  torch.randn(1, outplanes, res_output_imsize, res_output_imsize))
        self.relu = torch.jit.trace(nn.ReLU(inplace=True),
                                    torch.randn(1, outplanes, res_output_imsize, res_output_imsize))
        self.fc = torch.jit.trace(nn.Linear(in_features=imsize * imsize * outplanes,
                                            out_features=num_classes,
                                            bias=True),
                                  torch.randn(1, imsize * imsize * outplanes))

    @torch.jit.script_method
    def forward(self, x):
        out = self.conv(x)
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
                                    torch.randn(1, inplanes, res_output_imsize, res_output_imsize))
        self.bn = torch.jit.trace(nn.BatchNorm2d(outplanes,
                                                 track_running_stats=global_config.track_running_stats),
                                  torch.randn(1, outplanes, res_output_imsize, res_output_imsize))
        self.relu = torch.jit.trace(nn.ReLU(inplace=True),
                                    torch.randn(1, outplanes, res_output_imsize, res_output_imsize))

        self.mdn = mdn.MDN(in_features=imsize * imsize * outplanes, out_features=1,
                           num_gaussians=global_config.num_guassians_in_heads)

    @torch.jit.script_method
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        pi, sigma, mu = self.mdn(x)
        return pi, sigma, mu


class ValueHead(torch.jit.ScriptModule):

    def __init__(self, inplanes=64, imsize=16):
        super(ValueHead, self).__init__()
        outplanes = 4
        self.conv = torch.jit.trace(nn.Conv2d(inplanes, outplanes, kernel_size=1,
                                              stride=1, bias=False),
                                    torch.randn(1, inplanes, res_output_imsize, res_output_imsize))
        self.bn = torch.jit.trace(nn.BatchNorm2d(outplanes,
                                                 track_running_stats=global_config.track_running_stats),
                                  torch.randn(1, outplanes, res_output_imsize, res_output_imsize))
        self.relu = torch.jit.trace(nn.ReLU(inplace=True),
                                    torch.randn(1, outplanes, res_output_imsize, res_output_imsize))

        self.fc = torch.jit.trace(nn.Linear(in_features=imsize * imsize * outplanes,
                                            out_features=1, bias=True), torch.randn(1, imsize * imsize * outplanes))

        self.ped_start, self.ped_end, self.gppn_end, self.hist_start, self.hist_end \
            = get_input_slicing()

    @torch.jit.script_method
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class PolicyValueNet(torch.jit.ScriptModule):
    __constants__ = ['num_steering_bins', 'num_vel_bins',
                     'num_acc_bins', 'output_channels_gppn', 'no_input_resnet',
                     'num_resnet_out_features', 'imsize',
                     'total_num_channels', 'num_hist_channels',
                     'resblock_in_layers']

    def __init__(self):
        super(PolicyValueNet, self).__init__()

        self.car_gppn = GPPN.GPPN(l_i=gppn_l_i, l_h=gppn_l_h, k=gppn_k, f=gppn_f)

        # 3 channels include the Value image and the 2 hist layers (idx 0 is value image)

        self.resnet = resnet_modified.ResNetModified(layers=resblock_in_layers,
                                                     ip_channels=input_channels_resnet,
                                                     resnet_width_local=resnet_width)

        self.pre_resnet = torch.jit.trace(nn.Sequential(
            resnet_modified.BasicBlock(input_channels_resnet, input_channels_resnet),
            resnet_modified.BasicBlock(input_channels_resnet, input_channels_resnet)),
            torch.randn(1, input_channels_resnet, input_imsize, input_imsize))

        self.value_head = ValueHead(inplanes=resnet_width)
        self.acc_head = ActionMdnHead(inplanes=resnet_width)
        self.ang_head = ActionHead(inplanes=resnet_width,
                                   num_classes=num_steering_bins)

        self.ped_start, self.ped_end, self.gppn_end, self.hist_start, self.hist_end \
            = get_input_slicing()

    @torch.jit.script_method
    def forward(self, X):
        # def forward(self, X, batch_size = X.size(0), config):
        # Input Tensor is of the form ==> batch_size * Agent * map/goal/hist * Width * height
        # switches batch and agent dim in order to iterate over the agent dim

        car_input = X.contiguous()

        reshape_car_input = car_input

        car_gppn_input = reshape_car_input[:, 0:self.gppn_end, :, :]  # with 4 ped maps and goal channel

        # car_hist_data = reshape_car_input[:, self.hist_start:self.hist_end, :, :]  # with 4 hist channels

        car_values = self.car_gppn(X=car_gppn_input)

        # car_features = torch.cat((car_values, car_hist_data), 1).contiguous()  # stack the channels
        car_features = car_values

        car_res_features = self.pre_resnet(car_features)

        res_image = self.resnet(car_res_features)

        value = self.value_head(res_image)
        acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
        ang = self.ang_head(res_image)
        return value, acc_pi, acc_mu, acc_sigma, ang, \
               car_values[0], res_image[0]


class PolicyValueNetMdn(torch.jit.ScriptModule):
    __constants__ = ['fc_modules_size', 'num_steering_bins', 'num_vel_bins',
                     'num_acc_bins', 'output_channels_gppn', 'no_input_resnet',
                     'num_resnet_out_features', 'imsize',
                     'total_num_channels', 'num_hist_channels',
                     'resblock_in_layers', 'batchsize']

    def __init__(self):
        super(PolicyValueNetMdn, self).__init__()
        self.car_gppn = GPPN.GPPN(l_i=gppn_l_i, l_h=gppn_l_h, k=gppn_k, f=gppn_f)
        self.pre_resnet = torch.jit.trace(nn.Sequential(
            resnet_modified.BasicBlock(input_channels_resnet, input_channels_resnet),
            resnet_modified.BasicBlock(input_channels_resnet, input_channels_resnet)),
            torch.randn(1, input_channels_resnet, input_imsize, input_imsize))

        self.resnet = resnet_modified.ResNetModified(layers=resblock_in_layers,
                                                     ip_channels=input_channels_resnet,
                                                     resnet_width_local=resnet_width)

        self.value_head = ValueHead(inplanes=resnet_width)
        self.acc_head = ActionMdnHead(inplanes=resnet_width)
        self.ang_head = ActionMdnHead(inplanes=resnet_width)

        self.ped_start, self.ped_end, self.gppn_end, self.hist_start, self.hist_end \
            = get_input_slicing()
        
    @torch.jit.script_method
    def forward(self, X):
        # def forward(self, X, batch_size = X.size(0), config):
        # Input Tensor is of the form ==> batch_size * Agent * map/goal/hist * Width * height
        # switches batch and agent dim in order to iterate over the agent dim
        # reshape_X = X.permute(1, 0, 2, 3, 4)

        reshape_car_input = X.contiguous()

        car_gppn_input = reshape_car_input[:, 0:self.gppn_end, :, :]  # with 4 ped maps and goal channel

        # car_hist_data = reshape_car_input[:, self.hist_start:self.hist_end, :, :]  # with 4 hist channels

        car_values = self.car_gppn(X=car_gppn_input)

        # car_features = torch.cat((car_values, car_hist_data), 1).contiguous()  # stack the channels
        car_features = car_values

        car_res_features = self.pre_resnet(car_features)

        res_image = self.resnet(car_res_features)

        value = self.value_head(res_image)
        acc_pi, acc_sigma, acc_mu = self.acc_head(res_image)
        ang_pi, ang_sigma, ang_mu = self.ang_head(res_image)

        return value, acc_pi, acc_mu, acc_sigma, ang_pi, ang_mu, ang_sigma, \
               car_values[0], res_image[0]


class ValueNet(torch.jit.ScriptModule):
    __constants__ = ['fc_modules_size', 'num_steering_bins', 'num_vel_bins',
                     'num_acc_bins', 'output_channels_gppn', 'no_input_resnet',
                     'num_resnet_out_features', 'imsize',
                     'total_num_channels', 'num_hist_channels',
                     'resblock_in_layers', 'batchsize']

    def __init__(self, batchsize):
        super(ValueNet, self).__init__()

        self.resnet = resnet_modified.ResNet_2L(layers=resblock_in_layers,
                                                ip_channels=input_channels_resnet,
                                                resnet_width_local=resnet_width)

        self.value_head = ValueHead(inplanes=resnet_width)

        self.ped_start, self.ped_end, self.gppn_end, self.hist_start, self.hist_end \
            = get_input_slicing()

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
