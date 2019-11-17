import numpy as np
import torch
import torch.nn as nn

compare_with_pytorch_version = False

from Components import GPPN, resnet_modified, mdn
import torchscript_models
from dataset import *
from Data_processing import global_params

global_config = global_params.config

model_device_id = global_config.GPU_devices[0]
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")

from train import set_fit_mode_bools
from policy_value_network import PolicyValueNet


def compare_model_size():
    pytorch_size = iter(PyTorch_module.state_dict())
    ts_size = iter(TorchScript_module.state_dict())
    checkpoint_size = iter(checkpoint['state_dict'])
    print('\nOriginal Net Layers\nTorchScript Net Layers\nLoaded Net Layers\n')
    for param_tensor in PyTorch_module.state_dict():
        print(next(pytorch_size) + '\n' + next(ts_size) + '\n' + next(checkpoint_size))
        print(PyTorch_module.state_dict()[param_tensor].size())
        print(TorchScript_module.state_dict()[param_tensor].size())
        print(checkpoint['state_dict']['module.' + param_tensor].size())


def init_pytorch_module(model_checkpoint):
    module = PolicyValueNet(cmd_args)
    gpu_module = nn.DataParallel(module, device_ids=[0]).to(device)  # device_ids= config.GPU_devices
    gpu_module.load_state_dict(model_checkpoint['state_dict'])
    return gpu_module


def init_ts_module(model_checkpoint):
    TorchScript_module = None
    if cmd_args.fit == 'all':
        if global_config.head_mode is "hybrid":
            TorchScript_module = torchscript_models.PolicyValueNet(cmd_args.batch_size)
        elif global_config.head_mode is "mdn":
            TorchScript_module = torchscript_models.PolicyValueNetMdn(cmd_args.batch_size)

    if cmd_args.fit == 'val':
        TorchScript_module = torchscript_models.ValueNet(cmd_args.batch_size)

    dict_for_params_match = {}
    for param_tensor in model_checkpoint['state_dict'].keys():
        dict_for_params_match[param_tensor[7:]] = model_checkpoint['state_dict'][param_tensor]
    TorchScript_module.load_state_dict(dict_for_params_match)
    TorchScript_module = TorchScript_module.cuda(device)  # device_ids= config.GPU_devices
    return TorchScript_module


from train import parse_cmd_args, update_global_config
if __name__ == '__main__':
    # Automatic swith of GPU mode if available
    # Parsing training parameters

    cmd_args = parse_cmd_args()
    update_global_config(cmd_args)

    set_fit_mode_bools(cmd_args)
    torchscript_models.set_globals()

    print("=> num_agents_in_NN ", global_config.num_agents_in_NN)
    print("=> vanilla_resnet ", global_config.vanilla_resnet)
    print("=> default_li ", global_config.default_li)
    print("=> default_lh ", global_config.default_lh)
    print("=> GPPN_kernelsize ", global_config.GPPN_kernelsize)
    print("=> resnet_width ", global_config.resnet_width)
    print("=> vin_out_channels ", global_config.vin_out_channels)
    print("=> Num_resnet_layers ", global_config.Num_resnet_layers)
    print("=> fit ", cmd_args.fit)

    checkpoint = torch.load(cmd_args.input_model)

    print("check point loaded")

    TorchScript_module = init_ts_module(checkpoint)

    if compare_with_pytorch_version:
        PyTorch_module = init_pytorch_module(checkpoint)
        compare_model_size()

    print("parameters loaded")

    if compare_with_pytorch_version:
        for p1, p2 in zip(PyTorch_module.parameters(), TorchScript_module.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                raise Exception('network parameters are not matching!')

    test_input = torch.randn(
        [cmd_args.batch_size, cmd_args.no_ped + cmd_args.no_car, 9, cmd_args.imsize, cmd_args.imsize]).to(device)

    test_input.cpu()

    print("test input size: ", test_input.size())

    value_debug, acc_pi_debug, acc_mu_debug, acc_sigma_debug = None, None, None, None
    value_torch, acc_pi_torch, acc_mu_torch, acc_sigma_torch = None, None, None, None

    if True:
        print("start time testing")
        TorchScript_module.eval()

        with torch.no_grad():
            for i in range(0, 10):
                start = time.time()
                test_input = torch.randn(
                    [cmd_args.batch_size, 9, cmd_args.imsize, cmd_args.imsize]).to(
                    device)

                if cmd_args.fit == 'all':
                    if global_config.head_mode is "hybrid":
                        value_torch, acc_pi_torch, acc_mu_torch, acc_sigma_torch, \
                        _, _, _ = TorchScript_module.forward(test_input)
                    elif global_config.head_mode is "mdn":
                        value_torch, acc_pi_torch, acc_mu_torch, acc_sigma_torch, \
                        ang_pi_torch, ang_mu_torch, ang_sigma_torch, \
                        _, _ = TorchScript_module.forward(test_input)
                elif cmd_args.fit == 'val':
                    value_torch = TorchScript_module.forward(test_input)
                    value_torch.cpu()

                end = time.time()
                print("jit version forward time: " + str(end - start) + 's')

    TorchScript_module.save(cmd_args.output_model)

    print("model saved")

# cmd: 
# python convert_pth_model_to_pt.py --batch_size 1 --lr 0.0001 --train train.h5 --val val.h5 --no_vin 0 --l_h 100 --vinout 28 --w 64 --fit action
