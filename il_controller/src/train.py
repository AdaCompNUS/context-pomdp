import time
import os
import argparse

import torch as torch

from dataset import DrivingData
from policy_value_network import PolicyValueNet
import torch.optim as optim

from tensorboardX import SummaryWriter
from Data_processing import global_params
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from Components.nan_police import *
from visualization import *
from Components.max_ent_loss import CELossWithMaxEntRegularizer
from Components import mdn

from gamma_dataset import GammaDataset, reset_global_params_for_gamma_dataset
from dataset import reset_global_params_for_pomdp_dataset

global_config = global_params.config

model_device_id = global_config.GPU_devices[0]
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")

best_val_loss = None
cur_val_loss = None

lr_factor = 0.3
last_train_loss = 0.0
first_val_loss = 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, flag):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.last = 0
        self.flag = flag

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.last = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.last = val


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num  # =============== here #################

        return loss


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, y):
        loss = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        loss = loss.sum()
        return loss


train_flag = 'train/'
val_flag = 'val/'


def train():
    sm = nn.Softmax(dim=1)

    clear_png_files('./visualize/', remove_flag='train_')
    clear_png_files('./visualize/', remove_flag='val_')

    print("Enter training")
    global best_val_loss
    global cur_val_loss

    for epoch in range(cmd_args.start_epoch, cmd_args.epochs):  # Loop over dataset multiple times
        # setting seed for random generator
        current_time = int(round(time.time() * 1000))
        torch.cuda.manual_seed_all(current_time)

        accuracy_recorders, loss_recorders = alloc_recorders()

        net.train()

        last_data = None
        last_loss = 0.0

        with tqdm(total=len(train_loader.dataset), ncols=100) as pbar:
            for i, data_mini_batch in enumerate(train_loader):  # Loop over batches of data

                visualize_data = False
                if i <= 0:  # 0 -> 20
                    visualize_data = True

                # Get input batch
                input_images, semantic_data, value_labels, acc_labels, ang_labels, velocity_labels, lane_labels = \
                    data_mini_batch
                input_images, semantic_data, value_labels, acc_labels, ang_labels, velocity_labels, lane_labels = \
                    input_images.to(device), semantic_data.to(device), value_labels.to(device), acc_labels.to(device), \
                    ang_labels.to(device), velocity_labels.to(device), lane_labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                if global_config.head_mode == "mdn":
                    # Forward pass
                    acc_pi, acc_mu, acc_sigma, ang_pi, ang_mu, ang_sigma, vel_pi, vel_mu, vel_sigma, lane_logits, value = \
                        forward_pass(input_images, semantic_data, step=epoch, print_time=visualize_data, image_flag=train_flag)

                    visualize_mdn_predictions(epoch, acc_mu, acc_pi, acc_sigma, acc_labels, ang_mu, ang_pi, ang_sigma,
                                              ang_labels, vel_mu, vel_pi, vel_sigma, velocity_labels, lane_logits,
                                              lane_labels, visualize_data, sm, train_flag)

                    # Loss
                    acc_loss, ang_loss, vel_loss, lane_loss, v_loss = \
                        calculate_mdn_loss(acc_pi, acc_mu, acc_sigma, acc_labels,
                                           ang_pi, ang_mu, ang_sigma, ang_labels,
                                           vel_pi, vel_mu, vel_sigma, velocity_labels,
                                           lane_logits, lane_labels,
                                           value, value_labels)

                    loss_dict = choose_loss(acc_loss, ang_loss, vel_loss, lane_loss, v_loss, global_config)

                    # accuracy
                    ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy = \
                        calculate_mdn_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                                               ang_pi, ang_mu, ang_sigma, ang_labels,
                                               vel_pi, vel_mu, vel_sigma, velocity_labels,
                                               lane_logits, lane_labels)

                    accuracy_dict = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, lane_accuracy,
                                                    global_config)

                    record_loss_and_accuracy(input_images, loss_dict, accuracy_dict, loss_recorders, accuracy_recorders)

                elif global_config.head_mode == "hybrid":
                    # Forward pass
                    acc_pi, acc_mu, acc_sigma, \
                    ang_logits, \
                    vel_pi, vel_mu, vel_sigma, lane_logits, value = \
                        forward_pass(input_images, semantic_data, step=epoch, print_time=visualize_data,
                                                                                 image_flag=train_flag)  # epoch

                    # epoch -> i
                    visualize_hybrid_predictions(epoch, acc_mu, acc_pi, acc_sigma, acc_labels, ang_logits, ang_labels,
                                                 vel_mu, vel_pi, vel_sigma, velocity_labels,
                                                 lane_logits, lane_labels, value, value_labels,
                                                 visualize_data, sm, train_flag)

                    # Loss
                    acc_loss, ang_loss, vel_loss, lane_loss, v_loss = \
                        calculate_hybrid_loss(acc_pi, acc_mu, acc_sigma, acc_labels,
                                              ang_logits, ang_labels,
                                              vel_pi, vel_mu, vel_sigma, velocity_labels,
                                              lane_logits, lane_labels,
                                              value, value_labels)

                    loss_dict = choose_loss(acc_loss, ang_loss, vel_loss, lane_loss, v_loss, global_config)

                    # accuracy
                    ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy = \
                        calculate_hybrid_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                                                  ang_logits, ang_labels,
                                                  vel_pi, vel_mu, vel_sigma, velocity_labels,
                                                  lane_logits, lane_labels)

                    accuracy_dict = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, lane_accuracy,
                                                    global_config)

                    record_loss_and_accuracy(input_images, loss_dict, accuracy_dict, loss_recorders, accuracy_recorders)

                else:
                    # Forward pass
                    acc_logits, ang_logits, vel_logits, lane_logits, value = \
                        forward_pass(input_images, semantic_data, step=epoch, print_time=visualize_data, image_flag=train_flag)
                    # print("lane_logits = {}".format(lane_logits))

                    visualize_predictions(epoch, acc_logits, acc_labels, ang_logits, ang_labels, vel_logits,
                                          velocity_labels, lane_logits, lane_labels, visualize_data, sm, train_flag)

                    # Loss
                    acc_loss, ang_loss, vel_loss, lane_loss, v_loss = \
                        calculate_loss(acc_logits, acc_labels, ang_logits, ang_labels, vel_logits, velocity_labels,
                                       lane_logits, lane_labels, value, value_labels)

                    loss_dict = choose_loss(acc_loss, ang_loss, vel_loss, lane_loss, v_loss, global_config)

                    # accuracy

                    ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy = \
                        calculate_accuracy(acc_logits, acc_labels, ang_logits, ang_labels,
                                           vel_logits, velocity_labels, lane_logits, lane_labels)

                    accuracy_dict = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, lane_accuracy,
                                                    global_config)

                    record_loss_and_accuracy(input_images, loss_dict, accuracy_dict, loss_recorders, accuracy_recorders)

                last_loss = loss_dict['combined'][0]
                # Backward pass
                loss_dict['combined'][0].backward()

                optimizer.step()
                pbar.update(input_images.size(0))

        # Evaluation
        validation_loss_recorders, validation_accuracy_recorders = evaluate(last_loss, epoch)

        # Schedule learning rate
        if global_config.enable_scheduler:
            scheduler.step(validation_loss_recorders['combined'].avg)

        # record the data for visualization in Tensorboard
        record_tensor_board_log(epoch, accuracy_recorders, loss_recorders, validation_accuracy_recorders,
                                validation_loss_recorders)

        cur_val_loss = validation_loss_recorders['combined'].avg

        save_model(epoch)

        if termination():
            break

    print('\nFinished training. \n')


def evaluate(last_train_loss, epoch):
    start_time = time.time()

    sm = nn.Softmax(dim=1)

    accuracy_recorders_val, loss_recorders_val = alloc_recorders()

    net.eval()

    first_loss = None

    for i, mini_batch in enumerate(val_loader):

        visualize_data = False
        if i <= 0:
            visualize_data = True

        with torch.no_grad():
            # Get input batch
            input_images, semantic_data, value_labels, acc_labels, ang_labels, velocity_labels, lane_labels = mini_batch
            if global_config.visualize_val_data:
                debug_data(input_images[1])
            input_images, semantic_data, value_labels, acc_labels, ang_labels, velocity_labels, lane_labels = \
                input_images.to(device), semantic_data.to(device), value_labels.to(device), acc_labels.to(device), \
                ang_labels.to(device), velocity_labels.to(device), lane_labels.to(device)

            if global_config.head_mode == "mdn":
                # Forward pass
                acc_pi, acc_mu, acc_sigma, \
                ang_pi, ang_mu, ang_sigma, \
                vel_pi, vel_mu, vel_sigma, \
                lane_logits, value = forward_pass(input_images, semantic_data, step=epoch, print_time=visualize_data,
                                                  image_flag=val_flag)

                visualize_mdn_predictions(epoch, acc_mu, acc_pi, acc_sigma, acc_labels, ang_mu, ang_pi, ang_sigma,
                                          ang_labels, vel_mu, vel_pi, vel_sigma, velocity_labels, lane_labels,
                                          lane_logits, visualize_data, sm, val_flag)

                # Loss
                acc_loss, ang_loss, vel_loss, lane_loss, v_loss = \
                    calculate_mdn_loss(acc_pi, acc_mu, acc_sigma, acc_labels,
                                       ang_pi, ang_mu, ang_sigma, ang_labels,
                                       vel_pi, vel_mu, vel_sigma, velocity_labels,
                                       lane_logits, lane_labels,
                                       value, value_labels)
                loss_dict = choose_loss(acc_loss, ang_loss, vel_loss, lane_loss, v_loss, global_config)

                # Accuracy
                ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy = \
                    calculate_mdn_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                                           ang_pi, ang_mu, ang_sigma, ang_labels,
                                           vel_pi, vel_mu, vel_sigma, velocity_labels,
                                           lane_logits, lane_labels)
                accuracy_dict = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, lane_accuracy, global_config)

                # Statistics
                record_loss_and_accuracy(input_images, loss_dict, accuracy_dict, loss_recorders_val,
                                         accuracy_recorders_val)

            elif global_config.head_mode == "hybrid":

                # Forward pass
                acc_pi, acc_mu, acc_sigma, \
                ang_logits, \
                vel_pi, vel_mu, vel_sigma, lane_logits, value = \
                    forward_pass(input_images, semantic_data, step=epoch, print_time=visualize_data, image_flag=val_flag)

                visualize_hybrid_predictions(epoch, acc_mu, acc_pi, acc_sigma, acc_labels, ang_logits, ang_labels,
                                             vel_mu, vel_pi, vel_sigma, velocity_labels,
                                             lane_logits, lane_labels, value, value_labels,
                                             visualize_data, sm, val_flag)

                # Loss
                acc_loss, ang_loss, vel_loss, lane_loss, v_loss = \
                    calculate_hybrid_loss(acc_pi, acc_mu, acc_sigma, acc_labels,
                                          ang_logits, ang_labels,
                                          vel_pi, vel_mu, vel_sigma, velocity_labels, lane_logits, lane_labels,
                                          value, value_labels)

                loss_dict = choose_loss(acc_loss, ang_loss, vel_loss, lane_loss, v_loss, global_config)

                # Accuracy
                ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy = \
                    calculate_hybrid_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                                              ang_logits, ang_labels,
                                              vel_pi, vel_mu, vel_sigma, velocity_labels,
                                              lane_logits, lane_labels)

                accuracy_dict = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, lane_accuracy, global_config)

                record_loss_and_accuracy(input_images, loss_dict, accuracy_dict, loss_recorders_val,
                                         accuracy_recorders_val)

            else:
                # Forward pass
                acc_logits, ang_logits, vel_logits, lane_logits, value = \
                    forward_pass(input_images, semantic_data, step=epoch, print_time=visualize_data, image_flag=val_flag)

                visualize_predictions(epoch, acc_logits, acc_labels, ang_logits, ang_labels, vel_logits,
                                      velocity_labels, lane_logits, lane_labels, visualize_data, sm, val_flag)

                # Loss
                acc_loss, ang_loss, vel_loss, lane_loss, v_loss = calculate_loss(acc_logits, acc_labels, ang_logits,
                                                                                 ang_labels,
                                                                                 vel_logits, velocity_labels,
                                                                                 lane_logits,
                                                                                 lane_labels, value, value_labels)

                loss_dict = choose_loss(acc_loss, ang_loss, vel_loss, lane_loss, v_loss, global_config)

                # Accuracy
                ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy = calculate_accuracy(acc_logits, acc_labels,
                                                                                             ang_logits, ang_labels,
                                                                                             vel_logits,
                                                                                             velocity_labels,
                                                                                             lane_logits, lane_labels)

                accuracy_dict = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, lane_accuracy, global_config)

                # Statistics
                record_loss_and_accuracy(input_images, loss_dict, accuracy_dict, loss_recorders_val,
                                         accuracy_recorders_val)

            if i == 0:
                first_loss = loss_dict['combined'][0]

    if abs(first_loss - last_train_loss) >= 4:
        print("Last mdn accuracy (val): %f" % accuracy_recorders_val['combined'].last)
    inference_time = time.time() - start_time
    print("Evaluation time %fs" % inference_time)

    return loss_recorders_val, accuracy_recorders_val


def visualize_mdn_predictions(epoch, acc_mu, acc_pi, acc_sigma, acc_labels, ang_mu, ang_pi, ang_sigma, ang_labels,
                              vel_mu, vel_pi, vel_sigma, velocity_labels, lane, lane_labels, visualize_data, sm, flag):
    if visualize_data:
        try:
            print('')
            lane_probs = None
            if lane is not None:
                lane_probs = sm(lane)
            for i in range(0, min(10, cmd_args.batch_size)):
                acc_mu_draw, acc_pi_draw, acc_sigma_draw, acc_label = None, None, None, None
                ang_mu_draw, ang_pi_draw, ang_sigma_draw, ang_label = None, None, None, None
                vel_mu_draw, vel_pi_draw, vel_sigma_draw, vel_label = None, None, None, None
                lane_prob, lane_label = None, None

                if acc_pi is not None:
                    acc_pi_draw = acc_pi[i].detach()
                    acc_mu_draw = acc_mu[i].detach()
                    acc_sigma_draw = acc_sigma[i].detach()
                if ang_pi is not None:
                    ang_pi_draw = ang_pi[i].detach()
                    ang_mu_draw = ang_mu[i].detach()
                    ang_sigma_draw = ang_sigma[i].detach()
                if vel_pi is not None:
                    vel_pi_draw = vel_pi[i].detach()
                    vel_mu_draw = vel_mu[i].detach()
                    vel_sigma_draw = vel_sigma[i].detach()
                if lane_probs is not None:
                    lane_prob = lane_probs[i].detach()

                acc_label, ang_label, vel_label, lane_label, _ = \
                    detach_labels(i, acc_labels, ang_labels, velocity_labels, lane_labels, None)

                visualize_mdn_output_with_labels(flag + str(epoch),
                                                 acc_mu_draw, acc_pi_draw, acc_sigma_draw,
                                                 ang_mu_draw, ang_pi_draw, ang_sigma_draw,
                                                 vel_mu_draw, vel_pi_draw, vel_sigma_draw,
                                                 lane_prob,
                                                 acc_label, ang_label, vel_label, lane_label)
        except Exception as e:
            print(e)
            exit(1)


def detach_labels(i, acc_labels, ang_labels, velocity_labels, lane_labels, value_labels):
    acc_label, ang_label, vel_label, lane_label, value_label = None, None, None, None, None
    if acc_labels is not None:
        acc_label = acc_labels[i].detach()
    if ang_labels is not None:
        ang_label = ang_labels[i].detach()
    if lane_labels is not None:
        lane_label = lane_labels[i].detach()
    if velocity_labels is not None:
        vel_label = velocity_labels[i].detach()
    if value_labels is not None:
        value_label = value_labels[i].detach()
    return acc_label, ang_label, vel_label, lane_label, value_label


def visualize_hybrid_predictions(epoch, acc_mu, acc_pi, acc_sigma, acc_labels, ang, ang_labels,
                                 vel_mu, vel_pi, vel_sigma, velocity_labels,
                                 lane, lane_labels, value, value_labels,
                                 visualize_data, sm, flag):
    if visualize_data:
        print('')
        ang_probs, lane_probs = None, None
        if ang is not None:
            ang_probs = sm(ang)
        if lane is not None:
            lane_probs = sm(lane)
        for i in range(0, min(10, cmd_args.batch_size)):
            acc_mu_draw, acc_pi_draw, acc_sigma_draw, acc_label = None, None, None, None
            vel_mu_draw, vel_pi_draw, vel_sigma_draw, vel_label = None, None, None, None
            ang_prob, ang_label = None, None
            lane_prob, lane_label = None, None
            value_draw, value_label = None, None

            if acc_pi is not None:
                acc_pi_draw = acc_pi[i].detach()
                acc_mu_draw = acc_mu[i].detach()
                acc_sigma_draw = acc_sigma[i].detach()
            if ang_probs is not None:
                ang_prob = ang_probs[i].detach()
            if lane_probs is not None:
                lane_prob = lane_probs[i].detach()
            if vel_pi is not None:
                vel_pi_draw = vel_pi[i].detach()
                vel_mu_draw = vel_mu[i].detach()
                vel_sigma_draw = vel_sigma[i].detach()
            if value is not None:
                value_draw = value[i].detach()

            acc_label, ang_label, vel_label, lane_label, value_label = detach_labels(
                i, acc_labels, ang_labels, velocity_labels, lane_labels, value_labels)

            visualize_hybrid_output_with_labels(flag + str(epoch) + '_' + str(i),
                                                acc_mu_draw, acc_pi_draw, acc_sigma_draw,
                                                ang_prob,
                                                vel_mu_draw, vel_pi_draw, vel_sigma_draw,
                                                lane_prob,
                                                value,
                                                acc_label, ang_label, vel_label, lane_label, value_label)


def visualize_predictions(epoch, acc, acc_labels, ang, ang_labels, vel, velocity_labels, lane, lane_labels,
                          visualize_data, sm, flag):
    if visualize_data:
        print('')
        acc_probs, ang_probs, vel_probs, lane_probs = None, None, None, None
        if acc is not None:
            acc_probs = sm(acc)
        if ang is not None:
            ang_probs = sm(ang)
        if vel is not None:
            vel_probs = sm(vel)
        if lane is not None:
            lane_probs = sm(lane)
        for i in range(0, min(10, cmd_args.batch_size)):
            acc_prob, ang_prob, vel_prob, lane_prob = None, None, None, None
            acc_label, ang_label, vel_label, lane_label = None, None, None, None
            if acc_probs is not None:
                acc_prob = acc_probs[i].detach()
            if ang_probs is not None:
                ang_prob = ang_probs[i].detach()
            if lane_probs is not None:
                lane_prob = lane_probs[i].detach()
            if vel_probs is not None:
                vel_prob = vel_probs[i].detach()

            if acc_labels is not None:
                acc_label = acc_labels[i].detach()
            if ang_labels is not None:
                ang_label = ang_labels[i].detach()
            if lane_labels is not None:
                lane_label = lane_labels[i].detach()
            if velocity_labels is not None:
                vel_label = velocity_labels[i].detach()

            visualize_output_with_labels(flag + str(epoch) + '_' + str(i),
                                         acc_prob, ang_prob, vel_prob, lane_prob,
                                         acc_label, ang_label, vel_label, lane_label)


def alloc_recorders():
    loss_recorders = {}
    accuracy_recorders = {}
    flags = ['combined']
    if global_config.fit_all:
        if config.use_vel_head:
            flags = flags + ['steer', 'acc', 'vel', 'lane', 'val']
        else:
            flags = flags + ['steer', 'acc', 'lane', 'val']
    if global_config.fit_action:
        if config.use_vel_head:
            flags = flags + ['steer', 'acc', 'vel', 'lane']
        else:
            flags = flags + ['steer', 'acc', 'lane']
    if global_config.fit_acc:
        flags = flags + ['acc']
    if global_config.fit_ang:
        flags = flags + ['steer']
    if global_config.fit_vel:
        flags = flags + ['vel']
    if global_config.fit_lane:
        flags = flags + ['lane']

    for flag in flags:
        loss_recorders[flag] = AverageMeter(flag)
        accuracy_recorders[flag] = AverageMeter(flag)

    return accuracy_recorders, loss_recorders


def forward_pass(input_images, semantic_input, step=0, drive_net=None, cmd_config=None, print_time=False,
                 image_flag=''):
    try:
        start_time = time.time()

        if drive_net is None:
            drive_net = net
        if cmd_config is None:
            cmd_config = cmd_args

        ped_gppn_out, car_gppn_out, res_out, res_image = \
            torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0)
        if global_config.head_mode == "mdn":
            value, acc_output, ang_output, vel_output, lane_logits, \
            car_gppn_out, res_image = drive_net.forward(input_images, semantic_input, cmd_config)
            acc_pi, acc_sigma, acc_mu = acc_output
            vel_pi, vel_sigma, vel_mu = vel_output
            ang_pi, ang_sigma, ang_mu = ang_output
            if global_config.print_preds:
                print("predicted angle:")
                print(ang_pi, ang_mu, ang_sigma)

            elapsed_time = time.time() - start_time

            if print_time:
                print("Inference time: " + str(elapsed_time) + " s")
                visualize_input_output(input_images, ped_gppn_out, car_gppn_out, res_image, step, image_flag)

            return acc_pi, acc_mu, acc_sigma, ang_pi, ang_mu, ang_sigma, \
                   vel_pi, vel_mu, vel_sigma, lane_logits, value

        elif global_config.head_mode == "hybrid":
            value, acc_output, ang_logits, vel_output, lane_logits, car_gppn_out, res_image = \
                drive_net.forward(input_images, semantic_input, cmd_config)
            acc_pi, acc_sigma, acc_mu = acc_output
            vel_pi, vel_sigma, vel_mu = vel_output
            if global_config.print_preds:
                print("predicted angle logits:")
                print(ang_logits)

            elapsed_time = time.time() - start_time
            if print_time and config.visualize_inter_data:
                print("Inference time: " + str(elapsed_time) + " s")
                visualize_input_output(input_images, ped_gppn_out, car_gppn_out, res_image, step, image_flag)

            return acc_pi, acc_mu, acc_sigma, \
                   ang_logits, vel_pi, vel_mu, vel_sigma, lane_logits, value

        else:
            # print('input sizes {} {}'.format(input_images.size(), semantic_input.size()))
            value, acc_logits, ang_logits, vel_logits, lane_logits, \
            car_gppn_out, res_image = drive_net.forward(input_images, semantic_input, cmd_config)
            if global_config.print_preds:
                print("predicted angle:")
                print(ang_logits)

            elapsed_time = time.time() - start_time
            if print_time:
                print("Inference time: " + str(elapsed_time) + " s")
                visualize_input_output(input_images, ped_gppn_out, car_gppn_out, res_image, step, image_flag)

            return acc_logits, ang_logits, vel_logits, lane_logits, value
    except Exception as e:
        error_handler(e)

def forward_pass_jit(X, step=0, drive_net=None, cmd_config=None, print_time=False, image_flag=''):
    start_time = time.time()

    if drive_net is None:
        drive_net = net
    if cmd_config is None:
        cmd_config = cmd_args
    ped_gppn_out, car_gppn_out, res_out, res_image = \
        torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0)

    if global_config.head_mode == "hybrid":
        value, acc_pi, acc_sigma, acc_mu, \
        ang_logits, lane_logits, car_gppn_out, res_image = drive_net.forward(X.squeeze(0))

        if global_config.print_preds:
            print("predicted angle:")
            print(ang_logits)

        # if value:
        #     value.cpu()
        elapsed_time = time.time() - start_time

        if print_time and config.visualize_inter_data:
            print("Inference time: " + str(elapsed_time) + " s")
            visualize_input_output(X, ped_gppn_out, car_gppn_out, res_image, step, image_flag)

        return acc_pi, acc_mu, acc_sigma, \
               ang_logits, None, None, None, lane_logits, value

    else:
        print("jit model only supports hybird head mode")
        exit(1)


def calculate_loss(acc, acc_labels, ang, ang_labels, vel, velocity_labels, lane, lane_labels, value, value_labels):
    vel_loss, acc_loss, ang_loss, lane_loss, v_loss = None, None, None, None, None
    # print('[calculate_loss]')

    if config.fit_acc or config.fit_action or config.fit_all:
        acc_loss = cel_criterion(acc, acc_labels)
    if config.fit_ang or config.fit_action or config.fit_all:
        ang_loss = cel_criterion(ang, ang_labels)
        # print('ang_loss = {}'.format(ang_loss))
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            vel_loss = cel_criterion(vel, velocity_labels)
            # print('vel_loss = {}'.format(vel_loss))
    if config.fit_lane or config.fit_action or config.fit_all:
        lane_loss = cel_criterion(lane, lane_labels)
    if config.fit_val or config.fit_all:
        v_loss = criterion(value, value_labels)
    return acc_loss, ang_loss, vel_loss, lane_loss, v_loss


def calculate_mdn_loss(acc_pi, acc_mu, acc_sigma, acc_labels, ang_pi, ang_mu, ang_sigma, ang_labels,
                       vel_pi, vel_mu, vel_sigma, velocity_labels, lane, lane_labels, value, value_labels):
    vel_loss, acc_loss, ang_loss, lane_loss, v_loss = None, None, None, None, None

    if config.fit_acc or config.fit_action or config.fit_all:
        acc_loss = calculate_mdn_loss_action(acc_pi, acc_mu, acc_sigma, acc_labels)
    if config.fit_ang or config.fit_action or config.fit_all:
        ang_loss = calculate_mdn_loss_action(ang_pi, ang_mu, ang_sigma, ang_labels)
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            vel_loss = calculate_mdn_loss_action(vel_pi, vel_mu, vel_sigma, velocity_labels)
    if config.fit_lane or config.fit_action or config.fit_all:
        lane_loss = cel_criterion(lane, lane_labels)
    if config.fit_val or config.fit_all:
        v_loss = criterion(value, value_labels)
    return acc_loss, ang_loss, vel_loss, lane_loss, v_loss


def calculate_hybrid_loss(acc_pi, acc_mu, acc_sigma, acc_labels, ang, ang_labels,
                          vel_pi, vel_mu, vel_sigma, velocity_labels, lane, lane_labels, value, value_labels):
    v_loss, vel_loss, acc_loss, ang_loss, lane_loss = None, None, None, None, None

    if config.fit_val or config.fit_all:
        v_loss = criterion(value, value_labels)
    if config.fit_acc or config.fit_action or config.fit_all:
        acc_loss = calculate_mdn_loss_action(acc_pi, acc_mu, acc_sigma, acc_labels)
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            vel_loss = calculate_mdn_loss_action(vel_pi, vel_mu, vel_sigma, velocity_labels)
            if vel_loss != vel_loss:
                print('vel loss {}, pi {}, mu {}, sigma {}, label {}'.format(vel_loss,
                    vel_pi.cpu().detach().numpy()[0], vel_mu.cpu().detach().numpy()[0], vel_sigma.cpu().detach().numpy()[0],
                    velocity_labels.cpu().detach().numpy()[0]))
    if config.fit_ang or config.fit_action or config.fit_all:
        ang_loss = cel_criterion(ang, ang_labels)
    if config.fit_lane or config.fit_action or config.fit_all:
        lane_loss = cel_criterion(lane, lane_labels)
    return acc_loss, ang_loss, vel_loss, lane_loss, v_loss


def calculate_mdn_loss_action(pi, mu, sigma, labels):
    loss = mdn.mdn_loss(pi, sigma, mu, labels)
    return loss


def termination():
    terminal = False
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        print("Current learning rate: {}".format(current_lr))
        if current_lr <= cmd_args.lr * lr_factor ** 6:
            terminal = True
            print("Terminate learning: learning rate too small")
        break
    return terminal


def save_model(epoch):
    global best_val_loss
    global cur_val_loss
    if not best_val_loss or cur_val_loss < best_val_loss:
        print("Saving best model at epoch ", epoch, "...")
        best_val_loss = cur_val_loss
        save_checkpoint(make_checkpoint_dict(epoch, best_val_loss, global_config, cmd_args), True, cmd_args.modelfile)
    else:
        print("Saving cur model at epoch ", epoch, "...")
        model_file = cmd_args.modelfile.split('.')[0] + '_cur' + '.pth'
        save_checkpoint(make_checkpoint_dict(epoch, best_val_loss, global_config, cmd_args), True, model_file)


def record_tensor_board_log(epoch, accuracy_recorders, loss_recorders, val_accuracy_recorders, val_loss_recorders):
    flag_mapping = {'combined': '',
                    'steer': '_Steer',
                    'acc': '_Acc',
                    'vel': '_Vel',
                    'lane': '_Lane',
                    'val': '_Val',
                    }

    for key in loss_recorders.keys():
        flag = flag_mapping[key]
        train_accuracy = None
        if key in accuracy_recorders.keys():
            train_accuracy = accuracy_recorders[key]
        train_loss = loss_recorders[key]
        val_accuracy = None
        if key in val_accuracy_recorders.keys():
            val_accuracy = val_accuracy_recorders[key]
        val_loss = val_loss_recorders[key]
        record_tensor_board_data(epoch, train_accuracy, train_loss, val_accuracy, val_loss, flag)

    current_lr = 0.0
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    writer.add_scalars('data/lr', {'lr': current_lr, '0': 0.0}, epoch + 1)


def record_tensor_board_data(epoch, epoch_accuracy, epoch_loss, validation_accuracy, validation_loss, flag):
    if epoch_loss:
        writer.add_scalars('data/loss_group', {'Train_Loss' + flag: epoch_loss.avg,
                                               'Validation_Loss' + flag: validation_loss.avg,
                                               '0': 0.0}, epoch + 1)
    if 'Val' in flag:
        if epoch_loss:
            print("Epoch %d value loss(train) %f | loss(eval) %f" % (epoch, epoch_loss.avg, validation_loss.avg))
        else:
            print("Epoch %d value loss(eval) %f" % (epoch, validation_loss.avg))
    else:
        if epoch_accuracy:
            writer.add_scalars('data/accuracy_group', {'Train_Accuracy' + flag: epoch_accuracy.avg,
                                                       'Validation_Accuracy' + flag: validation_accuracy.avg,
                                                       '0': 0.0}, epoch + 1)
        else:
            pass

        if epoch_accuracy and epoch_loss:
            print("Epoch %d %s Accuracy(train) %f loss(train) %f | Accuracy(val) %f loss(eval) %f" % (
                epoch, flag.replace('_', ''),
                epoch_accuracy.avg, epoch_loss.avg, validation_accuracy.avg, validation_loss.avg))
        else:
            print("Epoch %d %s Accuracy(val) %f loss(eval) %f" % (
                epoch, flag.replace('_', ''), validation_accuracy.avg, validation_loss.avg))


def calculate_accuracy(acc, acc_labels, ang, ang_labels, vel, velocity_labels, lane, lane_labels):
    vel_accuracy, acc_accuracy, ang_accuracy, lane_accuracy = None, None, None, None
    if config.fit_acc or config.fit_action or config.fit_all:
        acc_accuracy = get_accuracy(acc, acc_labels, topk=(1,))[0]
    if config.fit_ang or config.fit_action or config.fit_all:
        ang_accuracy = get_accuracy(ang, ang_labels, topk=(1, 2))[1]
    if global_config.use_vel_head:
        if config.fit_vel or config.fit_action or config.fit_all:
            vel_accuracy = get_accuracy(vel, velocity_labels, topk=(1, 2))[1]
    # print("lane: {}, lane_labels: {}".format(lane, lane_labels), flush=True)

    if config.fit_lane or config.fit_action or config.fit_all:
        lane_accuracy = get_accuracy(lane, lane_labels, topk=(1,))[0]

    return ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy


def calculate_mdn_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                           ang_pi, ang_mu, ang_sigma, ang_labels,
                           vel_pi, vel_mu, vel_sigma, velocity_labels,
                           lane, lane_labels):
    vel_accuracy, acc_accuracy, ang_accuracy, lane_accuracy = None, None, None, None
    if ang_pi is not None:
        ang_accuracy = mdn.mdn_accuracy(ang_pi, ang_sigma, ang_mu, ang_labels)
    if acc_pi is not None:
        acc_accuracy = mdn.mdn_accuracy(acc_pi, acc_sigma, acc_mu, acc_labels)
    if global_config.use_vel_head:
        vel_accuracy = mdn.mdn_accuracy(vel_pi, vel_sigma, vel_mu, velocity_labels)
    if config.fit_lane or config.fit_action or config.fit_all:
        lane_accuracy = get_accuracy(lane, lane_labels, topk=(1,))[0]

    return ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy


def calculate_hybrid_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                              ang, ang_labels,
                              vel_pi, vel_mu, vel_sigma, velocity_labels,
                              lane, lane_labels):
    vel_accuracy, acc_accuracy, ang_accuracy, lane_accuracy = None, None, None, None
    if ang is not None:
        ang_accuracy = get_accuracy(ang, ang_labels, topk=(1, 2))[1]
    if acc_pi is not None:
        acc_accuracy = mdn.mdn_accuracy(acc_pi, acc_sigma, acc_mu, acc_labels)
    if global_config.use_vel_head:
        vel_accuracy = mdn.mdn_accuracy(vel_pi, vel_sigma, vel_mu, velocity_labels)
    if config.fit_lane or config.fit_action or config.fit_all:
        lane_accuracy = get_accuracy(lane, lane_labels, topk=(1,))[0]

    return ang_accuracy, acc_accuracy, vel_accuracy, lane_accuracy


def record_loss_and_accuracy(batch_data, loss_dict, accuracy_dict,
                             loss_recorders, accuracy_recorders):
    # print('loss_recorders.keys()={}'.format(loss_recorders.keys()))
    for idx, flag in enumerate(loss_dict):
        # print("flag {}, loss_item {}".format(flag, loss_item), flush=True)
        loss_item = loss_dict[flag]
        if loss_item[1] > 0:
            loss_recorders[flag].update(float(loss_item[0]), batch_data.size(0))
    for idx, flag in enumerate(accuracy_dict):
        accuracy_item = accuracy_dict[flag]
        if accuracy_item[1] > 0:
            accuracy_recorders[flag].update(float(accuracy_item[0]), batch_data.size(0))


def choose_loss(acc_loss, ang_loss, vel_loss, lane_loss, v_loss, config):
    loss_dict = {
        'combined': (0.0, 1.0),
        'steer': (0.0, 0.0),
        'acc': (0.0, 0.0),
        'vel': (0.0, 0.0),
        'lane': (0.0, 0.0),
        'val': (0.0, 0.0),
    }
    if config.fit_ang or config.fit_action or config.fit_all:
        # print('recording steer loss')
        loss_dict['steer'] = (ang_loss, config.ang_scale)
    if config.fit_acc or config.fit_action or config.fit_all:
        loss_dict['acc'] = (acc_loss, config.acc_scale)
    if config.use_vel_head and (config.fit_vel or config.fit_action or config.fit_all):
        loss_dict['vel'] = (vel_loss, config.vel_scale)
        # print('recording vel loss')
    if config.fit_lane or config.fit_action or config.fit_all:
        loss_dict['lane'] = (lane_loss, config.lane_scale)
    if config.fit_val or config.fit_all:
        loss_dict['val'] = (v_loss, config.val_scale)

    total_loss = 0.0
    total_weight = 0.0
    for idx, flag in enumerate(loss_dict):
        loss_item = loss_dict[flag]
        if flag != "combined":
            total_loss += loss_item[0]
            total_weight += loss_item[1]

    loss_dict['combined'] = (total_loss / total_weight, 1.0)

    # print('loss dict: {}'.format(loss_dict))
    return loss_dict


def choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, lane_accuracy, config):
    accuracy_dict = {
        'combined': (0.0, 1.0),
        'steer': (0.0, 0.0),
        'acc': (0.0, 0.0),
        'vel': (0.0, 0.0),
        'lane': (0.0, 0.0),
    }

    if config.fit_ang or config.fit_action or config.fit_all:
        accuracy_dict['steer'] = (ang_accuracy, config.ang_scale)
    if config.fit_acc or config.fit_action or config.fit_all:
        accuracy_dict['acc'] = (acc_accuracy, config.acc_scale)
    if config.use_vel_head and (config.fit_vel or config.fit_action or config.fit_all):
        accuracy_dict['vel'] = (vel_accuracy, config.vel_scale)
    if config.fit_lane or config.fit_action or config.fit_all:
        accuracy_dict['lane'] = (lane_accuracy, config.lane_scale)

    total_accuracy = 0.0
    total_weight = 0.0
    for idx, flag in enumerate(accuracy_dict):
        accuracy_item = accuracy_dict[flag]
        if flag != 'combined':
            total_accuracy += accuracy_item[0]
            total_weight += accuracy_item[1]

    accuracy_dict['combined'] = (total_accuracy / total_weight, 1.0)
    return accuracy_dict


def debug_data(data_point):
    car_channel = 0

    print("max values in last ped:")
    print("map maximum: %f" % torch.max(data_point[car_channel, 0]))
    print("map sum: %f" % torch.sum(data_point[car_channel, 0]))


def get_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        if len(target.size()) != 1:
            _, target = target.max(dim=1)  # cast target from distribution to class index
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, modelfile):
    torch.save(state, modelfile)
    if is_best:
        pass


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str,
                        default=None, help='Path to train file')
    parser.add_argument('--val', type=str,
                        default=None, help='Path to val file')
    parser.add_argument('--imsize',
                        type=int,
                        default=config.imsize,
                        help='Size of input image')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,
                        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--epochs',
                        type=int,
                        default=config.num_epochs,
                        help='Number of epochs to train')
    parser.add_argument('--moreepochs',
                        type=int,
                        default=-1,
                        help='Number of more epochs to train (only used when in resume mode)')
    parser.add_argument('--start_epoch',
                        type=int,
                        default=0,
                        help='start epoch for training')
    parser.add_argument('--k',
                        type=int,
                        default=config.num_gppn_iterations,  # 10
                        help='Number of Value Iterations')
    parser.add_argument('--f',
                        type=int,
                        default=config.gppn_kernelsize,
                        help='Number of Value Iterations')
    parser.add_argument('--l_i',
                        type=int,
                        default=config.num_gppn_inputs,
                        help='Number of channels in input layer')
    parser.add_argument('--l_h',
                        type=int,
                        default=config.num_gppn_hidden_channels,  # 150
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q',
                        type=int,
                        default=10,
                        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--l2',
                        type=float,
                        default=global_config.l2_reg_weight,
                        help='Weight decay')
    parser.add_argument('--no_action',
                        type=int,
                        default=4,
                        help='Number of actions')
    parser.add_argument('--modelfile',
                        type=str,
                        default="drive_net_params",
                        help='Name of model file to be saved')
    parser.add_argument('--exactname',
                        type=int,
                        default=0,
                        help='Use exact model file name as given')
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Name of model file to be loaded for resuming training')
    parser.add_argument('--no_vin',
                        type=int,
                        default=int(config.vanilla_resnet),
                        help='Number of pedistrians')
    parser.add_argument('--fit',
                        type=str,
                        default='action',
                        help='Number of pedistrians')
    parser.add_argument('--w',
                        type=int,
                        default=config.resnet_width,
                        help='ResNet Width')
    parser.add_argument('--vinout',
                        type=int,
                        default=config.gppn_out_channels,
                        help='ResNet Width')
    parser.add_argument('--nres',
                        type=int,
                        default=config.Num_resnet_layers,
                        help='Number of resnet layers')
    parser.add_argument('--goalfile',
                        type=str,
                        default="../../Maps/indian_cross_goals_15_10_15.txt",
                        help='Name of model file to be saved')
    parser.add_argument('--ssm',
                        type=float,
                        default=config.sigma_smoothing,
                        help='smoothing scalar added on to sigma predictions in mdn heads')
    parser.add_argument('--trainpath',
                        type=str,
                        default='',
                        help='path for the set of training h5 files')
    parser.add_argument('--logdir',
                        type=str,
                        default='runs/**CURRENT_DATETIME_HOSTNAME**',
                        help='path for putting tensor board logs')
    parser.add_argument('--goalx',
                        type=float,
                        default=config.car_goal[0],
                        help='goal_x for car')
    parser.add_argument('--goaly',
                        type=float,
                        default=config.car_goal[1],
                        help='goal_y for car')
    parser.add_argument('--v_scale',
                        type=float,
                        default=config.val_scale,
                        help='scale up of value loss')
    parser.add_argument('--ang_scale',
                        type=float,
                        default=config.ang_scale,
                        help='scale up of value loss')
    parser.add_argument('--do_p',
                        type=float,
                        default=config.do_prob,
                        help='drop out prob')
    parser.add_argument('--input_model',
                        type=str,
                        default='',
                        help='[model conversion] Input model in pth format')
    parser.add_argument('--output_model',
                        type=str,
                        default="torchscript_version.pt",
                        help='[model conversion] Output model in pt format')
    parser.add_argument('--monitor',
                        type=str,
                        default="data_monitor",
                        help='which data monitor to use: data_monitor or summit_dql')

    return parser.parse_args()


def update_global_config(cmd_args):
    print("\n=========================== Command line arguments ==========================")
    for arg in vars(cmd_args):
        print("==> {}: {}".format(arg, getattr(cmd_args, arg)))
    print("=========================== Command line arguments ==========================\n")

    # Update the global configurations according to command line
    config.l2_reg_weight = cmd_args.l2
    config.vanilla_resnet = bool(cmd_args.no_vin)
    config.num_gppn_inputs = cmd_args.l_i
    config.num_gppn_hidden_channels = cmd_args.l_h
    config.gppn_kernelsize = cmd_args.f
    config.num_gppn_iterations = cmd_args.k
    config.resnet_width = cmd_args.w
    config.gppn_out_channels = cmd_args.vinout
    config.sigma_smoothing = cmd_args.ssm
    print("Sigma smoothing " + str(cmd_args.ssm))
    config.train_set_path = cmd_args.trainpath
    config.Num_resnet_layers = cmd_args.nres

    if not config.train_set_path == '':
        config.sample_mode = 'hierarchical'
    else:
        config.sample_mode = 'random'

    print("Using sampling mode " + str(config.sample_mode))

    config.car_goal[0] = float(cmd_args.goalx)
    config.car_goal[1] = float(cmd_args.goaly)
    config.val_scale = cmd_args.v_scale
    config.ang_scale = cmd_args.ang_scale
    config.do_prob = cmd_args.do_p

    set_fit_mode_bools(cmd_args)

    if cmd_args.train:
        if 'stateactions.h5' in cmd_args.train:
            reset_global_params_for_gamma_dataset(cmd_args)
        elif 'train.h5' in cmd_args.train:
            reset_global_params_for_pomdp_dataset(cmd_args)
    else: # test.py
        if cmd_args.monitor == 'data_monitor':
            reset_global_params_for_pomdp_dataset(cmd_args)
        elif cmd_args.monitor == 'summit_dql':
            reset_global_params_for_gamma_dataset(cmd_args)
            config.data_type = np.uint8
            config.control_freq = 10
        else:
            error_handler("unsupported data monitor")

    print("Fitting " + cmd_args.fit)

    print("\n=========================== Global configuration ==========================")
    for arg in vars(config):
        print("===> {}: {}".format(arg, getattr(config, arg)))
    print("=========================== Global configuration ==========================\n")


def set_fit_mode_bools(cmd_args):
    if cmd_args.fit == 'action':
        config.fit_action = True
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = False
        config.fit_all = False
    elif cmd_args.fit == 'acc':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = True
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = False
        config.fit_all = False
    elif cmd_args.fit == 'steer':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = True
        config.fit_val = False
        config.fit_lane = False
        config.fit_all = False
    elif cmd_args.fit == 'vel':
        config.fit_action = False
        config.fit_vel = True
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = False
        config.fit_all = False
        config.use_vel_head = True
    elif cmd_args.fit == 'lane':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = True
        config.fit_all = False
    elif cmd_args.fit == 'val':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = True
        config.fit_lane = False
        config.fit_all = False
    elif cmd_args.fit == 'all':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = False
        config.fit_all = True


def set_file_names():
    global cmd_args
    train_file_name = cmd_args.train.split('.')[0]
    val_file_name = cmd_args.val.split('.')[0]

    model_dir = "trained_models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if model_dir in cmd_args.modelfile:
        model_dir = ''

    if cmd_args.exactname > 0:
        cmd_args.modelfile = model_dir + cmd_args.modelfile
    else:
        cmd_args.modelfile = model_dir + cmd_args.modelfile + '_' + \
                             time.strftime("%Y_%m_%d_%I_%M_%S") + \
                             '_t_' + train_file_name + \
                             '_v_' + val_file_name + \
                             '_vanilla_' + str(global_config.vanilla_resnet) + \
                             '_k_' + str(cmd_args.k) + \
                             '_lh_' + str(cmd_args.l_h) + \
                             '_nped_' + str(0) + \
                             '_gppn_o_' + str(global_config.gppn_out_channels) + \
                             '_nres_' + str(global_config.Num_resnet_layers) + \
                             '_w_' + str(global_config.resnet_width) + \
                             '_bn_' + str(not global_config.disable_bn_in_resnet) + \
                             '_layers_' + str(global_config.resblock_in_layers[0]) + \
                             str(global_config.resblock_in_layers[1]) + \
                             str(global_config.resblock_in_layers[2]) + \
                             str(global_config.resblock_in_layers[3]) + \
                             "_lr_" + str(cmd_args.lr) + \
                             "_ada_" + str(global_config.enable_scheduler) + \
                             "_l2_" + str(global_config.l2_reg_weight) + \
                             "_bs_" + str(cmd_args.batch_size) + \
                             "_aug_" + str(global_config.augment_data) + \
                             "_do_" + str(global_config.do_dropout) + \
                             "_bn_stat_" + str(global_config.track_running_stats) + \
                             "_sm_labels_" + str(global_config.label_smoothing) + \
                             ".pth"
    print("using model file name: " + cmd_args.modelfile)

    return train_file_name, val_file_name


def resume_partial_model(pretrained_dict, net):
    model_dict = net.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(model_dict)


def resume_model():
    global best_val_loss
    if os.path.isfile(cmd_args.resume):
        print("=> loading checkpoint '{}'".format(cmd_args.resume))
        checkpoint = torch.load(cmd_args.resume)

        cmd_args.start_epoch = checkpoint['epoch'] + 1

        if cmd_args.resume and cmd_args.moreepochs > 0:
            cmd_args.epochs = cmd_args.start_epoch + cmd_args.moreepochs

        best_val_loss = checkpoint['best_prec1']
        load_settings_from_model(checkpoint, global_config, cmd_args)
        resume_partial_model(checkpoint['state_dict'], net)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cmd_args.lr,
                               weight_decay=config.l2_reg_weight)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, threshold=1e-2, factor=lr_factor,
                                                         verbose=True)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(cmd_args.resume, checkpoint['epoch']))
        print("=> using learning rate {}".format(cmd_args.lr))

        return optimizer, scheduler
    else:
        print("=> '{}' does not exist! ".format(cmd_args.resume))
        return None, None


def load_settings_from_model(checkpoint, config, cmd_args):
    try:
        # config.head_mode = checkpoint['head_mode']
        config.vanilla_resnet = checkpoint['novin']
        config.num_gppn_hidden_channels = checkpoint['l_h']
        cmd_args.l_h = checkpoint['l_h']
        config.gppn_out_channels = checkpoint['vinout']
        config.resnet_width = checkpoint['w']
        config.resblock_in_layers = checkpoint['layers']
        # config.fit_ang = checkpoint['fit_ang']
        # config.fit_acc = checkpoint['fit_acc']
        # config.fit_vel = checkpoint['fit_vel']
        # config.fit_lane = checkpoint['fit_lane']
        # config.fit_val = checkpoint['fit_val']
        # config.fit_action = checkpoint['fit_action']
        # config.fit_all = checkpoint['fit_all']
        config.sigma_smoothing = checkpoint['ssm']
        config.track_running_stats = checkpoint['runn_stat']
        config.do_prob = checkpoint['do_p']

    except Exception as e:
        print(e)
    finally:
        print("=> model at epoch {}"
              .format(checkpoint['epoch']))
        print("=> best val loss {}"
              .format(checkpoint['best_prec1']))
        print("=> head mode {}"
              .format(global_config.head_mode))
        print("=> vanilla resnet {}"
              .format(global_config.vanilla_resnet))
        print("=> width of vin {}"
              .format(global_config.num_gppn_hidden_channels))
        print("=> out channels of vin {}"
              .format(global_config.gppn_out_channels))
        print("=> resnet width {}"
              .format(global_config.resnet_width))
        print("=> res layers {}"
              .format(global_config.resblock_in_layers))
        print("=> mdn sigma smoothing {}"
              .format(global_config.sigma_smoothing))
        print("=> track running status {}"
              .format(global_config.track_running_stats))


def save_final_model(epoch):
    global best_val_loss
    global cur_val_loss
    if cur_val_loss > best_val_loss:
        print("Saving final model...")

        save_checkpoint(make_checkpoint_dict(epoch, best_val_loss, global_config, cmd_args), True, cmd_args.modelfile)
    else:
        print("Saving final model...")
        model_file = cmd_args.modelfile.split('.')[0] + '_cur' + '.pth'
        save_checkpoint(make_checkpoint_dict(epoch, best_val_loss, global_config, cmd_args), True, model_file)


def make_checkpoint_dict(epoch, best_val_loss, config, cmd_args):
    return {
        'epoch': epoch,
        'novin': config.vanilla_resnet,
        'l_h': config.num_gppn_hidden_channels,
        'vinout': config.gppn_out_channels,
        'w': config.resnet_width,
        'layers': config.resblock_in_layers,
        'head_mode': config.head_mode,
        'do_p': config.do_prob,
        'fit_ang': config.fit_ang,
        'fit_acc': config.fit_acc,
        'fit_vel': config.fit_vel,
        'fit_val': config.fit_val,
        'fit_lane': config.fit_lane,
        'fit_action': config.fit_action,
        'fit_all': config.fit_all,
        'ssm': config.sigma_smoothing,
        'runn_stat': config.track_running_stats,
        'state_dict': net.state_dict(),
        'best_prec1': best_val_loss,
        'optimizer': optimizer.state_dict(),
    }


if __name__ == '__main__':
    config = global_params.config

    # Parsing training parameters
    cmd_args = parse_cmd_args()
    update_global_config(cmd_args)

    writer = SummaryWriter(log_dir=cmd_args.logdir)

    train_filename, val_filename = set_file_names()

    # Instantiate the NN model
    net = nn.DataParallel(PolicyValueNet(cmd_args), device_ids=config.GPU_devices).to(device)

    # Loss
    criterion = nn.MSELoss().to(device)
    # Cross Entropy Loss criterion
    if config.label_smoothing:
        cel_criterion = SoftCrossEntropy().to(device)
    else:
        cel_criterion = nn.CrossEntropyLoss().to(device)
        # cel_criterion = CELossWithMaxEntRegularizer().to(device)
    # Optimizer: weight_decay is the scaling factor for L2 regularization
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cmd_args.lr,
                           weight_decay=config.l2_reg_weight)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, threshold=1e-2, factor=lr_factor,
                                                     verbose=True)

    if cmd_args.resume:
        optimizer, scheduler = resume_model()
    else:
        best_val_loss = None

    # Define Dataset
    if 'stateactions.h5' in cmd_args.train:
        train_set = GammaDataset(filename=cmd_args.train, start=0.0, end=0.7)
        val_set = GammaDataset(filename=cmd_args.train, start=0.7, end=1.0)
    else:
        train_set = DrivingData(filename=cmd_args.train, flag="Training")
        val_set = DrivingData(filename=cmd_args.val, flag="Validation")
    # Create Dataloader
    do_shuffle_training = False
    if config.sample_mode == 'hierarchical':
        do_shuffle_training = False
    else:
        do_shuffle_training = True
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cmd_args.batch_size, shuffle=do_shuffle_training, num_workers=config.num_data_loaders)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cmd_args.batch_size, shuffle=True, num_workers=config.num_data_loaders)

    # Train the model
    if cmd_args.resume:
        init_loss, init_accuracy = evaluate(10000, -1)  # check the untrained model
        record_tensor_board_log(-1, None, None, init_accuracy, init_loss)

    train()

    save_final_model(cmd_args.epochs)

    # visualize in tensor board
    writer.export_scalars_to_json("./all_losses.json")
    writer.close()
