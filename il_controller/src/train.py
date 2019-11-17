import time
import os
import argparse
from dataset import *
from policy_value_network import *
import torch.optim as optim

from tensorboardX import SummaryWriter
from Data_processing import global_params
from tqdm import tqdm
import torch.nn.functional as F
from Components.nan_police import *
from visualization import *

model_device_id = global_config.GPU_devices[0]
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")

best_val_loss = None
cur_val_loss = None

lr_factor = 0.3
last_train_loss = 0.0
first_val_loss = 0.0
acc_pi, acc_mu, acc_sigma, \
ang_pi, ang_mu, ang_sigma, \
vel_pi, vel_mu, vel_sigma = None, None, None, None, None, None, None, None, None

acc_pi_train, acc_mu_train, acc_sigma_train, \
ang_pi_train, ang_mu_train, ang_sigma_train, \
vel_pi_train, vel_mu_train, vel_sigma_train = None, None, None, None, None, None, None, None, None

ang_train = None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.last = 0

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


train_flag = 'train/'
val_flag = 'val/'


def train():
    sm = nn.Softmax(dim=1)

    clear_png_files('./visualize/', remove_flag='train_')
    clear_png_files('./visualize/', remove_flag='val_')

    print("Enter training")
    global best_val_loss
    global cur_val_loss
    global acc_pi, acc_mu, acc_sigma, \
        ang_pi, ang_mu, ang_sigma, \
        vel_pi, vel_mu, vel_sigma

    global acc_pi_train, acc_mu_train, acc_sigma_train
    global ang_pi_train, ang_mu_train, ang_sigma_train
    global vel_pi_train, vel_mu_train, vel_sigma_train

    for epoch in range(cmd_args.start_epoch, cmd_args.epochs):  # Loop over dataset multiple times
        # setting seed for random generator
        current_time = int(round(time.time() * 1000))
        torch.cuda.manual_seed_all(current_time)

        epoch_accuracy, epoch_loss = alloc_recorders()

        net.train()

        last_data = None
        last_loss = 0.0

        with tqdm(total=len(train_loader.dataset), ncols=100) as pbar:
            for i, data_point in enumerate(train_loader):  # Loop over batches of data

                visualize_results = False
                if i <= 0:  # 0 -> 20
                    visualize_results = True
                    # print('step:', i)

                # Get input batch
                X, v_labels, acc_labels, ang_labels, vel_labels = data_point
                X, v_labels, acc_labels, ang_labels, vel_labels = X.to(
                    device), v_labels.to(device), acc_labels.to(device), ang_labels.to(device), vel_labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                if global_config.head_mode == "mdn":
                    # Forward pass
                    acc_pi, acc_mu, acc_sigma, \
                    ang_pi, ang_mu, ang_sigma, \
                    vel_pi, vel_mu, vel_sigma, value = forward_pass(X, print_time=visualize_results,
                                                                    image_flag=train_flag, step=epoch)

                    visualize_mdn_predictions(epoch, acc_labels, acc_mu, acc_pi, acc_sigma, ang_labels, ang_mu, ang_pi,
                                              ang_sigma, vel_labels, vel_mu, vel_pi, vel_sigma, visualize_results,
                                              train_flag)

                    # Loss
                    acc_loss, ang_loss, v_loss, vel_loss = \
                        calculate_mdn_loss(acc_pi, acc_mu, acc_sigma, acc_labels,
                                           ang_pi, ang_mu, ang_sigma, ang_labels,
                                           vel_pi, vel_mu, vel_sigma, vel_labels,
                                           value, v_labels)

                    loss = choose_loss(acc_loss, ang_loss, vel_loss, v_loss, global_config)

                    # accuracy
                    ang_accuracy, acc_accuracy, vel_accuracy = \
                        calculate_mdn_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                                               ang_pi, ang_mu, ang_sigma, ang_labels,
                                               vel_pi, vel_mu, vel_sigma, vel_labels)

                    accuracy = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, global_config)

                    record_loss_and_accuracy(X, loss, accuracy, epoch_loss, epoch_accuracy)

                elif global_config.head_mode == "hybrid":
                    # Forward pass
                    acc_pi, acc_mu, acc_sigma, \
                    ang, \
                    vel_pi, vel_mu, vel_sigma, value = forward_pass(X, print_time=visualize_results,
                                                                    image_flag=train_flag, step=epoch)  # epoch

                    # epoch -> i
                    visualize_hybrid_predictions(epoch, acc_labels, acc_mu, acc_pi, acc_sigma, ang, ang_labels,
                                                 vel_labels, value, v_labels,
                                                 visualize_results, sm, train_flag)

                    # Loss
                    acc_loss, ang_loss, v_loss, vel_loss = \
                        calculate_hybrid_loss(acc_pi, acc_mu, acc_sigma, acc_labels,
                                              ang, ang_labels,
                                              vel_pi, vel_mu, vel_sigma, vel_labels,
                                              value, v_labels)

                    loss = choose_loss(acc_loss, ang_loss, vel_loss, v_loss, global_config)

                    # accuracy
                    ang_accuracy, acc_accuracy, vel_accuracy = \
                        calculate_hybrid_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                                                  ang, ang_labels,
                                                  vel_pi, vel_mu, vel_sigma, vel_labels)

                    accuracy = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, global_config)

                    record_loss_and_accuracy(X, loss, accuracy, epoch_loss, epoch_accuracy)

                else:
                    # Forward pass
                    acc, ang, value, vel = forward_pass(X, print_time=visualize_results, image_flag=train_flag,
                                                        step=epoch)

                    visualize_predictions(acc, acc_labels, ang, ang_labels, epoch, vel, vel_labels, visualize_results,
                                          sm, train_flag)

                    # Loss
                    acc_loss, ang_loss, v_loss, vel_loss = \
                        calculate_loss(acc, acc_labels, ang, ang_labels, vel, vel_labels, value, v_labels)

                    loss = choose_loss(acc_loss, ang_loss, vel_loss, v_loss, global_config)

                    # accuracy

                    ang_accuracy, acc_accuracy, vel_accuracy = \
                        calculate_accuracy(acc, acc_labels, ang, ang_labels, vel, vel_labels)

                    accuracy = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, global_config)

                    record_loss_and_accuracy(X, loss, accuracy, epoch_loss, epoch_accuracy)

                last_loss = loss[0]

                # Backward pass
                loss[0].backward()

                optimizer.step()
                pbar.update(X.size(0))

                last_data = X

        # record data for debugging
        if config.head_mode == 'hybrid':
            acc_pi_train, acc_mu_train, acc_sigma_train, \
            ang_train, \
            vel_pi_train, vel_mu_train, vel_sigma_train, _ = forward_pass(last_data)
        elif config.head_mode == 'mdn':
            acc_pi_train, acc_mu_train, acc_sigma_train, \
            ang_pi_train, ang_mu_train, ang_sigma_train, \
            vel_pi_train, vel_mu_train, vel_sigma_train, _ = forward_pass(last_data)
        # print("Last mdn accuracy (train): %f" % epoch_accuracy.last)

        # Evaluation
        validation_loss, validation_accuracy = evaluate(last_loss, epoch)

        # Schedule learning rate
        if global_config.enable_scheduler:
            scheduler.step(validation_loss[0].avg)

        # record the data for visualization in tensorboard
        record_tb_data_list(epoch, epoch_accuracy, epoch_loss, validation_accuracy, validation_loss)

        cur_val_loss = validation_loss[0].avg

        save_model(epoch)

        if termination():
            break

    print('\nFinished training. \n')


def evaluate(last_train_loss, epoch):
    start_time = time.time()

    sm = nn.Softmax(dim=1)

    epoch_accuracy_val, epoch_loss_val = alloc_recorders()

    global acc_pi, acc_mu, acc_sigma, \
        ang_pi, ang_mu, ang_sigma, \
        vel_pi, vel_mu, vel_sigma

    global acc_pi_train, acc_mu_train, acc_sigma_train
    global ang_pi_train, ang_mu_train, ang_sigma_train
    global vel_pi_train, vel_mu_train, vel_sigma_train

    net.eval()
    # print("Evaluation")

    first_loss = None

    for i, data_point in enumerate(val_loader):

        visualize_results = False
        if i <= 0:
            visualize_results = True

        with torch.no_grad():
            # Get input batch
            X, v_labels, acc_labels, ang_labels, vel_labels = data_point
            if global_config.visualize_val_data:
                debug_data(X[1])
            X, v_labels, acc_labels, ang_labels, vel_labels = X.to(device), v_labels.to(
                device), acc_labels.to(device), ang_labels.to(device), vel_labels.to(device)

            if global_config.head_mode == "mdn":
                # Forward pass
                acc_pi, acc_mu, acc_sigma, \
                ang_pi, ang_mu, ang_sigma, \
                vel_pi, vel_mu, vel_sigma, \
                value = forward_pass(X, print_time=visualize_results, image_flag=val_flag, step=epoch)

                visualize_mdn_predictions(epoch, acc_labels, acc_mu, acc_pi, acc_sigma, ang_labels, ang_mu, ang_pi,
                                          ang_sigma, vel_labels, vel_mu, vel_pi, vel_sigma, visualize_results, val_flag)

                # Loss
                acc_loss, ang_loss, v_loss, vel_loss = \
                    calculate_mdn_loss(acc_pi, acc_mu, acc_sigma, acc_labels,
                                       ang_pi, ang_mu, ang_sigma, ang_labels,
                                       vel_pi, vel_mu, vel_sigma, vel_labels,
                                       value, v_labels)

                loss = choose_loss(acc_loss, ang_loss, vel_loss, v_loss, global_config)

                # Accuracy
                ang_accuracy, acc_accuracy, vel_accuracy = \
                    calculate_mdn_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                                           ang_pi, ang_mu, ang_sigma, ang_labels,
                                           vel_pi, vel_mu, vel_sigma, vel_labels)

                accuracy = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, global_config)

                # Statistics
                record_loss_and_accuracy(X, loss, accuracy, epoch_loss_val, epoch_accuracy_val)

            elif global_config.head_mode == "hybrid":

                start = time.time()
                # Forward pass
                acc_pi, acc_mu, acc_sigma, \
                ang, \
                vel_pi, vel_mu, vel_sigma, value = forward_pass(X, print_time=visualize_results, image_flag=val_flag,
                                                                step=epoch)

                # if value:
                #     value.cpu()

                end = time.time()

                # print("Evaluate mode forward time for batchsize " + str(cmd_args.batch_size) + ": " + str(end - start) + ' s')

                visualize_hybrid_predictions(epoch, acc_labels, acc_mu, acc_pi, acc_sigma, ang, ang_labels, vel_labels,
                                             value, v_labels,
                                             visualize_results, sm, val_flag)

                # Loss
                acc_loss, ang_loss, v_loss, vel_loss = \
                    calculate_hybrid_loss(acc_pi, acc_mu, acc_sigma, acc_labels,
                                          ang, ang_labels,
                                          vel_pi, vel_mu, vel_sigma, vel_labels,
                                          value, v_labels)

                loss = choose_loss(acc_loss, ang_loss, vel_loss, v_loss, global_config)

                # Accuracy
                ang_accuracy, acc_accuracy, vel_accuracy = \
                    calculate_hybrid_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                                              ang, ang_labels,
                                              vel_pi, vel_mu, vel_sigma, vel_labels)

                accuracy = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, global_config)

                record_loss_and_accuracy(X, loss, accuracy, epoch_loss_val, epoch_accuracy_val)

            else:
                # Forward pass
                acc, ang, value, vel = forward_pass(X, print_time=visualize_results, image_flag=val_flag, step=epoch)

                visualize_predictions(acc, acc_labels, ang, ang_labels, epoch, vel, vel_labels, visualize_results, sm,
                                      val_flag)

                # Loss
                acc_loss, ang_loss, v_loss, vel_loss = calculate_loss(acc, acc_labels, ang, ang_labels,
                                                                      vel, vel_labels, value, v_labels)

                loss = choose_loss(acc_loss, ang_loss, vel_loss, v_loss, global_config)

                # Accuracy
                ang_accuracy, acc_accuracy, vel_accuracy = calculate_accuracy(acc, acc_labels, ang, ang_labels,
                                                                              vel, vel_labels)

                accuracy = choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, global_config)

                # Statistics
                record_loss_and_accuracy(X, loss, accuracy, epoch_loss_val, epoch_accuracy_val)

            if i == 0:
                first_loss = loss[0]

    if abs(first_loss - last_train_loss) >= 4:
        print("Last mdn accuracy (val): %f" % epoch_accuracy_val[0].last)

    inference_time = time.time() - start_time

    print("Evaluation time %fs" % inference_time)

    return epoch_loss_val, epoch_accuracy_val


def visualize_mdn_predictions(epoch, acc_labels, acc_mu, acc_pi, acc_sigma, ang_labels, ang_mu, ang_pi, ang_sigma,
                              vel_labels, vel_mu, vel_pi, vel_sigma, visualize_results, flag):
    if visualize_results:
        try:
            if config.use_vel_head:
                visualize_mdn_output_with_labels(flag + str(epoch), acc_mu[0].detach(), acc_pi[0].detach(),
                                                 acc_sigma[0].detach(), ang_mu[0].detach(),
                                                 ang_pi[0].detach(), ang_sigma[0].detach(), vel_mu[0].detach(),
                                                 vel_pi[0].detach(), vel_sigma[0].detach(), acc_labels[0].detach(),
                                                 ang_labels[0].detach(),
                                                 vel_labels[0].detach())
            else:

                if acc_pi is None:
                    acc_pi = torch.zeros(1, global_config.num_guassians_in_heads)
                    acc_mu = torch.zeros(1, global_config.num_guassians_in_heads, 1)
                    acc_sigma = torch.zeros(1, global_config.num_guassians_in_heads, 1)
                if ang_pi is None:
                    ang_pi = torch.zeros(1, global_config.num_guassians_in_heads)
                    ang_mu = torch.zeros(1, global_config.num_guassians_in_heads, 1)
                    ang_sigma = torch.zeros(1, global_config.num_guassians_in_heads, 1)

                visualize_mdn_output_with_labels(flag + str(epoch), acc_mu[0].detach(), acc_pi[0].detach(),
                                                 acc_sigma[0].detach(), ang_mu[0].detach(),
                                                 ang_pi[0].detach(), ang_sigma[0].detach(), None,
                                                 None, None, acc_labels[0].detach(),
                                                 ang_labels[0].detach(),
                                                 vel_labels[0].detach())
        except Exception as e:
            print(e)
            exit(1)


def visualize_hybrid_predictions(epoch, acc_labels, acc_mu, acc_pi, acc_sigma, ang, ang_labels, vel_labels,
                                 value, v_labels,
                                 visualize_results, sm, flag):
    if visualize_results:

        if acc_pi is None:
            acc_pi = torch.zeros(1, global_config.num_guassians_in_heads)
            acc_mu = torch.zeros(1, global_config.num_guassians_in_heads, 1)
            acc_sigma = torch.zeros(1, global_config.num_guassians_in_heads, 1)
        if ang is None:
            ang = torch.zeros(1, global_config.num_steering_bins)

        ang_probs = sm(ang)

        for i in range(0, min(10, cmd_args.batch_size)):
            if value is None:
                value_vis = None
            else:
                value_vis = value[i].detach()

            visualize_hybrid_output_with_labels(flag + str(epoch) + '_' + str(i),
                                            acc_mu[i].detach(), acc_pi[i].detach(), acc_sigma[i].detach(),
                                            ang_probs[i].detach(),
                                            None, None, None,
                                            value_vis, 
                                            acc_labels[i].detach(), ang_labels[i].detach(),vel_labels[i].detach(),
                                            v_labels[i])


def visualize_predictions(acc, acc_labels, ang, ang_labels, epoch, vel, vel_labels, visualize_results, sm, flag):
    if visualize_results:
        acc_probs = sm(acc)
        ang_probs = sm(ang)
        if config.use_vel_head:
            vel_probs = sm(vel)
            visualize_output_with_labels(flag + str(epoch), acc_probs[0].detach(), ang_probs[0].detach(),
                                     vel_probs[0].detach(),
                                     acc_labels[0].detach(),
                                     ang_labels[0].detach(), vel_labels[0].detach())
        else:

            visualize_output_with_labels(flag + str(epoch), acc_probs[0].detach(), ang_probs[0].detach(),
                                         None,
                                         acc_labels[0].detach(),
                                         ang_labels[0].detach(), None)


def alloc_recorders():
    if global_config.fit_all:
        if config.use_vel_head:
            # one combined,
            # the other four for steer, acc, vel, and val
            epoch_loss = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
            epoch_accuracy = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        else:
            # one combined,
            # the other four for steer, acc, and val
            epoch_loss = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]  # one combined,
            epoch_accuracy = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    elif global_config.fit_action:
        if config.use_vel_head:
            epoch_loss = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]  # one combined,
            # the other three for steer, acc, and vel
            epoch_accuracy = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        else:
            epoch_loss = [AverageMeter(), AverageMeter(), AverageMeter()]  # one combined,
            # the other two for steer and acc
            epoch_accuracy = [AverageMeter(), AverageMeter(), AverageMeter()]
    else:
        epoch_loss = [AverageMeter()]
        epoch_accuracy = [AverageMeter()]
    return epoch_accuracy, epoch_loss


def forward_pass(X, step=0, drive_net=None, cmd_config=None, print_time=False, image_flag=''):
    start_time = time.time()

    if drive_net == None:
        drive_net = net
    if cmd_config == None:
        cmd_config = cmd_args
    ped_vin_out, car_vin_out, res_out, res_image = \
        torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0)
    if global_config.head_mode == "mdn":
        value, acc_pi, acc_mu, acc_sigma, \
        ang_pi, ang_mu, ang_sigma, vel_pi, vel_mu, vel_sigma, car_vin_out, res_image = drive_net.forward(X,
                                                                                                             cmd_config)
        if global_config.print_preds:
            print("predicted angle:")
            print(ang_pi, ang_mu, ang_sigma)

        elapsed_time = time.time() - start_time

        if print_time:
            print("Inference time: " + str(elapsed_time) + " s")
            visualize_input_output(X, ped_vin_out, car_vin_out, res_out, res_image, step, image_flag)

        return acc_pi, acc_mu, acc_sigma, \
               ang_pi, ang_mu, ang_sigma, vel_pi, vel_mu, vel_sigma, value

    elif global_config.head_mode == "hybrid":
        value, acc_pi, acc_mu, acc_sigma, \
        ang, vel_pi, vel_mu, vel_sigma, car_vin_out, res_image = drive_net.forward(X, cmd_config)

        if global_config.print_preds:
            print("predicted angle:")
            print(ang)

        # if value:
        #     value.cpu()

        elapsed_time = time.time() - start_time

        if print_time and config.visualize_inter_data:
            print("Inference time: " + str(elapsed_time) + " s")
            visualize_input_output(X, ped_vin_out, car_vin_out, res_out, res_image, step, image_flag)

        return acc_pi, acc_mu, acc_sigma, \
               ang, vel_pi, vel_mu, vel_sigma, value

    else:
        value, acc, ang, vel, car_vin_out, res_image = drive_net.forward(X, cmd_config)
        if global_config.print_preds:
            print("predicted angle:")
            print(ang)

        elapsed_time = time.time() - start_time
        if print_time:
            print("Inference time: " + str(elapsed_time) + " s")
            visualize_input_output(X, ped_vin_out, car_vin_out, res_out, res_image, step, image_flag)

        return acc, ang, value, vel


def forward_pass_jit(X, step=0, drive_net=None, cmd_config=None, print_time=False, image_flag=''):
    start_time = time.time()

    # print("forward_pass_jit input:", X.squeeze(0))

    if drive_net == None:
        drive_net = net
    if cmd_config == None:
        cmd_config = cmd_args
    ped_vin_out, car_vin_out, res_out, res_image = \
        torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0)

    if global_config.head_mode == "hybrid":
        value, acc_pi, acc_mu, acc_sigma, \
        ang, car_vin_out, res_image = drive_net.forward(X.squeeze(0))

        if global_config.print_preds:
            print("predicted angle:")
            print(ang)

        # if value:
        #     value.cpu()
        elapsed_time = time.time() - start_time
        
        if print_time and config.visualize_inter_data:
            print("Inference time: " + str(elapsed_time) + " s")
            visualize_input_output(X, ped_vin_out, car_vin_out, res_out, res_image, step, image_flag)

        return acc_pi, acc_mu, acc_sigma, \
               ang, vel_pi, vel_mu, vel_sigma, value

    else:
        print("jit model only supports hybird head mode")
        exit(1)


def calculate_loss(acc, acc_labels, ang, ang_labels, vel, vel_labels, value, v_labels):
    v_loss = criterion(value, v_labels)
    vel_loss, acc_loss, ang_loss = None, None, None

    if config.fit_acc or config.fit_action or config.fit_all:
        acc_loss = cel_criterion(acc, acc_labels)
    if config.fit_ang or config.fit_action or config.fit_all:
        ang_loss = cel_criterion(ang, ang_labels)
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            vel_loss = cel_criterion(vel, vel_labels)
    return acc_loss, ang_loss, v_loss, vel_loss


def calculate_mdn_loss(acc_pi, acc_mu, acc_sigma, acc_labels, ang_pi, ang_mu, ang_sigma, ang_labels,
                       vel_pi, vel_mu, vel_sigma, vel_labels, value, v_labels):
    v_loss = criterion(value, v_labels)
    vel_loss, acc_loss, ang_loss = None, None, None

    if config.fit_acc or config.fit_action or config.fit_all:
        acc_loss = calculate_mdn_loss_action(acc_pi, acc_mu, acc_sigma, acc_labels)
    if config.fit_ang or config.fit_action or config.fit_all:
        ang_loss = calculate_mdn_loss_action(ang_pi, ang_mu, ang_sigma, ang_labels)
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            vel_loss = calculate_mdn_loss_action(vel_pi, vel_mu, vel_sigma, vel_labels)
    return acc_loss, ang_loss, v_loss, vel_loss


def calculate_hybrid_loss(acc_pi, acc_mu, acc_sigma, acc_labels, ang, ang_labels,
                          vel_pi, vel_mu, vel_sigma, vel_labels, value, v_labels):

    v_loss, vel_loss, acc_loss, ang_loss = None, None, None, None

    if config.fit_val or config.fit_all:
        v_loss = criterion(value, v_labels)
    if config.fit_acc or config.fit_action or config.fit_all:
        acc_loss = calculate_mdn_loss_action(acc_pi, acc_mu, acc_sigma, acc_labels)
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            vel_loss = calculate_mdn_loss_action(vel_pi, vel_mu, vel_sigma, vel_labels)
    if config.fit_ang or config.fit_action or config.fit_all:
        ang_loss = cel_criterion(ang, ang_labels)

    return acc_loss, ang_loss, v_loss, vel_loss


def calculate_mdn_loss_action(pi, mu, sigma, labels):
    loss = mdn.mdn_loss(pi, sigma, mu, labels)
    return loss


def termination():
    terminal = False
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
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


def record_tb_data_list(epoch, epoch_accuracy_list, epoch_loss_list, validation_accuracy_list, validation_loss_list):
    for i, validation_loss in enumerate(validation_loss_list):
        if epoch_accuracy_list:
            epoch_accuracy = epoch_accuracy_list[i]
        else:
            epoch_accuracy = None
        if epoch_loss_list:
            epoch_loss = epoch_loss_list[i]
        else:
            epoch_loss = None
        validation_accuracy = validation_accuracy_list[i]
        flag = ''
        if i == 0:
            flag = ''
        elif i == 1:
            flag = '_Steer'
        elif i == 2:
            flag = '_Acc'
        elif i == 3:
            if config.use_vel_head:
                flag = '_Vel'
            else:
                flag = '_Val'
        elif i == 4:
            flag = '_Val'

        record_tb_data(epoch, epoch_accuracy, epoch_loss, validation_accuracy, validation_loss, flag)

    current_lr = 0.0
    writer.add_scalars('data/lr', {'lr': current_lr, '0': 0.0}, epoch + 1)


def record_tb_data(epoch, epoch_accuracy, epoch_loss, validation_accuracy, validation_loss, flag):
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


def calculate_accuracy(acc, acc_labels, ang, ang_labels, vel, vel_labels):
    vel_accuracy = None
    ang_accuracy = get_accuracy(ang, ang_labels, topk=(1, 2))[1]
    acc_accuracy = get_accuracy(acc, acc_labels, topk=(1,))[0]
    if global_config.use_vel_head:
        vel_accuracy = get_accuracy(vel, vel_labels, topk=(1, 2))[1]
    return ang_accuracy, acc_accuracy, vel_accuracy


def calculate_mdn_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                           ang_pi, ang_mu, ang_sigma, ang_labels,
                           vel_pi, vel_mu, vel_sigma, vel_labels):
    vel_accuracy,ang_accuracy, acc_accuracy = None, None, None
    if ang_pi is not None:
        ang_accuracy = mdn.mdn_accuracy(ang_pi, ang_sigma, ang_mu, ang_labels)
    if acc_pi is not None:
        acc_accuracy = mdn.mdn_accuracy(acc_pi, acc_sigma, acc_mu, acc_labels)
    if global_config.use_vel_head:
        vel_accuracy = mdn.mdn_accuracy(vel_pi, vel_sigma, vel_mu, vel_labels)
    return ang_accuracy, acc_accuracy, vel_accuracy


def calculate_hybrid_accuracy(acc_pi, acc_mu, acc_sigma, acc_labels,
                              ang, ang_labels,
                              vel_pi, vel_mu, vel_sigma, vel_labels):
    vel_accuracy, ang_accuracy, acc_accuracy = None, None, None
    if ang is not None:
        ang_accuracy = get_accuracy(ang, ang_labels, topk=(1, 2))[1]
    if acc_pi is not None:
        acc_accuracy = mdn.mdn_accuracy(acc_pi, acc_sigma, acc_mu, acc_labels)
    if global_config.use_vel_head:
        vel_accuracy = mdn.mdn_accuracy(vel_pi, vel_sigma, vel_mu, vel_labels)
    return ang_accuracy, acc_accuracy, vel_accuracy


def choose_accuracy(acc_accuracy, ang_accuracy, vel_accuracy, config):
    if config.fit_ang:
        accuracy = [ang_accuracy]
    elif config.fit_acc:
        accuracy = [acc_accuracy]
    elif config.fit_vel:
        accuracy = [vel_accuracy]
    elif config.fit_action or config.fit_all:
        if config.use_vel_head:
            accuracy = [(ang_accuracy + acc_accuracy + vel_accuracy) / 3.0, ang_accuracy, acc_accuracy, vel_accuracy]
        else:
            accuracy = [(ang_accuracy + acc_accuracy) / 2.0, ang_accuracy, acc_accuracy]
    else:  # no accuracy for value
        accuracy = None

    return accuracy


def record_loss_and_accuracy(batch_data, loss_list, accuracy_list,
                             epoch_loss_list, epoch_accuracy_list):
    for i, loss in enumerate(loss_list):
        if accuracy_list and epoch_accuracy_list:
            if i < len(accuracy_list):
                accuracy = accuracy_list[i]
                epoch_accuracy = epoch_accuracy_list[i]
                epoch_accuracy.update(float(accuracy), batch_data.size(0))

        epoch_loss = epoch_loss_list[i]

        epoch_loss.update(float(loss), batch_data.size(0))


def choose_loss(acc_loss, ang_loss, vel_loss, v_loss, config):
    if config.fit_ang:
        loss = [ang_loss]
    elif config.fit_acc:
        loss = [acc_loss]
    elif config.fit_vel:
        loss = [vel_loss]
    elif config.fit_action:
        if config.use_vel_head:
            loss = [(ang_loss + acc_loss + vel_loss) / 3.0, ang_loss, acc_loss, vel_loss]
        else:
            loss = [(ang_loss + acc_loss) / 2.0, ang_loss, acc_loss]
    elif config.fit_val:
        loss = [v_loss]
    elif config.fit_all:
        if config.use_vel_head:
            loss = [(ang_loss + acc_loss + vel_loss + config.val_scale * v_loss) / (config.val_scale + 3.0), ang_loss, acc_loss, vel_loss, v_loss]
        else:
            loss = [(config.ang_scale * ang_loss + acc_loss + config.val_scale * v_loss) / (config.ang_scale + config.val_scale + 1.0), ang_loss, acc_loss, v_loss]
    else:
        loss = None

    return loss


def debug_data(data_point):
    car_channel = config.0

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
                        default=32,
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
                        default=config.num_iterations,  # 10
                        help='Number of Value Iterations')
    parser.add_argument('--f',
                        type=int,
                        default=config.GPPN_kernelsize,
                        help='Number of Value Iterations')
    parser.add_argument('--l_i',
                        type=int,
                        default=config.default_li,
                        help='Number of channels in input layer')
    parser.add_argument('--l_h',
                        type=int,
                        default=config.default_lh,  # 150
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
    parser.add_argument('--no_ped',
                        type=int,
                        default=config.0,
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
                        default=config.vin_out_channels,
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

    return parser.parse_args()


def update_global_config(cmd_args):
    print("\n=========================== Command line arguments ==========================")
    for arg in vars(cmd_args):
        print("==> {}: {}".format(arg, getattr(cmd_args, arg)))
    print("=========================== Command line arguments ==========================\n")

    # Update the global configurations according to command line
    config.0 = cmd_args.no_ped
    config.l2_reg_weight = cmd_args.l2
    config.vanilla_resnet = bool(cmd_args.no_vin)
    config.default_li = cmd_args.l_i
    config.default_lh = cmd_args.l_h
    config.GPPN_kernelsize = cmd_args.f
    config.num_iterations = cmd_args.k
    config.resnet_width = cmd_args.w
    config.vin_out_channels = cmd_args.vinout
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
        config.fit_all = False
    elif cmd_args.fit == 'acc':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = True
        config.fit_ang = False
        config.fit_val = False
        config.fit_all = False
    elif cmd_args.fit == 'steer':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = True
        config.fit_val = False
        config.fit_all = False
    elif cmd_args.fit == 'vel':
        config.fit_action = False
        config.fit_vel = True
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_all = False
    elif cmd_args.fit == 'val':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = True
        config.fit_all = False
    elif cmd_args.fit == 'all':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
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
                         '_nped_' + str(cmd_args.no_ped) + \
                         '_vin_o_' + str(global_config.vin_out_channels) + \
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

        # net.load_state_dict(checkpoint['state_dict'])
        resume_partial_model(checkpoint['state_dict'], net)

        # optimizer.load_state_dict(checkpoint['optimizer'])
        #
        # for g in optimizer.param_groups:
        #     g['lr'] = cmd_args.lr

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cmd_args.lr, weight_decay=config.l2_reg_weight)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=1e-2, factor=lr_factor,
                                                     verbose=True)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(cmd_args.resume, checkpoint['epoch']))
        print("=> using learning rate {}".format(cmd_args.lr))
        # cmd_args.modelfile = cmd_args.resume

        return optimizer, scheduler
    else:
        print("=> '{}' does not exist! ".format(cmd_args.resume))
        return None, None


def load_settings_from_model(checkpoint, config, cmd_args):
    try:
        # config.head_mode = checkpoint['head_mode']
        config.vanilla_resnet = checkpoint['novin']
        config.default_lh = checkpoint['l_h']
        cmd_args.l_h = checkpoint['l_h']
        config.vin_out_channels = checkpoint['vinout']
        config.resnet_width = checkpoint['w']
        config.resblock_in_layers = checkpoint['layers']
        # config.fit_ang = checkpoint['fit_ang']
        # config.fit_acc = checkpoint['fit_acc']
        # config.fit_vel = checkpoint['fit_vel']
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
              .format(global_config.default_lh))
        print("=> out channels of vin {}"
              .format(global_config.vin_out_channels))
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
        'l_h': config.default_lh,
        'vinout': config.vin_out_channels,
        'w': config.resnet_width,
        'layers': config.resblock_in_layers,
        'head_mode': config.head_mode,
        'do_p': config.do_prob,
        'fit_ang': config.fit_ang,
        'fit_acc': config.fit_acc,
        'fit_vel': config.fit_vel,
        'fit_val': config.fit_val,
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
    # Optimizer: weight_decay is the scaling factor for L2 regularization
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cmd_args.lr, weight_decay=config.l2_reg_weight)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, threshold=1e-2, factor=lr_factor,
                                                     verbose=True)

    if cmd_args.resume:
        optimizer, scheduler = resume_model()
    else:
        best_val_loss = None

    # Define Dataset
    train_set = Ped_data(filename=cmd_args.train, flag="Training")
    val_set = Ped_data(filename=cmd_args.val, flag="Validation")
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
        record_tb_data_list(-1, None, None, init_accuracy, init_loss)

    train()

    save_final_model(cmd_args.epochs)

    # visualize in tensor board
    writer.export_scalars_to_json("./all_losses.json")
    writer.close()
