import sys

sys.path.append('./Data_processing/')
from dataset import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Components.mdn import gaussian_probability, gaussian_probability_np, mdn_accuracy

import pdb

from Data_processing import global_params
from Data_processing.global_params import error_handler

config = global_params.config
from transforms import make_onehot


def inspect(X):
    last_ped = 0 - 1
    car_channel = 0

    print("X data dims: %d %d %d %d" % (X.size(0), X.size(1), X.size(2), X.size(3)))
    # print ("max values in last ped:")
    # print ("map: %f"%torch.topk(X[last_ped,config.channel_map].view(X[last_ped,config.channel_map].numel()), 1)[0] )
    # print ("map maximum: %f"%torch.max(X[car_channel,0]) )
    # print ("map sum: %f"%torch.sum(X[car_channel,0]) )
    # print ("hist1: %f"%torch.topk(X[last_ped,config.channel_hist1].view(X[last_ped,config.channel_hist1].numel()), 1)[0] )
    # print ("hist2: %f"%torch.topk(X[last_ped,config.channel_hist2].view(X[last_ped,config.channel_hist2].numel()), 1)[0] )
    # print ("max values in car:")
    # print ("map: %f"%torch.topk(X[car_channel,config.channel_map].view(X[car_channel,config.channel_map].numel()), 1)[0] )
    # print ("goal: %f"%torch.topk(X[car_channel,config.channel_goal].view(X[car_channel,config.channel_goal].numel()), 1)[0] )
    # print ("hist1: %f"%torch.topk(X[car_channel,config.channel_hist1].view(X[car_channel,config.channel_hist1].numel()), 1)[0] )
    # print ("hist2: %f"%torch.topk(X[car_channel,config.channel_hist2].view(X[car_channel,config.channel_hist2].numel()), 1)[0] )    


map_type = 'gray'  # 'hot'
title_loc = 'center'


def visualize_gppn(Z, flag, step):
    # print("[visualize_gppn] ")

    Z = Z.detach().cpu().numpy()

    x_dim = 4
    y_dim = 4

    try:
        fig, axarr = plt.subplots(x_dim, y_dim)

        for i in range(0, x_dim):
            for j in range(0, y_dim):
                k = i * y_dim + j
                if k < config.gppn_out_channels:
                    axarr[i, j].imshow(Z[k],
                                       cmap=map_type, interpolation='nearest')
                axarr[i, j].axis('off')
    except Exception as e:
        print(e)

    plt.tight_layout(pad=0.0, w_pad=-11.0, h_pad=-1.0)

    image_subfolder = 'visualize/' + flag + '_value_image/'
    save_figure(fig, image_subfolder, root="", step=step)

    plt.close(fig)


def visualize_resimage(Z, step):
    # print("[visualize_resimage] ")

    x_dim = 8
    y_dim = 8

    try:
        fig, axarr = plt.subplots(x_dim, y_dim)

        for i in range(0, x_dim):
            for j in range(0, y_dim):
                k = i * y_dim + j

                if k < config.resnet_width:
                    axarr[i, j].imshow(Z[k].detach().cpu(),
                                       cmap=map_type, interpolation='nearest')
                    axarr[i, j].axis('off')
    except Exception as e:
        print(e)

    plt.tight_layout(pad=0.0, w_pad=-11.0, h_pad=-1.0)

    image_subfolder = 'visualize/res_out_image/'
    save_figure(fig, image_subfolder, root="", step=step)


def visualize_res(Z, step):
    # print("[visualize_res] ")

    fig = plt.figure(1)
    horizontal = np.arange(-500, 500, 1)

    Z = Z.cpu().data.numpy()
    plt.plot(horizontal, Z)
    plt.title('Resnet output', loc=title_loc)

    image_subfolder = 'visualize/res_output/'
    save_figure(fig, image_subfolder, root="", step=step)


def visualize_distribution(p, true_p, step, flag):
    # print("[visualize_distribution] {}: p={} true_p={}".format(flag, p, true_p))

    try:
        assert p is not None, "Inferred probability is None"
        assert true_p is not None, "Label is None"
        num_bins = None

        min_val = -config.max_steering_degree
        max_val = config.max_steering_degree
        resolution = config.steering_resolution

        if flag == "steering":
            num_bins = config.num_steering_bins
        elif flag == "acc":
            min_val = -config.max_acc
            max_val = config.max_acc
            num_bins = config.num_acc_bins
            resolution = (max_val - min_val) / num_bins
        elif flag == "vel":
            # print("plotting velocity...")
            max_val = config.vel_max
            min_val = 0
            num_bins = config.num_vel_bins
            resolution = (max_val - min_val) / num_bins
        elif flag == "lane":
            min_val = -1.0
            max_val = 1.0
            num_bins = config.num_lane_bins
            resolution = (max_val - min_val) / num_bins

        bin_width = resolution

        fig = plt.figure(1)
        horizontal = np.arange(min_val + resolution / 2.0, max_val + resolution / 2.0, resolution)

        plt.subplot(211)

        if torch.is_tensor(p):
            p = p.view(num_bins).cpu().data.numpy()
        plt.bar(horizontal, height=p, facecolor='k', alpha=0.75, width=bin_width)

        plt.ylim(0, 1)
        plt.title(flag + ' distribution', loc=title_loc)

        plt.subplot(212)

        if not isinstance(true_p, np.ndarray):
            true_p = true_p.cpu().numpy()

        if true_p.size == 1:
            plot_p = make_onehot(num_bins, int(true_p), prob=1.0)
        else:
            plot_p = true_p

        plt.bar(horizontal, height=plot_p, facecolor='r', alpha=0.75, width=bin_width)
        plt.ylim(0, 1)
        plt.title(flag + 'ground-truth', loc=title_loc)

        plt.tight_layout()

        image_subfolder = 'visualize/' + flag + '_distrib/'
        save_figure(fig, image_subfolder, root="", step=step)
    except Exception as e:
        error_handler(e)


def visualize_guassian_mixture(pi_list, mu_list, sigma_list, label, step, flag, draw_ground_truth=True, show_axis=True):
    # print("[visualize_guassian_mixture]")
    try:

        if torch.is_tensor(pi_list):
            pi_list = pi_list.cpu()
            mu_list = mu_list.cpu()
            sigma_list = sigma_list.cpu()

        if config.num_guassians_in_heads > 1:
            pi_list = pi_list.squeeze(0).numpy()
            mu_list = mu_list.squeeze(0).squeeze(1).numpy()
            sigma_list = sigma_list.squeeze(0).squeeze(1).numpy()
        else:
            pi_list = pi_list.numpy()
            mu_list = mu_list.squeeze(0).numpy()
            sigma_list = sigma_list.squeeze(0).numpy()

        num_points = 1000

        min_val = - 1.0
        max_val = 1.0

        if draw_ground_truth:
            fig = plt.figure(1)
        else:
            fig = plt.figure(figsize=(8, 3))

        horizontal = np.linspace(start=min_val, stop=max_val, num=num_points)

        if draw_ground_truth:
            plt.subplot(211)

        max_height = 0.0

        for i, pi_entry in enumerate(pi_list):
            mu_entry = mu_list[i]
            sigma_entry = sigma_list[i]
            pi = np.ones(num_points) * pi_entry
            mu = np.ones(num_points) * mu_entry
            sigma = np.ones(num_points) * sigma_entry
            p = pi * gaussian_probability_np(sigma, mu, horizontal)

            max_height = np.maximum(max_height, np.max(p))
            plt.plot(horizontal, p, color='k', alpha=0.75)

        if not draw_ground_truth:
            height = 0.0
            for i, pi_entry in enumerate(pi_list):
                mu_entry = mu_list[i]
                sigma_entry = sigma_list[i]
                pi = np.ones(1) * pi_entry
                mu = np.ones(1) * mu_entry
                sigma = np.ones(1) * sigma_entry
                height += pi * gaussian_probability_np(sigma, mu, label)

            plt.plot([label, label], [0, height], color='b', alpha=0.75)

        plt.xlim(-1.2, 1.2)
        plt.ylim(0, max(max_height * 1.2, 0.1))
        plt.title(flag + ' distribution', loc=title_loc)

        if draw_ground_truth:
            plt.subplot(212)
            plt.plot([label, label], [0, max_height], color='r', alpha=0.75)
            plt.xlim(-1.2, 1.2)
            plt.ylim(0, max(max_height * 1.2, 0.1))
            plt.title(flag + 'ground-truth', loc=title_loc)

        if not show_axis:
            plt.yticks([])

        plt.tight_layout()

        image_subfolder = 'visualize/' + flag + '_distrib/'
        save_figure(fig, image_subfolder, root="", step=step)
    except Exception as e:
        error_handler(e)


def visualize_value(v, true_v, step):
    fig = plt.figure(1)
    try:
        v = float(v)
        true_v = float(true_v)

        x = np.array([0, 1])
        y = np.array([v, v])
        plt.plot(x, y, lw=3, color='k')

        y = np.array([true_v, true_v])
        plt.plot(x, y, lw=3, color='r')

        plt.ylim(-1.5, 1.5)
        plt.title('Acceleration', loc=title_loc)

        plt.tight_layout()

        image_subfolder = 'visualize/acceleration/'
        save_figure(fig, image_subfolder, root="", step=step)

    except Exception as e:
        print(e)
        print("v", v)
        print("true_v", true_v)


def export_prediction(value, true_value, step, flag):
    sub_folder = 'visualize/' + flag + '_distrib/'
    root = ""
    folder = root + sub_folder + step.split('/')[0] + '/ssm' + str(config.sigma_smoothing) + '/'

    if sys.version_info[0] > 2:
        os.makedirs(folder, exist_ok=True)
    else:
        if not os.path.exists(folder):
            os.makedirs(folder)

    flag = step.replace('/', '_')

    txt_name = folder + flag + '.txt'

    file = open(txt_name, 'w+')
    file.write('label: ' + str(true_value))
    file.write('pred: ' + str(value))
    file.close()


def visualize(X, step, root=""):
    print("[visualize] ")
    # print("==> goal channel to plot: {}".format(config.channel_goal))

    last_ped = 0
    car_channel = 0

    x_dim = 2
    y_dim = 4  # config.num_hist_channels
    stride = config.num_hist_channels // 4

    # print('stride', stride)

    fig, axarr = plt.subplots(x_dim, y_dim)

    for i in range(0, x_dim):
        for j in range(0, y_dim):
            try:
                if i == 0:
                    if j == 0:
                        axarr[i, j].imshow(X[last_ped, config.channel_map[0]],
                                           cmap=map_type, interpolation='nearest')
                    if j == 1:
                        if config.use_goal_channel:
                            axarr[i, j].imshow(X[last_ped, config.channel_goal],
                                               cmap=map_type, interpolation='nearest')
                    if j == 2:
                        axarr[i, j].imshow(X[last_ped, config.channel_lane],
                                           cmap=map_type, interpolation='nearest')
                    if j >= 3:
                        axarr[i, j].imshow(np.ones((config.imsize, config.imsize), dtype=np.float32),
                                           cmap='gist_yarg', interpolation='nearest')
                        axarr[i, j].axis('off')
                        continue
                # elif i == 1:
                #     axarr[i, j].imshow(X[last_ped, config.channel_hist[0] + j * stride],
                #                        cmap=map_type, interpolation='nearest')
                elif i == 1:
                    axarr[i, j].imshow(X[car_channel, config.channel_map[0] + j * stride],
                                       cmap=map_type, interpolation='nearest')

                axarr[i, j].axis('off')

            except Exception as e:
                print(e)

    plt.tight_layout()

    image_subfolder = 'visualize/input_ped/'
    save_figure(fig, image_subfolder, root, step)


vis_step = 0


def visualized_exo_agent_data(env_maps, root=""):
    global vis_step
    vis_step += 1
    fig, axarr = plt.subplots(2, 2)
    fig.set_figheight(6 * 2)
    fig.set_figwidth(6 * 2)
    for i in range(0, 2):
        for j in range(0, 2):
            k = i * 2 + j
            axarr[i, j].imshow(env_maps[k],
                               cmap=map_type, interpolation='nearest')
    plt.tight_layout()

    image_subfolder = 'visualize/h5_env_maps/'

    save_figure(fig, image_subfolder, root, 'raw/' + str(vis_step))


def visualize_image(car_map, root="", subfolder="h5_car_map"):
    try:
        global vis_step
        fig, axarr = plt.subplots(1, 1)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        axarr.imshow(car_map,
                     cmap=map_type, interpolation='nearest')
        plt.tight_layout()

        image_subfolder = os.path.join('visualize/', subfolder)

        save_figure(fig, image_subfolder, root, 'raw/' + str(vis_step))
    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def visualize_overlay_image(base_image, top_image, top_image1, lane_label, acc_label, root="", subfolder="h5_overlay_map"):
    try:
        global vis_step
        vis_step += 1
        fig, axarr = plt.subplots(1, 1)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        draw_image = np.stack([top_image, top_image, base_image], axis=-1)
        axarr.imshow(draw_image, interpolation='nearest')
        axarr.annotate('lane: {}'.format(lane_label), color='white',
                    xy=(80, 80), xycoords='figure pixels')
        axarr.annotate('acc: {}'.format(acc_label), color='white',
                       xy=(80, 100), xycoords='figure pixels')
        plt.tight_layout()

        image_subfolder = os.path.join('visualize/', subfolder)
        save_figure(fig, image_subfolder, root, 'raw/' + str(vis_step))

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

def visualize_both_agent_inputs(Cart_data, image_data, step, root=""):
    # print("[visualize_cart] ")

    x_dim = 2 + 2
    stride = 2
    y_dim = config.num_hist_channels // stride

    fig, axarr = plt.subplots(x_dim, y_dim)

    fig.set_figheight(3 * x_dim)
    fig.set_figwidth(3 * y_dim)

    # draw ped map images

    for i in range(0, 2):
        for j in range(0, y_dim):
            try:
                if i == 0:
                    axarr[i, j].imshow(image_data[0, config.channel_hist[0] + j * stride],
                                       cmap=map_type, interpolation='nearest')
                elif i == 1:
                    axarr[i, j].imshow(image_data[0, config.channel_map[0] + j * stride],
                                       cmap=map_type, interpolation='nearest')

                axarr[i, j].axis('off')
            except Exception as e:
                print(e)

    # draw cart agents

    for i in range(0, 2):
        for j in range(0, y_dim):

            ts = i * y_dim + j

            row = i + 2

            try:
                start = ts * 2 * (1 + config.num_peds_in_map)
                end = (ts + 1) * 2 * (1 + config.num_peds_in_map)
                draw_cart(axarr[row, j], Cart_data[start:end])

            except Exception as e:
                print(e)

    plt.tight_layout()

    image_subfolder = 'visualize/input_cart/'
    save_figure(fig, image_subfolder, root, step)


def draw_cart(ax, data):
    ax.set_ylim(-20.0, 20.0)
    ax.set_xlim(-20.0, 20.0)

    ax.set_aspect(1.0)

    size = 0.3

    patches = []

    fig_data = iter(data)

    is_car = True
    for x in fig_data:
        y = next(fig_data)

        if is_car:
            circle1 = plt.Circle((x, -y), 1, fill=True, alpha=1, color='green')
            patches.append(ax.add_patch(circle1))
            is_car = False
        else:
            circle1 = plt.Circle((x, -y), size, fill=True, alpha=1, color='red')
            patches.append(ax.add_patch(circle1))


def save_figure(fig, image_subfolder, root, step):
    # print('[save_figure]')
    try:
        folder = os.path.join(os.path.join(root, image_subfolder),
                              os.path.join(step.split('/')[0], 'ssm' + str(config.sigma_smoothing)))
        if sys.version_info[0] > 2:
            os.makedirs(folder, exist_ok=True)
        else:
            if not os.path.exists(folder):
                os.makedirs(folder)
        flag = step.split("/")[1]
        fig.savefig(os.path.join(folder, flag + '.png'), bbox_inches='tight', transparent=False)

        print("Figure {} generated".format(folder + '/' + flag + '.png'))
        plt.close(fig)

    except Exception as e:
        print(e)


def visualize_input_output(X, ped_gppn_out, car_gppn_out, res_image, step, image_flag):
    if config.visualize_inter_data:
        flag = image_flag + str(step)
        visualize(X.cpu()[0], flag)
        if ped_gppn_out.size(0) > 0:
            visualize_gppn(ped_gppn_out, 'ped', flag)
        if car_gppn_out.size(0) > 0:
            visualize_gppn(car_gppn_out, 'car', flag)
        if res_image.size(0) > 0:
            visualize_resimage(res_image, flag)


def visualize_hybrid_output_with_labels(count, acc_mu, acc_pi, acc_sigma, ang_probs,
                                        vel_mu, vel_pi, vel_sigma, lane_probs, value,
                                        encoded_acc_label, encoded_ang_label, encoded_vel_label, encoded_lane_label,
                                        encoded_value_label,
                                        accelaration=None, draw_truth=True, show_axis=True):
    print('')
    # print("\n[visualize_hybrid_output_with_labels] ")
    if config.fit_ang or config.fit_action or config.fit_all:
        # print('steer')
        visualize_distribution(ang_probs, encoded_ang_label, count, 'steering_hybrid')
    if config.fit_acc or config.fit_action or config.fit_all:
        if draw_truth:
            visualize_guassian_mixture(acc_pi, acc_mu, acc_sigma, encoded_acc_label, count, 'acc_hybrid',
                                       draw_truth, show_axis)
        else:
            visualize_guassian_mixture(acc_pi, acc_mu, acc_sigma, accelaration, count, 'acc_hybrid',
                                       draw_truth, show_axis)
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            # print('vel')
            visualize_guassian_mixture(vel_pi, vel_mu, vel_sigma, encoded_vel_label, count, 'vel_hybrid')
    if config.fit_val or config.fit_all:
        if encoded_value_label is not None:
            export_prediction(value, encoded_value_label, count, 'value_hybrid')
    if config.fit_lane or config.fit_action or config.fit_all:
        visualize_distribution(lane_probs, encoded_lane_label, count, 'lane_hybrid')


def visualize_mdn_output_with_labels(count, acc_mu, acc_pi, acc_sigma, ang_mu, ang_pi, ang_sigma,
                                     vel_mu, vel_pi, vel_sigma, lane_probs,
                                     encoded_acc_label, encoded_ang_label, encoded_vel_label, encoded_lane_label):
    print('')
    # print("\n[visualize_mdn_output_with_labels] ")
    if config.fit_ang or config.fit_action or config.fit_all:
        visualize_guassian_mixture(ang_pi, ang_mu, ang_sigma, encoded_ang_label, count, 'steering_mdn')
    if config.fit_acc or config.fit_action or config.fit_all:
        visualize_guassian_mixture(acc_pi, acc_mu, acc_sigma, encoded_acc_label, count, 'acc_mdn')
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            visualize_guassian_mixture(vel_pi, vel_mu, vel_sigma, encoded_vel_label, count, 'vel_mdn')
    if config.fit_lane or config.fit_action or config.fit_all:
        visualize_distribution(lane_probs, encoded_lane_label, count, 'lane_mdn')


def visualize_output_with_labels(count, acc_probs, ang_probs, vel_probs, lane_probs,
                                 encoded_acc_label, encoded_ang_label,
                                 encoded_vel_label, encoded_lane_label):
    # print("\n[visualize_output_with_labels] ")
    if config.fit_ang or config.fit_action or config.fit_all:
        visualize_distribution(ang_probs, encoded_ang_label, count, 'steering')
    if config.fit_acc or config.fit_action or config.fit_all:
        visualize_distribution(acc_probs, encoded_acc_label, count, 'acc')
    if config.fit_vel or config.fit_action or config.fit_all:
        if config.use_vel_head:
            visualize_distribution(vel_probs, encoded_vel_label, count, 'vel')
    if config.fit_lane or config.fit_action or config.fit_all:
        visualize_distribution(lane_probs, encoded_lane_label, count, 'lane')


import os


def clear_png_files(root_folder, subfolder=None, remove_flag=''):
    if subfolder:
        folder = root_folder + subfolder + '/'
    else:
        folder = root_folder

    if not os.path.exists(folder):
        os.makedirs(folder)

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path) and remove_flag in the_file:
                os.remove(file_path)
            if os.path.isdir(file_path):
                clear_png_files(root_folder=folder, subfolder=the_file, remove_flag=remove_flag)
        except Exception as e:
            error_handler(e)
    # else:
    #     clear_png_files('input_car', remove_flag)
    #     clear_png_files('input_ped', remove_flag)
    #     clear_png_files('car_value_image', remove_flag)
    #     clear_png_files('ped_value_image', remove_flag)
    #     clear_png_files('res_out_image', remove_flag)
    #     clear_png_files('res_output', remove_flag)
    #     clear_png_files('steering_distrib', remove_flag)
    #     clear_png_files('acc_distrib', remove_flag)
    #     clear_png_files('vel_distrib', remove_flag)


if __name__ == '__main__':
    from train import parse_cmd_args, update_global_config
    from train import forward_pass, forward_pass_jit, load_settings_from_model, debug_data, visualize_hybrid_predictions

    from policy_value_network import PolicyValueNet
    import torch
    import torch.nn as nn

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = global_params.config
    global_config = global_params.config

    # Parsing training parameters
    cmd_args = parse_cmd_args()

    update_global_config(cmd_args)

    config.augment_data = True

    # Instantiate a BTS-RL model
    net = PolicyValueNet(cmd_args)

    net = nn.DataParallel(net, device_ids=[0]).to(device)  # device_ids= config.GPU_devices

    config.model_type = ''
    print("=> loading checkpoint '{}'".format(cmd_args.modelfile))
    try:
        checkpoint_vis = torch.load(cmd_args.modelfile)
        load_settings_from_model(checkpoint_vis, global_config, cmd_args)
        net.load_state_dict(checkpoint_vis['state_dict'])
        print("=> model at epoch {}"
              .format(checkpoint_vis['epoch']))

        config.model_type = "pytorch"
    except Exception as e:
        print(e)

    if config.model_type is '':
        try:
            config.batch_size = 1
            net = torch.jit.load(cmd_args.modelfile).cuda(device)
            print("=> JIT model loaded")
            config.model_type = "jit"
        except Exception as e:
            print(e)

    if config.model_type is not "pytorch" and config.model_type is not "jit":
        print("model is not pytorch or jit model!!!")
        exit(1)

    data_set_vis = DrivingData(filename=cmd_args.train, flag="Training")
    # Create Dataloader
    do_shuffle_training = False
    if config.sample_mode == 'hierarchical':
        do_shuffle_training = False
    else:
        do_shuffle_training = True

    train_loader_vis = torch.utils.data.DataLoader(
        data_set_vis, batch_size=cmd_args.batch_size, shuffle=do_shuffle_training, num_workers=config.num_data_loaders)

    sm_vis = nn.Softmax(dim=1)

    for i, data_point_vis in enumerate(train_loader_vis):

        visualize_results = True

        with torch.no_grad():
            # Get input batch
            X, v_labels_test, acc_labels_test, ang_labels_test, vel_labels_test, lane_labels_test = data_point_vis
            if global_config.visualize_val_data:
                debug_data(X[1])
            X, v_labels_test, acc_labels_test, ang_labels_test, vel_labels_test, lane_labels_test = X.to(
                device), v_labels_test.to(device), acc_labels_test.to(device), ang_labels_test.to(device), \
                vel_labels_test.to(device), lane_labels_test.to(device)

            if global_config.head_mode == "hybrid":
                # Forward pass
                acc_pi_test, acc_mu_test, acc_sigma_test, \
                ang_test, vel_pi_test, vel_mu_test, vel_sigma_test, lane_test, value_test = \
                    forward_pass(X, None, step=i, drive_net=net, cmd_config=cmd_args, print_time=visualize_results,
                                 image_flag='load/')

                visualize_hybrid_predictions(i, acc_mu_test, acc_pi_test, acc_sigma_test, acc_labels_test, ang_test,
                                             ang_labels_test, vel_labels_test, lane_test, lane_labels_test,
                                             value_test, v_labels_test,
                                             visualize_results, sm_vis, 'load/')
