import sys

from collections import OrderedDict

sys.path.append('./Data_processing/')
sys.path.append('./')

import math

import matplotlib

matplotlib.use('Agg')
from data_monitor import *
from train import forward_pass, forward_pass_jit, load_settings_from_model
from dataset import set_encoders
from Components.mdn import sample_mdn, sample_mdn_ml
from policy_value_network import PolicyValueNet


def set_decoders():
    if config.head_mode == "mdn":
        decode_steer = MdnSteerDecoder()  # conversion from id to normalized steering
        decode_acc = MdnAccDecoder()  # conversion from id to normalized acceleration
        decode_vel = MdnVelDecoder()  # conversion from id to normalized command velocity
    elif config.head_mode == "hybrid":
        decode_steer = SteerDecoder()  # one-hot vector of steering
        decode_acc = MdnAccDecoder()  # conversion from id to normalized acceleration
        decode_vel = MdnVelDecoder()  # conversion from id to normalized command velocity
    else:
        decode_steer = SteerDecoder()  # one-hot vector of steering
        decode_acc = AccDecoder()  # one-hot vector of acceleration
        decode_vel = VelDecoder()  # one-hot vector of command velocity
    return decode_steer, decode_acc, decode_vel


def get_copy(t):
    if t is not None:
        return t.clone()
    else:
        return None


class DriveController(nn.Module):
    def __init__(self, net):
        super(DriveController, self).__init__()
        clear_png_files('./visualize/', remove_flag='test_')
        self.data_monitor = DataMonitor()
        self.drive_net = net
        self.cmd_pub = rospy.Publisher('cmd_vel_drive_net', Twist, queue_size=1)

        rospy.Subscriber("odom", Odometry, self.odom_call_back)

        self.drive_timer = rospy.Timer(rospy.Duration(1.0 / config.control_freq), self.control_loop)

        self.cur_vel = None
        self.sm = nn.Softmax(dim=1)
        self.encode_input = InputEncoder()

        self.encode_steer, self.encode_acc, self.encode_vel = set_encoders()
        self.decode_steer, self.decode_acc, self.decode_vel = set_decoders()

        self.count = 0
        self.true_steering = 0
        self.update_steering = True
        self.dummy_count = 0

        self.label_ts = None

        self.acc_iter = 0
        self.old_acceleration = 0
        self.inference_count = 0

        # for visualization
        self.input_record = OrderedDict()
        self.output_record = OrderedDict()

    def vel_call_back(self, data):
        self.label_ts = time.time()
        self.cur_vel = data.linear.y
        print('Update current vel %f' % self.cur_vel)

    def odom_call_back(self, odo):
        self.cur_vel = odo.twist.twist.linear.x
        # print('Update current vel %f from odometry' % self.cur_vel)

    def control_loop(self, time_step):

        data_monitor_alive = self.data_monitor.check_alive()

        if not data_monitor_alive:

            if global_config.draw_prediction_records:

                self.visualize_hybrid_record()

            print("Node shutting down: data supply is broken")
            rospy.signal_shutdown("Data supply is broken")

        do_loop = self.check_do_loop()
        if not do_loop:
            print('skipping loop')
            return False

        if self.data_monitor.steering_is_triggered and self.data_monitor.steering_topic_alive:
            self.data_monitor.steering_topic_alive = False

        self.update_steering = False
        # self.data_monitor.update_steering = False

        self.data_monitor.update_data = False

        print('Disable data update')
        steering_label = None
        acc_label = None
        vel_label = None

        self.data_monitor.lock.acquire()

        start_time = time.time()

        try:
            if not self.data_monitor.data_valid():  # wait for valid data
                self.update_steering = True
                self.data_monitor.update_data = True
                self.data_monitor.update_steering = True
                print('Data not valid, skipping inference')
                return False

            if self.data_monitor.test_terminal():  # stop the car after reaching goal
                self.publish_terminal_cmd()
                print('Goal reached, skipping inference')
                return True

            acc_label, steering_label, vel_label = self.get_labels_combined()

            print("start inference: counter: " + str(self.count))

            # query the drive_net using current data
            if global_config.head_mode == "mdn":
                # Forward pass
                acc_pi, acc_mu, acc_sigma, \
                ang_pi, ang_mu, ang_sigma, \
                vel_pi, vel_mu, vel_sigma, value = self.inference()

                self.update_steering = True

                # print("re-open steering update")

                acceleration, steering, velocity = self.sample_from_mdn_distribution(acc_pi, acc_mu, acc_sigma,
                                                                                     ang_pi, ang_mu, ang_sigma,
                                                                                     vel_pi, vel_mu, vel_sigma)

                true_acc, true_vel = self.visualize_mdn_predictions(acc_pi, acc_mu, acc_sigma,
                                                                    ang_pi, ang_mu, ang_sigma,
                                                                    vel_pi, vel_mu, vel_sigma,
                                                                    acc_label, steering_label, vel_label)

            elif global_config.head_mode == "hybrid":
                # Forward pass
                acc_pi, acc_mu, acc_sigma, \
                ang, \
                vel_pi, vel_mu, vel_sigma, value = self.inference()

                # print("================predicted value:", value)

                self.update_steering = True

                # print("re-open steering update")

                ang_probs = self.sm(ang)

                acceleration, steering, velocity = self.sample_from_hybrid_distribution(acc_pi, acc_mu, acc_sigma,
                                                                                        ang_probs,
                                                                                        vel_pi, vel_mu, vel_sigma)

                true_acc, true_vel = self.visualize_hybrid_predictions(acc_pi, acc_mu, acc_sigma,
                                                                       ang_probs,
                                                                       vel_pi, vel_mu, vel_sigma,
                                                                       value,
                                                                       acc_label, steering_label, vel_label,
                                                                       None,
                                                                       acceleration)

            else:
                # Forward pass
                acc, ang, value, vel = self.inference()

                self.update_steering = True

                # print("re-open steering update")

                acc_probs, ang_probs, vel_probs = self.get_sm_probs(acc, ang, vel)

                true_acc, true_vel = self.visualize_predictions(acc_probs, ang_probs, vel_probs, acc_label,
                                                                steering_label, vel_label)

                acceleration, steering, velocity = \
                    self.sample_from_categorical_distribution(acc_probs, ang_probs, vel_probs)

            self.count += 1

            # construct ros topics for the outputs
            print("Steering bin: %d" % steering)
            steering = self.decode_steer(steering)
            acceleration = self.decode_acc(acceleration)

            if self.acc_iter == config.acc_slow_down:
                self.old_acceleration = acceleration
                self.acc_iter = 0
            else:
                self.acc_iter += 1
                acceleration = self.old_acceleration

            # # !!!!!! debugging !!!!!!
            # acceleration = 0.5
            # steering = 0.0
            # # !!!!!! debugging !!!!!!

            if global_config.use_vel_head:
                velocity = self.decode_acc(velocity)
            self.publish_actions(acceleration, steering, velocity, steering_label, true_acc, true_vel)

            self.data_monitor.update_data = True

            elapsed_time = time.time() - start_time
            print("Elapsed time in controlloop: %fs" % elapsed_time)

            self.data_monitor.update_steering = True

            return True
        finally:
            self.release_all_locks()
            return False

    def release_all_locks(self):
        if self.data_monitor.lock.locked():
            self.data_monitor.lock.release()
        self.update_steering = True
        self.data_monitor.update_steering = True
        self.data_monitor.update_data = True

    def publish_actions(self, acceleration, steering, velocity, steering_label, true_acc, true_vel):
        try:
            cmd = Twist()
            # cmd.linear.x: target_speed
            # cmd.linear.y: real_speed
            # cmd.linear.z: accelaration
            # cmd.angular.z: steering

            publish_true_steering = False
            if publish_true_steering:
                print('Publishing ground-truth angle')
                cmd.angular.z = float(steering_label)
                publish_true_steering = math.fabs(steering - np.degrees(steering_label)) > 20
            else:
                print('Publishing predicted angle')
                cmd.angular.z = np.radians(steering)

            velocity = self.cur_vel + acceleration / config.control_freq
            cmd.linear.x = max(min(velocity, config.vel_max), 0.0)  # target_speed_
            cmd.linear.y = self.cur_vel  # real_speed
            cmd.linear.z = acceleration  # _

            if config.fit_ang or config.fit_action or config.fit_all:
                print("output angle in degrees: %f" % float(steering))
                print("ground-truth angle: " + str(np.degrees(steering_label)))
            if config.fit_acc or config.fit_action or config.fit_all:
                print("output acc: %f" % float(acceleration))
                print("ground-truth acc: " + str(true_acc))
            if (config.fit_vel or config.fit_action or config.fit_all) and config.use_vel_head:
                print("output vel: %f" % float(velocity))
                print("ground-truth angle: " + str(true_vel))

            # publish action and acc commands
            self.cmd_pub.publish(cmd)
        except Exception as e:
            print("Exception when publishing commands: %s", e)
            error_handler(e)

    @staticmethod
    def sample_categorical(probs):
        distrib = Categorical(probs=probs)
        bin = distrib.sample()
        return bin

    @staticmethod
    def sample_categorical_ml(probs):
        # print('probs: ', probs)
        values, indices = probs.max(1)
        # print('indices: ', indices)
        bin = indices[0]
        return bin

    def sample_from_categorical_distribution(self, acc_probs, ang_probs, vel_probs):
        steering_bin = self.sample_categorical_ml(probs=ang_probs)
        acceleration_bin = self.sample_categorical(probs=acc_probs)
        velocity_bin = None
        if global_config.use_vel_head:
            velocity_bin = self.sample_categorical(probs=vel_probs)

        return acceleration_bin, steering_bin, velocity_bin

    @staticmethod
    def sample_guassian_mixture(pi, mu, sigma, mode="ml", component="acc"):
        # print('mdn mu params:', mu)

        if mode == 'ml':
            return float(sample_mdn_ml(pi, sigma, mu, component))
        else:
            return float(sample_mdn(pi, sigma, mu))

    def sample_from_mdn_distribution(self, acc_pi, acc_mu, acc_sigma,
                                     ang_pi, ang_mu, ang_sigma,
                                     vel_pi, vel_mu, vel_sigma):
        steering = self.sample_guassian_mixture(ang_pi, ang_mu, ang_sigma, mode="ml", component="steer")
        acceleration = self.sample_guassian_mixture(acc_pi, acc_mu, acc_sigma, mode="ml", component="acc")
        velocity = None
        if global_config.use_vel_head:
            velocity = self.sample_guassian_mixture(vel_pi, vel_mu, vel_sigma)
        return acceleration, steering, velocity

    def sample_from_hybrid_distribution(self, acc_pi, acc_mu, acc_sigma,
                                        ang_probs,
                                        vel_pi, vel_mu, vel_sigma):
        # steering_bin = self.sample_categorical(probs=ang_probs)
        steering_bin = self.sample_categorical(probs=ang_probs)

        # sample_mode = 'default'
        # if np.random.uniform(0.0, 1.0) > max(1.0 - float(self.count)/(20.0*global_config.control_freq), 0.1):
        #     sample_mode = 'ml'
        sample_mode = 'ml'
        acceleration = self.sample_guassian_mixture(acc_pi, acc_mu, acc_sigma, sample_mode)

        velocity = None
        if global_config.use_vel_head:
            velocity = self.sample_guassian_mixture(vel_pi, vel_mu, vel_sigma)
        return acceleration, steering_bin, velocity

    def visualize_predictions(self, acc_probs, ang_probs, vel_probs, acc_label, steering_label, vel_label):
        true_acc = None
        true_vel = None
        if config.visualize_inter_data:
            start_time = time.time()
            true_acc, true_vel, true_acc_labels, true_ang_labels, true_vel_labels = self.get_encoded_labels(
                acc_label, steering_label, vel_label)

            try:
                visualize_output_with_labels('test/' + str(self.count), acc_probs, ang_probs, vel_probs,
                                             true_acc_labels,
                                             true_ang_labels, true_vel_labels)

            except Exception as e:
                print("Exception when visualizing angles:", e)
                error_handler(e)

            elapsed_time = time.time() - start_time
            print("Visualization time: " + str(elapsed_time) + " s")
        return true_acc, true_vel

    def visualize_mdn_predictions(self, acc_pi, acc_mu, acc_sigma,
                                  ang_pi, ang_mu, ang_sigma,
                                  vel_pi, vel_mu, vel_sigma,
                                  acc_label, steering_label, vel_label):
        true_acc = None
        true_vel = None
        if config.visualize_inter_data:
            start_time = time.time()
            true_acc, true_vel, true_acc_labels, true_ang_labels, true_vel_labels = self.get_encoded_mdn_labels(
                acc_label, steering_label, vel_label)

            try:
                visualize_mdn_output_with_labels('test/' + str(self.count), acc_mu, acc_pi, acc_sigma, ang_mu, ang_pi,
                                                 ang_sigma, vel_mu, vel_pi, vel_sigma, true_acc_labels, true_ang_labels,
                                                 true_vel_labels)

            except Exception as e:
                print("Exception when visualizing angles:", e)
                error_handler(e)

            elapsed_time = time.time() - start_time
            print("Visualization time: " + str(elapsed_time) + " s")
        return true_acc, true_vel

    def visualize_hybrid_predictions(self, acc_pi, acc_mu, acc_sigma,
                                     ang_probs,
                                     vel_pi, vel_mu, vel_sigma,
                                     value,
                                     acc_label, steering_label, vel_label,
                                     v_label,
                                     accelearation,
                                     draw_truth=True, show_axis=True):
        true_acc = None
        true_vel = None
        true_acc, true_vel, true_acc_labels, true_ang_labels, true_vel_labels = self.get_encoded_hybrid_labels(
            acc_label, steering_label, vel_label)
        if config.visualize_inter_data:
            start_time = time.time()

            try:
                visualize_hybrid_output_with_labels('test/' + str(self.count), acc_mu, acc_pi, acc_sigma, ang_probs,
                                                    vel_mu, vel_pi, vel_sigma,
                                                    value,
                                                    true_acc_labels, true_ang_labels, true_vel_labels,
                                                    v_label,
                                                    accelearation,
                                                    draw_truth, show_axis)

            except Exception as e:
                print("Exception when visualizing angles:", e)
                error_handler(e)

            elapsed_time = time.time() - start_time
            print("Visualization time: " + str(elapsed_time) + " s")
        else:

            try:
                if global_config.draw_prediction_records:
                    self.output_record[str(self.count)] = [get_copy(acc_mu), get_copy(acc_pi), get_copy(acc_sigma),
                                                           get_copy(ang_probs),
                                                           get_copy(vel_mu), get_copy(vel_pi), get_copy(vel_sigma),
                                                           acc_label, steering_label, vel_label, accelearation]
            except Exception as e:
                print(e)
                exit(3)
        return true_acc, true_vel

    def visualize_hybrid_record(self):
        print('Visualizing prediction records')
        for step in self.output_record.keys():
            data = self.output_record[step]
            self.count = int(step)
            print('=> step', step)

            acc_mu = data[0]
            acc_pi = data[1]
            acc_sigma = data[2]
            ang_probs = data[3]
            vel_mu = data[4]
            vel_pi = data[5]
            vel_sigma = data[6]
            acc_label = data[7]
            steering_label = data[8]
            vel_label = data[9]
            accelaration = data[10]

            # X = self.input_record[step]

            config.visualize_inter_data = True

            # image_flag = 'test/'
            # flag = image_flag + str(self.count)
            # visualize(X.cpu()[0], flag)

            self.visualize_hybrid_predictions(acc_pi, acc_mu, acc_sigma, ang_probs,
                                              vel_pi, vel_mu, vel_sigma,
                                              None,
                                              acc_label, steering_label, vel_label,
                                              None,
                                              accelaration,
                                              draw_truth=False, show_axis=False)
        print('done')

    def get_sm_probs(self, acc, ang, vel):
        ang_probs = None
        acc_probs = None
        vel_probs = None
        try:
            ang_probs = self.sm(ang)
            acc_probs = self.sm(acc)
            if config.use_vel_head:
                vel_probs = self.sm(vel)
        except Exception as e:
            print("Exception at calculating ang distribution:", e)
            error_handler(e)
        # print ("ang_probs", ang_probs)
        return acc_probs, ang_probs, vel_probs

    def get_labels_combined(self):
        steering_label = self.data_monitor.cur_data['true_steer']
        acc_label = self.data_monitor.cur_data['true_acc']
        vel_label = self.data_monitor.cur_data['true_vel']
        return acc_label, steering_label, vel_label

    def get_labels_seperate(self, steering_label):
        steering_label = self.true_steering
        return steering_label

    def check_do_loop(self):
        do_loop = True
        self.dummy_count += 1
        if self.dummy_count % 1 != 0:
            do_loop = False
        return do_loop

    def publish_terminal_cmd(self):
        cmd = Twist()
        cmd.angular.z = 0
        cmd.linear.x = 0
        # publish action and acc commands
        self.cmd_pub.publish(cmd)
        self.data_monitor.lock.release()
        self.update_steering = True
        self.data_monitor.update_steering = True
        self.data_monitor.update_data = True

    def get_encoded_labels(self, acc_label, steering_label, vel_label):
        true_acc, true_acc_labels, true_steer_labels, true_vel, true_vel_labels = None, None, None, None, None

        try:
            true_steer_labels = self.get_steer_label(steering_label, true_steer_labels)

            true_acc, true_acc_labels = self.get_acc_label(acc_label, true_acc, true_acc_labels)

            true_vel, true_vel_labels = self.get_vel_label(true_vel, true_vel_labels, vel_label)

        except Exception as e:
            print("Exception when converting true label:", e)
            error_handler(e)

        return true_acc, true_vel, true_acc_labels, true_steer_labels, true_vel_labels

    def get_encoded_mdn_labels(self, acc_label, steering_label, vel_label):
        true_acc, true_acc_labels, true_steer_labels, true_vel, true_vel_labels = None, None, None, None, None

        try:
            true_steer_labels = self.get_mdn_steer_label(steering_label, true_steer_labels)

            true_acc, true_acc_labels = self.get_mdn_acc_label(acc_label, true_acc, true_acc_labels)

            true_vel, true_vel_labels = self.get_mdn_vel_label(true_vel, true_vel_labels, vel_label)

        except Exception as e:
            print("Exception when converting true label:", e)
            error_handler(e)

        return true_acc, true_vel, true_acc_labels, true_steer_labels, true_vel_labels

    def get_encoded_hybrid_labels(self, acc_label, steering_label, vel_label):
        true_acc, true_acc_labels, true_steer_labels, true_vel, true_vel_labels = None, None, None, None, None

        try:
            true_steer_labels = self.get_steer_label(steering_label, true_steer_labels)

            true_acc, true_acc_labels = self.get_mdn_acc_label(acc_label, true_acc, true_acc_labels)

            true_vel, true_vel_labels = self.get_mdn_vel_label(true_vel, true_vel_labels, vel_label)

        except Exception as e:
            print("Exception when converting true label:", e)
            error_handler(e)

        return true_acc, true_vel, true_acc_labels, true_steer_labels, true_vel_labels

    def get_vel_label(self, true_vel, true_vel_labels, vel_label):
        true_vel_labels = np.zeros(config.num_vel_bins, dtype=np.float32)
        if config.fit_vel or config.fit_action or config.fit_all:
            vel_label_np = float_to_np(vel_label)
            bin_idx, true_vel = self.encode_vel(vel_label_np)

            if config.label_smoothing:
                true_vel_labels = bin_idx
            else:
                true_vel_labels[bin_idx] = 1  # one hot vector
        return true_vel, true_vel_labels

    def get_acc_label(self, acc_label, true_acc, true_acc_labels):
        true_acc_labels = np.zeros(config.num_acc_bins, dtype=np.float32)
        if config.fit_acc or config.fit_action or config.fit_all:
            acc_label_np = float_to_np(acc_label)
            bin_idx, true_acc = self.encode_acc(acc_label_np)

            if config.label_smoothing:
                true_acc_labels = bin_idx
            else:
                true_acc_labels[bin_idx] = 1  # one hot vector
        return true_acc, true_acc_labels

    def get_steer_label(self, steering_label, true_steer_labels):
        true_steer_labels = np.zeros(config.num_steering_bins, dtype=np.float32)
        if config.fit_ang or config.fit_action or config.fit_all:
            true_steering_label = np.degrees(steering_label)
            bin_idx = self.encode_steer(true_steering_label)
            if config.label_smoothing:
                true_steer_labels = bin_idx
            else:
                true_steer_labels[bin_idx] = 1  # one hot vector
        return true_steer_labels

    def get_mdn_vel_label(self, true_vel, true_vel_labels, vel_label):
        true_vel_labels = np.zeros(1, dtype=np.float32)
        if config.fit_vel or config.fit_action or config.fit_all:
            vel_label_np = float_to_np(vel_label)
            true_vel_labels, true_vel = self.encode_vel(vel_label_np)
        return true_vel, true_vel_labels

    def get_mdn_acc_label(self, acc_label, true_acc, true_acc_labels):

        try:
            true_acc_labels = np.zeros(1, dtype=np.float32)
            if config.fit_acc or config.fit_action or config.fit_all:
                acc_label_np = float_to_np(acc_label)
                true_acc_labels, true_acc = self.encode_acc(acc_label_np)
        except Exception as e:
            print(e)
            print("Exception when encoding true acc label")
            exit(1)

        return true_acc, true_acc_labels

    def get_mdn_steer_label(self, steering_label, true_steer_labels):
        true_steer_labels = np.zeros(1, dtype=np.float32)
        if config.fit_ang or config.fit_action or config.fit_all:
            true_steering_label = np.degrees(steering_label)
            true_steer_labels = self.encode_steer(true_steering_label)
        return true_steer_labels

    def print_full(self, msg, tensor):
        print(msg)
        for i in range(tensor.size()[0]):
            for j in range(tensor.size()[1]):
                value = float(tensor[i][j].cpu())
                print(value, end=',')
            print()

    def inference(self):
        self.drive_net.eval()
        print("[inference] ")

        try:
            with torch.no_grad():
                X = self.get_current_data()

                # self.print_full("map:   ", X[0][0][config.channel_map])
                # self.print_full("path:   ", X[0][0][config.channel_goal])
                # self.print_full("car:    ", X[0][0][config.channel_hist1])

                self.data_monitor.update_data = True

                # debuging
                # image_flag = 'test/'
                # flag = image_flag + str(self.count)
                # visualize(X.cpu()[0], flag)
                # self.input_record[str(self.count)] = np.copy(X.cpu().numpy()[0])

                if config.model_type is "pytorch":
                    return forward_pass(X, self.count, self.drive_net, cmd_args, print_time=True, image_flag='test/')
                elif config.model_type is "jit":
                    return forward_pass_jit(X, self.count, self.drive_net, cmd_args, print_time=False, image_flag='test/')

        except Exception as e:
            error_handler(e)

    def get_current_data(self):
        X_np = self.data_monitor.cur_data['nn_input']
        data_len = X_np.shape[0]
        for i in range(0, data_len):
            X_np[i] = self.encode_input(X_np[i])
        X = torch.from_numpy(X_np)
        if config.visualize_val_data:
            pass
        X = X.to(device)
        return X


def print_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("No. parameters in model: %d", params)


from train import parse_cmd_args, update_global_config

if __name__ == '__main__':
    # Automatic swith of GPU mode if available
    # use_GPU = torch.cuda.is_available()
    # Parsing training parameters
    config = global_params.config

    # Parsing training parameters
    cmd_args = parse_cmd_args()

    update_global_config(cmd_args)

    config.augment_data = False

    load_goals(cmd_args.goalfile)

    config.model_type = ''
    print("=> loading checkpoint '{}'".format(cmd_args.modelfile))
    try:
        checkpoint = torch.load(cmd_args.modelfile)
        load_settings_from_model(checkpoint, global_config, cmd_args)

        # Instantiate the NN model
        net = PolicyValueNet(cmd_args)

        print_model_size(net)
        net = nn.DataParallel(net, device_ids=[0]).to(device)  # device_ids= config.GPU_devices

        net.load_state_dict(checkpoint['state_dict'])
        print("=> model at epoch {}"
              .format(checkpoint['epoch']))

        config.model_type = "pytorch"
    except Exception as e:
        print(e)

    # if config.model_type is '':
    #     try:
    #         config.batch_size = 1
    #         net = torch.jit.load(cmd_args.modelfile).cuda(device)
    #         print("=> JIT model loaded")
    #         config.model_type = "jit"
    #     except Exception as e:
    #         print(e)

    if config.model_type is not "pytorch" and  config.model_type is not "jit":
        print("model is not pytorch or jit model!!!")
        exit(1)

    rospy.init_node('drive_net', anonymous=True)

    DriveController = DriveController(net)

    rospy.spin()
    # spin listner
