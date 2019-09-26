import subprocess
from argparse import Namespace
import argparse
from os.path import expanduser
import time
import os
import sys
from socketIO_client import SocketIO, LoggingNamespace
from clear_process import clear_process


config = Namespace()

home = expanduser("/home/yuanfu")
nn_folder = '/workspace/BTS_RL_NN/catkin_ws/src/IL_controller/src/BTS_RL_NN/'
if not os.path.isdir(home):
    home = expanduser("/home/panpan")
if not os.path.isdir(home + nn_folder):
    nn_folder = '/workspace/catkin_ws/src/IL_controller/src/BTS_RL_NN/'
print("nn working dir: ", home + nn_folder)

root_path = home + '/Unity/DESPOT-Unity'

dataset_path = 'dataset'

print("root path: ", root_path)

def check_process(p_handle, p_name):
    return_code = ''
    try:
        return_code = subprocess.check_output("ps -a " + "| grep " + str(p_handle.pid), shell=True)
        return_code = return_code.decode()
    except Exception:
        # if config.verbosity > 0:
        print('Subprocess', p_name, "has died")
        return False

    feature = 'defunct'
    if feature in return_code:
        print('Subprocess', p_name, "has been defuncted")
        if config.verbosity > 0:
            print("pgrep returned", return_code)
        return False

    return True


import atexit


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)


def exit_handler():
    print('Client is exiting! Clearing process...')
    clear_process()


atexit.register(exit_handler)


def clear_old_data(root_folder, subfolder=None, remove_flag='', skip_flag='.py'):
    if subfolder:
        folder = root_folder + subfolder + '/'
    else:
        folder = root_folder
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path) and remove_flag in the_file and skip_flag not in the_file:
                os.remove(file_path)
            if os.path.isdir(file_path):
                clear_old_data(root_folder=folder, subfolder=the_file, remove_flag=remove_flag)
        except Exception as e:
            error_handler(e)


client_order = -1
num_clients = -1


def on_data_ready_response(*args):
    print('INFO: on_data_ready_response', args)
    model_version = args[0]
    model = args[1]

    if len(args) > 2:
        global client_order
        client_order = args[2]

        print('Receive client_order %d' % client_order)

    if len(args) > 3:
        global num_clients
        num_clients = args[3]

        print('Receive num_clients %d' % num_clients)

    if model_version >= config.num_iter:
        print("loop finished, exiting client")

        exit(0)

    if 'trained_models/' in model:
        model = model.replace('trained_models/', '')

    print("INFO: New model %s version %d is ready" % (model, model_version))

    if config.dummy:
        time.sleep(1)
        # notify the server to start training
        socketIO.emit('data_ready', {'empty': ''}, on_data_ready_response)
    else:

        if config.collect:
            # launch the data collection script and wait for completion
            print("Collecting data using model version %d" % model_version)
            gpu = config.gpu
            s = model_version * num_clients + client_order
            e = model_version * num_clients + client_order + 1
            num_rounds = 1

            shell_cmd = 'rm result/exp_log_' + str(s) + '_' + str(e)
            subprocess.call(shell_cmd.split())
            for i in range(s, e):
                print("[repeat_run] starting run_data_collection.py script, batch ", str(i))

                start_batch = i * num_rounds
                print("[repeat_run] start_batch:", str(start_batch))

                end_batch = (i + 1) * num_rounds
                print("[repeat_run] end_batch:", end_batch)

                time_out = num_rounds * 11 * 120 * 4
                print("[repeat_run] time_out:", time_out)

                print("[repeat_run] gpu_id:", gpu)

                shell_cmd = 'python3 run_data_collection.py --sround ' + str(start_batch) + \
                            ' --eround ' + str(end_batch) + ' --timeout ' + str(time_out) + ' --model ' + model + \
                            ' --gpu_id ' + str(gpu) + ' --record 1 --net 1 --verb 1 --test 1'

                data_collection_out = open('result/exp_log_' + str(s) + '_' + str(e), 'a+')

                subprocess.call(shell_cmd.split(), stdout=data_collection_out, stderr=data_collection_out)

                data_collection_out.close()

                print("[repeat_run] clearing process")
                # shell_cmd = root_path + "/clear_process.sh"
                # subprocess.call(shell_cmd.split())
                clear_process()
                time.sleep(3)

                print("[repeat_run] end batch ", i)

        if config.process:
            # process data upon completion
            shell_cmd = "bash data_processing.sh " + root_path + "/result/ " + dataset_path
            if config.verbosity > 0:
                print('[cmd] ', shell_cmd)
            subprocess.call(shell_cmd.split(), cwd= home + nn_folder)

        # notify the server to start training
        if config.verbosity > 0:
            print('[INFO] notifying server')
        socketIO.emit('data_ready', {'empty': ''}, on_data_ready_response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--verb',
                        type=int,
                        default=0,
                        help='Verbosity')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='gpu id')
    parser.add_argument('--s',
                        type=int,
                        default=0,
                        help='start batch')
    parser.add_argument('--e',
                        type=int,
                        default=1,
                        help='end batch')
    parser.add_argument('--model',
                        type=str,
                        default='hybrid_unicorn.pth',
                        help='model filename')
    parser.add_argument('--dummy',
                        type=int,
                        default=0,
                        help='dummmy run flag')
    parser.add_argument('--collection',
                        type=int,
                        default=1,
                        help='choose whether to do data collection')
    parser.add_argument('--processing',
                        type=int,
                        default=1,
                        help='choose whether to do data processing')

    parser.add_argument('--iter',
                        type=int,
                        default=1,
                        help='number of iterations to be executed')

    cmd_args = parser.parse_args()
    config.verbosity = cmd_args.verb
    config.gpu = int(cmd_args.gpu)
    config.s = int(cmd_args.s)
    config.e = int(cmd_args.e)
    config.model = cmd_args.model
    config.dummy = cmd_args.dummy
    config.collect = bool(cmd_args.collection)
    config.process = bool(cmd_args.processing)
    config.num_iter = cmd_args.iter

    try:
        # localhost 127.0.0.1, server port 8080
        server_address = '172.17.0.1'  # local host ip to be used by the docker
        socketIO = SocketIO(server_address, 8080)

        print("INFO: ego client connected with server")

        socketIO.emit('ready', {'empty': ''}, on_data_ready_response)

        print("INFO: ready signal sent to server")

        socketIO.wait()
    except ConnectionError:
        print('ERROR: The server is down. Try again later.')
