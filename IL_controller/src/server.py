import socketio
import subprocess
from argparse import Namespace
import argparse
from os.path import expanduser
import time
import os
import eventlet
import eventlet.wsgi
from flask import Flask

sio = socketio.Server()
app = Flask(__name__)

model_version = -1

config = Namespace()

home = expanduser("~")

# TODO: Change this to your project location
# TODO: You need to link the external OpenAI project into the BTS_RL_NN project

nn_folder = '/workspace/BTS_RL_NN/catkin_ws/src/IL_controller/src/BTS_RL_NN/'

# TODO: Change the home path to your own

if not os.path.isdir(home):
    home = expanduser("/home/panpan")
if not os.path.isdir(home + nn_folder):
    nn_folder = '/workspace/catkin_ws/src/IL_controller/src/BTS_RL_NN/'

print("working dir: ", home + nn_folder)

# Client_address = 'Unicorn1:'


# TODO: Create a dummy client on your computer by putting client.py in a location
#  and run:   python3 Client.py --dummy

# TODO: Change the three paths to point to the location of your own dummy client.
Client_address = ''
Client_nn_workspace = '/home/panpan/workspace/catkin_ws/src/IL_controller/src/BTS_RL_NN/'
Client_dataset_path = "dataset/"

Connection_count = 0
Data_ready_count = 0
Model_ready = False
new_model = ''

Expected_client_number = 1

Client_ids = {}

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


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)


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


@sio.on('connect')
def on_connect(sid, environ):
    print("Client %s connected" % sid)
    print(environ)

    global Connection_count

    Client_ids[str(sid)] = Connection_count

    Connection_count += 1

    print("Connection_count:", Connection_count)
    print("Expected_client_number:", Expected_client_number)

    pass


@sio.on('disconnect')
def on_disconnect(sid):
    print("Client %s dis-connected" % sid)
    global Connection_count
    Connection_count -= 1
    pass


@sio.on('ready')
def on_ready(sid, data_no_use):
    print("Received ready from client %s" % sid)

    # all expected clients connected, kick start the loop
    while not Connection_count == Expected_client_number:
        if config.verbosity > 0:
            print("waiting for client: %d connected, %d expected " % (Connection_count, Expected_client_number))
        time.sleep(1)

    print("kick start to on_data_ready!")

    # all clients ready, kick start the loop
    info = on_data_ready(sid, 'dummy_data')

    info = list(info)

    more_info = [Client_ids[str(sid)], Expected_client_number]

    for i in more_info:
        info.append(i)

    return tuple(info)


@sio.on('data_ready')
def on_data_ready(sid, data_no_use):
    global model_version
    print("Received data ready from client %s" % sid)

    # # remove old data in local folder
    # data_folder = home + "/Raw_driving_data"
    # clear_old_data(root_folder=data_folder)

    global Data_ready_count, Model_ready, new_model

    if config.dummy:
        time.sleep(1)
        model_version += 1
        return model_version, 'dummy_model'

    Data_ready_count += 1

    print("clients: %d connected, %d expected " % (Connection_count, Expected_client_number))

    if Data_ready_count < Expected_client_number:  # wait for data to be ready
        if config.verbosity > 0:
            print("Data not ready yet, Data_ready_count:", Data_ready_count)
        while not Model_ready:  # wait for model training
            time.sleep(1)
    else:  # only do this for one client
        Data_ready_count = 0

        model_version += 1

        # TODO: Change the model name conversion to something like 'rl_'

        new_model = 'trained_models/lets_drive_' + str(model_version) + '.pth'

        # model_version 0 is already trained in the imitation stage
        if model_version > 0:
            print("Training started for client ", sid)
            existing_model = 'trained_models/lets_drive_' + str(model_version - 1) + '.pth'

            # TODO: Change the data set names. For testing you can make a copy of test.h5

            # copy driving data from client
            shell_cmd = 'scp ' + Client_address + Client_nn_workspace + Client_dataset_path + 'train.h5 ' \
                        + home + nn_folder + 'train.h5'
            if config.verbosity > 0:
                print("[cmd] " + shell_cmd)
            subprocess.call(shell_cmd.split())
            shell_cmd = 'scp ' + Client_address + Client_nn_workspace + Client_dataset_path + 'val.h5 ' \
                        + home + nn_folder + 'val.h5'
            if config.verbosity > 0:
                print("[cmd] " + shell_cmd)
            subprocess.call(shell_cmd.split())
            shell_cmd = 'scp ' + Client_address + Client_nn_workspace + Client_dataset_path + 'test.h5 ' \
                        + home + nn_folder + 'test.h5'
            if config.verbosity > 0:
                print("[cmd] " + shell_cmd)
            subprocess.call(shell_cmd.split())

            # start training for imitation learning
            # load the existing model (existing_model) and train a new model (new_model)
            # TODO: Change this supervised learning code to calling PPO

            shell_cmd = 'python3 train.py --batch_size 128 --lr ' + str(cmd_args.lr) + ' --train train.h5 --val val.h5 --no_vin 0 ' \
                        '--l_h 100 --vinout 28 --w 64 --fit action --ssm 0.03 ' \
                        '--logdir runs/lets_drive_' + str(model_version) + ' ' \
                        '--resume ' + existing_model + ' ' \
                        '--modelfile ' + new_model + \
                        ' --moreepochs 1' + \
                        ' --exactname 1'

            if config.verbosity > 0:
                print("[cmd] " + shell_cmd)

            subprocess.call(shell_cmd.split(), cwd=home + nn_folder)
            print("Training finished")

        else:
            new_model = new_model

        # copy new model to the client
        shell_cmd = 'scp ' + home + nn_folder + new_model + \
                    ' ' + Client_address + \
                    Client_nn_workspace + \
                    new_model
        if config.verbosity > 0:
            print("[cmd] " + shell_cmd)
        subprocess.call(shell_cmd.split())

        Model_ready = True

    # signal the client to evaluate the new model
    # inform it with the name of the new model

    return model_version, new_model

    # shell_cmd = 'rsync -avz Unicorn1:/home/panpan/result/ ' + data_folder
    # data_streaming_proc = subprocess.Popen(shell_cmd.split())
    #
    # time.sleep(1)
    # while check_process(data_streaming_proc, '[on_data_ready] data streaming') is False:
    #     time.sleep(1)
    #     continue
    #
    # print("Data streaming setup")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--verb',
                        type=int,
                        default=0,
                        help='Verbosity')

    parser.add_argument('--dummy',
                        type=int,
                        default=0,
                        help='dummmy run flag')

    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate for training')

    cmd_args = parser.parse_args()
    config.verbosity = cmd_args.verb
    config.dummy = cmd_args.dummy

    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 8080)), app)
