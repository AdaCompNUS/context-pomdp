import os
from os.path import expanduser
import subprocess

home = expanduser("~")

nn_folder = '/workspace/BTS_RL_NN/catkin_ws/src/drive_net_controller/src/BTS_RL_NN/'

if not os.path.isdir(home + nn_folder):
    nn_folder = '/workspace/catkin_ws/src/drive_net_controller/src/BTS_RL_NN/'

if not os.path.isdir(home + nn_folder):
    nn_folder = '/catkin_ws/src/drive_net_controller/src/BTS_RL_NN/'


catkin_ws_path = home + '/workspace/catkin_ws/'

if not os.path.isdir(catkin_ws_path):
    catkin_ws_path = home + '/catkin_ws/'

result_path = home + '/result'


# check whether the folders exist:
if not os.path.isdir(result_path):
	os.makedirs(result_path)
	print("Made result folder " + result_path)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('--image',
						type=str,
						default="cppmayo/lets_drive_docker:lib-torch-and-opencv-added",
						help='Image to launch')

	config = parser.parse_args()


	additional_mounts = "-v " + nn_folder + "trained_models/:/home/panpan/workspace/catkin_ws/src/drive_net_controller/src/BTS_RL_NN/trained_models/ " + \
				"-v " + nn_folder + "dataset/:/home/panpan/workspace/catkin_ws/src/drive_net_controller/src/BTS_RL_NN/dataset/ " + \
				"-v " + nn_folder + "h5/:/home/panpan/workspace/catkin_ws/src/drive_net_controller/src/BTS_RL_NN/h5/ " + \
				"-v " + nn_folder + "Data_processing/visualize/:/home/panpan/workspace/catkin_ws/src/drive_net_controller/src/BTS_RL_NN/Data_processing/visualize/ "

        #                        "--user summit:sudo" + \
	cmd_args = "docker run --runtime=nvidia -it " + \
	            "--net=host " + \
				"-e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix " + \
                "-v " + result_path + ":/home/panpan/Unity/DESPOT-Unity/result " + \
                config.image + " bash"

	subprocess.call(cmd_args.split())
