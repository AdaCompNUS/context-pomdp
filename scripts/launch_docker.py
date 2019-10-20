import os
from os.path import expanduser
import subprocess

home = expanduser("~")

catkin_ws_path = home + '/workspace/catkin_ws'

if not os.path.isdir(catkin_ws_path):
    catkin_ws_path = home + '/catkin_ws'

result_path = home + '/driving_data'


# check whether the folders exist:
if not os.path.isdir(result_path):
	os.makedirs(result_path)
	print("Made result folder " + result_path)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('--image',
						type=str,
						default="cppmayo/melodic_cuda10_1_cudnn7_libtorch_opencv4_ws",
						help='Image to launch')

	config = parser.parse_args()


	additional_mounts = "-v " + catkin_ws_path + ":/root/catkin_ws "

	cmd_args = "docker run --runtime=nvidia -it " + \
				"-v " + result_path + ":/root/driving_data " + \
                                additional_mounts + \
				"-e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix " + \
				config.image + " bash " 

	subprocess.call(cmd_args.split())
