import torch
import socketio, eventlet
import eventlet.wsgi
from flask import Flask
import os
import sys
import rospy
from query_nn.srv import TensorData

import time

# sio = socketio.Server()
# app = Flask(__name__)


model_device_id = 0
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")


def on_data_ready(data):
	print("Received data ")
	bs = data.batchsize
	mode = data.mode

	start = time.time()
	if mode == 'val':
		print("val mode")

		print("view")
		pt_tensor_from_list = torch.FloatTensor(data.tensor).view(bs,9,32,32)

		print("cuda")
		input = pt_tensor_from_list.to(device)

		print("forward")
		value = drive_net_val.forward(input)

		value = value.cpu().view(-1).detach().numpy().tolist()
		acc_pi = []
		acc_mu = []
		acc_sigma = []
		ang = []

	else:
		print("only val mode supported")

	print(mode+" model forward time: " + str(time.time()-start) +  's')

	return value, acc_pi, \
			acc_mu, acc_sigma, ang


if __name__ == '__main__':
	model_mode_val = ''

	val_model_name = "/home/yuanfu/Unity/DESPOT-Unity/jit_val.pt"
	if not os.path.isdir(val_model_name):
		val_model_name = "/home/panpan/Unity/DESPOT-Unity/jit_val.pt"
	try:
		drive_net_val = torch.jit.load(val_model_name).cuda(device)
		model_mode_val='jit'
	except Exception as e:
		pass

	if model_mode_val is '':
		drive_net_val = torch.load(val_model_name).cuda(device)
		model_mode_val='torch'

	print("val model mode:", model_mode_val)

	rospy.init_node('val_nn_query_node')
	s = rospy.Service('query_val', TensorData, on_data_ready)
	print ("Ready to query val nn.")
	rospy.spin()