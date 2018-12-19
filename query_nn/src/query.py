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

	if mode == 'all':
		print("all mode")

		print("view")
		pt_tensor_from_list = torch.FloatTensor(data.tensor).view(bs,9,32,32)

		print("cuda")
		input = pt_tensor_from_list.to(device)

		print("forward")
		value, acc_pi, acc_mu, acc_sigma, \
		ang, _, _ = drive_net.forward(input)

		value = value.cpu().view(-1).detach().numpy().tolist()
		acc_pi = acc_pi.cpu().view(-1).detach().numpy().tolist()
		acc_mu = acc_mu.cpu().view(-1).detach().numpy().tolist()
		acc_sigma = acc_sigma.cpu().view(-1).detach().numpy().tolist()
		ang = ang.cpu().view(-1).detach().numpy().tolist()
	else:
		print("only support all mode")

	print(mode+" model forward time: " + str(time.time()-start) +  's')

	return value, acc_pi, \
			acc_mu, acc_sigma, ang


if __name__ == '__main__':
	model_mode = ''

	test_model_name = "/home/yuanfu/Unity/DESPOT-Unity/torchscript_version.pt"
	if not os.path.isdir(test_model_name):
		test_model_name = "/home/panpan/Unity/DESPOT-Unity/torchscript_version.pt"

	try:
		drive_net = torch.jit.load(test_model_name).cuda(device)
		model_mode='jit'
	except Exception as e:
		pass

	if model_mode is '':
		drive_net = torch.load(test_model_name).cuda(device)
		model_mode='torch'

	print("drive_net model mode:", model_mode)

	rospy.init_node('nn_query_node')
	s = rospy.Service('query', TensorData, on_data_ready)
	print ("Ready to query nn.")
	rospy.spin()