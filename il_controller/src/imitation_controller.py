#!/usr/bin/env python2

from summit_dql import SummitDQL
import carla
import rospy
import time
import sys
import os
from pathlib2 import Path
import random
import numpy as np

import std_msgs.msg
import msg_builder.msg

sys.path.append('/home/leeyiyuan/catkin_ws/src/deeplearning')

from models import ConvImitation

import torch

class ImitationController(SummitDQL):
    def __init__(self):
        super(ImitationController, self).__init__()
        
        self.model_path = rospy.get_param('~model_path', 'model.pt')
        self.gpu_num = rospy.get_param('~gpu_num', 0)
        self.device = torch.device("cuda:{}".format(self.gpu_num) if torch.cuda.is_available() else "cpu")
        
        self.model = ConvImitation()
        print('Loading existing critic model...')
        self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
        self.model.to(self.device)

        self.step = -1

        self.prev_exo_network_state_frame = self.create_frame()
        self.prev_control = [0.0, 0.0]
        self.prev_prev_control = None
        self.prev_frames = None
        self.prev_variables = None

        self.cmd_speed_pub = rospy.Publisher('/cmd_speed', std_msgs.msg.Float32, queue_size=1)
        self.cmd_steer_pub = rospy.Publisher('/cmd_steer', std_msgs.msg.Float32, queue_size=1)
        self.ego_dead_sub = rospy.Subscriber('/ego_dead', std_msgs.msg.Bool, self.ego_dead_callback, queue_size=1)

        self.update_rate = rospy.Rate(10)

    def start(self):
        while not rospy.is_shutdown():
            self.update()

    def dispose(self):
        pass
    
    def ego_dead_callback(self, ego_dead):
        rospy.signal_shutdown('Ego vehicle dead')

    def update(self):
        update_time = rospy.Time.now()

        self.step += 1
        self.update_coordinate_frame()
        
        # Get exo network state frame.
        exo_network_state_frame = self.draw_exo_network_state()

        # Get state frames
        frames = [self.draw_ego_path(), exo_network_state_frame, self.prev_exo_network_state_frame]
        
        # Write variables at current step.
        variables = self.get_variables()

        # Termination condition.
        '''
        if variables['path_length_offset'] > 10.0 or np.cos(variables['path_angle_offset']) <= 0:
            print('Ego too far from path.')
            sys.stdout.flush()
            rospy.signal_shutdown('Ego vehicle too far from path')
            return            
        '''

        # Turn frames into tensor.
        model_frames = torch.stack([torch.stack([torch.Tensor(f).to(self.device) for f in frames])])

        # Get control for previous step.
        model_prev_control = torch.stack([torch.Tensor(self.prev_control).to(self.device)])

        # Calculate control for current step.
        with torch.no_grad():
            model_output = self.model(model_frames, model_prev_control)[0].cpu().numpy().tolist()

        control = [
            np.clip(model_output[0], 0.0, 1.0),
            np.clip(model_output[2], -1.0, 1.0)
        ]

        print(control)

        self.update_rate.sleep()
        self.cmd_speed_pub.publish(control[0] * 6.0)
        self.cmd_steer_pub.publish(control[1])

        self.prev_exo_network_state_frame = exo_network_state_frame
        self.prev_prev_control = self.prev_control
        self.prev_control = control
        self.prev_frames = frames
        self.prev_variables = variables

if __name__ == '__main__':
    rospy.init_node('imitation_controller')
    imitation_controller = ImitationController()
    rospy.wait_for_message("/ego_state", msg_builder.msg.car_info)

    rospy.on_shutdown(imitation_controller.dispose)
    imitation_controller.start()
