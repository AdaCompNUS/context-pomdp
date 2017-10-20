from __future__ import print_function
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys

img_height = 0
origin_x = 0
origin_y = 0
resolution = 0.0
output_file=open("obstacles.txt", "wr")

class LineBuilder:
    def __init__(self, line):
        self.line = line
        # self.xs = list(line.get_xdata())
        # self.ys = list(line.get_ydata())
        self.xs = []
        self.ys = []
        #self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self.draw_line)
        self.clear_points = line.figure.canvas.mpl_connect('key_release_event', self.next_obstacle)

    #def __call__(self, event):
    def draw_line(self, event):
        #print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        plt.plot(self.xs, self.ys)

    def next_obstacle(self, event):
    	#print ('obstacle:', end=' ')
    	for i in range(len(self.xs)):
        	print (self.xs[i] * resolution + origin_x, (img_height-self.ys[i]) * resolution + origin_y, end=' ', file = output_file)
        else:
            print ('', file = output_file)
        self.xs.append(self.xs[0])
        self.ys.append(self.ys[0])
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        plt.plot(self.xs, self.ys)
        
        self.xs = []
        self.ys = []

def get_param(filename):
    file_handler = open(filename,'r')
    content = file_handler.read().splitlines()
    for line in content:
        if "image" in line:
            tmp_img_name = line.split(' ')[1]
        if "origin" in line:
        	tmp_origin_x = float(line.split(' ')[1][1:-1])
        	tmp_origin_y = float(line.split(' ')[2][0:-1])
        if "resolution" in line:
            tmp_resolution = float(line.split(' ')[1])
    return tmp_img_name, tmp_origin_x, tmp_origin_y, tmp_resolution

if __name__ == '__main__' :
    param_file_name = 'market.yaml'
    if len(sys.argv) > 1:
        param_file_name = sys.argv[1]
    img_name, origin_x, origin_y, resolution = get_param(param_file_name)
    print (img_name, origin_x, origin_y)
    img = cv2.imread(img_name)
    img_height = len(img)
    print (img_height)
    imgplot = plt.imshow(img)
    line, = plt.plot([0], [0])  # empty line
    linebuilder = LineBuilder(line)
    plt.show()