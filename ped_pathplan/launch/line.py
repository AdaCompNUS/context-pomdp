from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

max_y = 0

class LineBuilder:
    def __init__(self, line):
        self.line = line
        # self.xs = list(line.get_xdata())
        # self.ys = list(line.get_ydata())
        self.xs = []
        self.ys = []
        #self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self.draw_line)
        self.clear_points = line.figure.canvas.mpl_connect('scroll_event', self.next_obstacle)

    #def __call__(self, event):
    def draw_line(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        print event.xdata, ' ', max_y - event.ydata
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        plt.plot(self.xs, self.ys)

    def next_obstacle(self, event):
        self.xs.append(self.xs[0])
        self.ys.append(self.ys[0])
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        plt.plot(self.xs, self.ys)
        self.xs = []
        self.ys = []

#img = mpimg.imread('utown_momdp_small_cleaningblock2.png')
img = mpimg.imread('market.pgm')
max_y = len(img)

imgplot = plt.imshow(img)
line, = plt.plot([0], [0])  # empty line
print line
linebuilder = LineBuilder(line)

plt.show()