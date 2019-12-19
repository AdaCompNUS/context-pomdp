import os, sys
import fnmatch
import argparse
import numpy as np
import math, pdb

import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import animation

sx=20.0
sy=20.0
fig = plt.figure()
plt.axis([-sx,sx,-sy,sy])
ax = plt.gca()
ax.set_aspect(sy/sx)

def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)

def parse_data(txt_file):
    ego_list = {}
    exos_list = {}
    coll_bool_list = {}
    ego_path_list = {}

    exo_count = 0
    start_recording = False

    with open(txt_file, 'r') as f:
        for line in f:

            if 'Round 0 Step' in line:
                line_split = line.split('Round 0 Step ', 1)[1]
                cur_step = int(line_split.split('-', 1)[0])
                start_recording = True

            if not start_recording:
                continue

            try:

                if "car pos / heading / vel" in line:  # ego_car info
                    # car pos / heading / vel = (162.54, 358.42) / 2.1142 / -6.1142e-10 car dim 2.0091 4.8541
                    speed = float(line.split(' ')[12])
                    heading = float(line.split(' ')[10])
                    pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                    pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                    bb_x = float(line.split(' ')[15])
                    bb_y = float(line.split(' ')[16])

                    pos = [pos_x, pos_y]

                    agent_dict = {'pos': [pos_x, pos_y],
                                    'heading': heading,
                                    'speed': speed,
                                    'bb': (bb_x, bb_y)
                                    }
                    ego_list[cur_step] = agent_dict

                elif " pedestrians" in line: # exo_car info start
                    exo_count = int(line.split(' ')[0])
                    exos_list[cur_step] = []
                elif "id / pos / speed / vel / intention / dist2car / infront" in line: # exo line, info start from index 16
                    # agent 0: id / pos / speed / vel / intention / dist2car / infront =  54288 / (99.732, 462.65) / 1 / (-1.8831, 3.3379) / -1 / 9.4447 / 0 (mode) 1 (type) 0 (bb) 0.90993 2.1039 (cross) 1 (heading) 2.0874
                    line_split = line.split(' ')
                    agent_id = int(line_split[16+1])

                    pos_x = float(line_split[18+1].replace('(', '').replace(',', ''))
                    pos_y = float(line_split[19+1].replace(')', '').replace(',', ''))
                    pos = [pos_x, pos_y]

                    vel_x = float(line_split[23+1].replace('(', '').replace(',', ''))
                    vel_y = float(line_split[24+1].replace(')', '').replace(',', ''))
                    vel = [vel_x, vel_y]

                    bb_x = float(line_split[36+1])
                    bb_y = float(line_split[37+1])

                    heading = float(line_split[41+1])

                    agent_dict = {  'id': agent_id,
                                    'pos': [pos_x, pos_y],
                                    'heading': heading,
                                    'vel': [vel_x, vel_y],
                                    'bb': (bb_x, bb_y)
                    }

                    exos_list[cur_step].append(agent_dict)
                    assert(len(exos_list[cur_step]) <= exo_count)
                elif "Path: " in line: # path info
                    # Path: 95.166 470.81 95.141 470.86 ...
                    line_split = line.split(' ')
                    path = []
                    for i in range(1, len(line_split)-1, 2):
                        x = float(line_split[i])
                        y = float(line_split[i+1])
                        path.append([x,y])
                    ego_path_list[cur_step] = path
                if 'collision = 1' in line or 'INININ' in line or 'in real collision' in line:
                    coll_bool_list[cur_step] = 1

            except Exception as e:
                error_handler(e)
                pdb.set_trace()

    return ego_list, ego_path_list, exos_list, coll_bool_list


def agent_rect(agent_dict, origin, color):
    pos = agent_dict['pos']
    heading = agent_dict['heading']
    bb_x, bb_y = agent_dict['bb']

    x_shift = [bb_y/2.0 * math.cos(heading), bb_y/2.0 * math.sin(heading)]
    y_shift = [-bb_x/2.0 * math.sin(heading), bb_x/2.0 * math.cos(heading)]
    
    # y_shift = -bb_y/2.0 * math.sin(heading) + bb_x/2.0 * math.cos(heading)
    # x_shift = [0.0, 0.0]

    rect = mpatches.Rectangle(
        xy=[pos[0] - origin[0] - x_shift[0] - y_shift[0], pos[1] - origin[1] - x_shift[1] - y_shift[1]], 
        # xy=[pos[0] - origin[0], pos[1] - origin[1]],  
        width=bb_y , height=bb_x, angle=np.rad2deg(heading), fill=True, color=color)
    return rect


def init():
    # initialize an empty list of cirlces
    return []

def animate(time_step):
    patches = []

    print("Drawing time step {}...".format(time_step))

    ego_pos = ego_list[time_step]['pos']
    # draw ego car
    if time_step in coll_bool_list.keys():
        ego_color = 'red'
    else:
        ego_color = 'green'
    patches.append(ax.add_patch(
        agent_rect(ego_list[time_step], ego_pos, ego_color)))

    # draw exo agents
    for agent_dict in exos_list[time_step]:
        patches.append(ax.add_patch(
            agent_rect(agent_dict, ego_pos, 'black')))

    # draw path
    path = ego_path_list[time_step]
    for i in range(0, len(path), 2):
        point = path[i]
        patches.append(ax.add_patch(
            mpatches.Circle([point[0]-ego_pos[0], point[1]-ego_pos[1]],
                                 0.1, color='orange')
            ))

    return patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
        default='some_txt_file',
        help='File to animate')

    ego_list, ego_path_list, exos_list, coll_bool_list = parse_data(parser.parse_args().file)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(ego_list.keys()), interval=300, blit=True)
    plt.show()



