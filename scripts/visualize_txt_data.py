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
anim_running = True


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)

def parse_data(txt_file):
    ego_list = {}
    exos_list = {}
    coll_bool_list = {}
    ego_path_list = {}
    pred_car_list = {}
    pred_exo_list = {}
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
                                    'bb': (bb_x*2, bb_y*2)
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
                elif 'predicted_car_' in line:
                    # predicted_car_0 378.632 470.888 5.541   
                    # (x, y, heading in rad)
                    line_split = line.split(' ')
                    pred_step = int(line_split[0][14:])
                    x = float(line_split[1])
                    y = float(line_split[2])
                    heading = float(line_split[3])
                    agent_dict = {'pos': [x, y],
                                'heading': heading,
                                'bb': (10.0, 10.0)
                                }
                    if pred_step == 0:
                        pred_car_list[cur_step] = []
                    pred_car_list[cur_step].append(agent_dict)

                elif 'predicted_agents_' in line:
                    # predicted_agents_0 380.443 474.335 5.5686 0.383117 1.1751
                    # [(x, y, heading, bb_x, bb_y)]
                    line_split = line.split(' ')
                    pred_step = int(line_split[0][17:])
                    if pred_step == 0:
                        pred_exo_list[cur_step] = []
                    num_agents = (len(line_split) - 1) / 5
                    agent_list = []
                    for i in range(num_agents):
                        start = 1 + i * 5
                        x = float(line_split[start])
                        y = float(line_split[start + 1])
                        heading = float(line_split[start + 2])
                        bb_x = float(line_split[start + 3])
                        bb_y = float(line_split[start + 4])
                        agent_dict = {'pos': [x, y],
                                    'heading': heading,
                                    'bb': (bb_x*2, bb_y*2) 
                                    }
                        agent_list.append(agent_dict)
                    pred_exo_list[cur_step].append(agent_list)
                if 'collision = 1' in line or 'INININ' in line or 'in real collision' in line:
                    coll_bool_list[cur_step] = 1

            except Exception as e:
                error_handler(e)
                pdb.set_trace()

    return ego_list, ego_path_list, exos_list, coll_bool_list, pred_car_list, pred_exo_list


def agent_rect(agent_dict, origin, color, fill=True):
    try:
        pos = agent_dict['pos']
        heading = agent_dict['heading']
        bb_x, bb_y = agent_dict['bb']
        x_shift = [bb_y/2.0 * math.cos(heading), bb_y/2.0 * math.sin(heading)]
        y_shift = [-bb_x/2.0 * math.sin(heading), bb_x/2.0 * math.cos(heading)]
        
        coord = [pos[0] - origin[0] - x_shift[0] - y_shift[0], pos[1] - origin[1] - x_shift[1] - y_shift[1]]
        rect = mpatches.Rectangle(
            xy=coord, 
            width=bb_y , height=bb_x, angle=np.rad2deg(heading), fill=fill, color=color)
        return rect

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

def vel_arrow(agent_dict, origin, color):
    try:
        vel = agent_dict['vel']
        arrow = mpatches.Arrow(
            x=origin[0], y=origin[1], dx=vel[0], dy=vel[1], color=color)
        return arrow

    except Exception as e:
        error_handler(e)
        pdb.set_trace()

def init():
    # initialize an empty list of cirlces
    return []

def animate(time_step):
    patches = []

    time_step =  time_step + config.frame

    print("Drawing time step {}...".format(time_step))

    ego_pos = ego_list[time_step]['pos']
    # draw ego car
    if time_step in coll_bool_list.keys():
        ego_color = 'red'
    else:
        ego_color = 'green'

    # print('ego_heading: {}'.format(ego_list[time_step]['heading']))

    patches.append(ax.add_patch(
        agent_rect(ego_list[time_step], ego_pos, ego_color)))

    # draw exo agents
    for agent_dict in exos_list[time_step]:
        patches.append(ax.add_patch(
            agent_rect(agent_dict, ego_pos, 'black')))
        patches.append(ax.add_patch(
            vel_arrow(agent_dict, ego_pos, 'grey')))

    if time_step in pred_car_list.keys():
        for car_dict in pred_car_list[time_step]:
            car_dict['bb'] = ego_list[time_step]['bb'] 
            patches.append(ax.add_patch(
                agent_rect(car_dict, ego_pos, 'lightgreen', False)))

    if time_step in pred_exo_list.keys():
        for agent_list in pred_exo_list[time_step]:
            for agent_dict in agent_list:
                patches.append(ax.add_patch(
                    agent_rect(agent_dict, ego_pos, 'grey', False)))
    # draw path
    path = ego_path_list[time_step]
    for i in range(0, len(path), 2):
        point = path[i]
        patches.append(ax.add_patch(
            mpatches.Circle([point[0]-ego_pos[0], point[1]-ego_pos[1]],
                                 0.1, color='orange')))

    return patches

def onClick(event):
    global anim_running
    if anim_running:
        anim.event_source.stop()
        anim_running = False
    else:
        anim.event_source.start()
        anim_running = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
        default='some_txt_file',
        help='File to animate')
    parser.add_argument(
        '--frame',
        type=int,
        default=0,
        help='start frame')
    config = parser.parse_args()
    ego_list, ego_path_list, exos_list, coll_bool_list, pred_car_list, pred_exo_list = parse_data(config.file)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(ego_list.keys()) - config.frame, interval=30, blit=True)
    fig.canvas.mpl_connect('button_press_event', onClick)

    plt.show()



