import os, sys, argparse, pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)


def parse_data(txt_file):
    cmd_vel_ts = []
    cmd_vel = []
    cur_vel_ts = []
    cur_vel = []
    
    with open(txt_file, 'r') as f:
        for line in f:
            try:
                if 'cmd_speed' in line:
                    line_split = line.split(' ')
                    cmd_vel.append(float(line_split[1])) 
                    cmd_vel_ts.append(float(line_split[2]))
                elif 'cur_speed' in line:
                    line_split = line.split(' ')
                    cur_vel.append(float(line_split[1])) 
                    cur_vel_ts.append(float(line_split[2]))
            except Exception as e:
                error_handler(e)
                # break
                # pdb.set_trace()

    return cmd_vel, cur_vel, cmd_vel_ts, cur_vel_ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
        default='some_txt_file',
        help='File to show')
    parser.add_argument(
        'start',
        type=float,
        default=0.0,
        help='File to show')
    parser.add_argument(
        'end',
        type=float,
        default=1.0,
        help='File to show')

    config = parser.parse_args()
    cmd_vel, cur_vel, cmd_vel_ts, cur_vel_ts = parse_data(config.file)

    cmd_vel = np.asarray(cmd_vel)
    cur_vel = np.asarray(cur_vel)
    diff_vel = np.absolute(cmd_vel - cur_vel)

    print(np.mean(diff_vel))

    # print('max_diff: {}'.format(np.max(diff_vel)))
    # print('ave_diff: {}'.format(np.mean(diff_vel)))

    fig, ax = plt.subplots(figsize=(30,5))
    ts_len = len(cur_vel_ts)
    start = int(config.start * ts_len)
    end = int(config.end * ts_len)
    ax.plot(cur_vel_ts[start:end], cur_vel[start:end], color='b')
    ax.plot(cmd_vel_ts[start:end], cmd_vel[start:end], color='r')

    ax.set(xlabel='time (s)', ylabel='speed',
           title='Speed control profile')
    ax.grid()

    plt.show()
