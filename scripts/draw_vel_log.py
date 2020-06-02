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
    cmd_steer_ts = []
    cmd_steer = []
    cur_steer_ts = []
    cur_steer = []
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
                elif 'cmd_steer' in line:
                    line_split = line.split(' ')
                    cmd_steer.append(float(line_split[1])) 
                    cmd_steer_ts.append(float(line_split[2]))
                elif 'cur_steer' in line:
                    line_split = line.split(' ')
                    cur_steer.append(float(line_split[1])) 
                    cur_steer_ts.append(float(line_split[2]))
            except Exception as e:
                error_handler(e)
                # break
                # pdb.set_trace()

    return cmd_vel, cur_vel, cmd_steer, cur_steer, cmd_vel_ts, cur_vel_ts, cmd_steer_ts, cur_steer_ts


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
    cmd_vel, cur_vel, cmd_steer, cur_steer, cmd_vel_ts, cur_vel_ts, cmd_steer_ts, cur_steer_ts = parse_data(config.file)

    cmd_vel = np.asarray(cmd_vel)
    cur_vel = np.asarray(cur_vel)
    cmd_steer = np.asarray(cmd_steer)
    cur_steer = np.asarray(cur_steer)
    diff_vel = np.absolute(cmd_vel - cur_vel)
    diff_steer = np.absolute(cmd_steer - cur_steer)

    print(np.mean(diff_vel))
    # print(steer_err_ts)
    # print(np.mean(diff_steer))

    # print('max_diff: {}'.format(np.max(diff_vel)))
    # print('ave_diff: {}'.format(np.mean(diff_vel)))

    fig, ax = plt.subplots(figsize=(30,5))
    # ts_len = min(min(len(cur_steer_ts), len(cur_steer)), len(cmd_steer_ts))
    ts_len = min(min(len(cur_vel_ts), len(cur_vel)), len(cmd_vel_ts))

    start = int(config.start * ts_len)
    end = int(config.end * ts_len)
    ax.plot(cur_vel_ts[start:end], cur_vel[start:end], color='b')
    ax.plot(cmd_vel_ts[start:end], cmd_vel[start:end], color='r')
    # ax.plot(cmd_steer_ts[start:end], cmd_steer[start:end], color='r')
    # ax.plot(cur_steer_ts[start:end], cur_steer[start:end], color='b')

    ax.set(xlabel='time (s)', ylabel='steer_error',
           title='Steer control profile')
    ax.grid()
    plt.savefig('steer_profile.png', format='png')

    plt.show()


