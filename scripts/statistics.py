import os
import fnmatch
import argparse
import numpy as np
import math

cap = 10 

def collect_txt_files(rootpath, flag):
    txt_files = list([])
    for root, dirnames, filenames in os.walk(rootpath):

        if flag in root and ignore_flag not in root and 'debug' not in root:
            # print("subfolder %s found" % root)
            for filename in fnmatch.filter(filenames, '*.txt'):
                # append the absolute path for the file
                txt_files.append(os.path.join(root, filename))
    print("%d files found in %s" % (len(txt_files), rootpath))
    return txt_files


def filter_txt_files(root_path, txt_files):
    # container for files to be converted to h5 data
    filtered_files = list([])
    
    no_aa_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    for txtfile in txt_files:
        ok_flag = False
        no_aa = False
        with open(txtfile, 'r') as f:
            for line in reversed(list(f)):
                if 'Step {}'.format(cap + 1) in line or 'step {}'.format(cap + 1) in line:
                    ok_flag = True
                if 'No agent array messages received after' in line:
                    no_aa_count += 1 
                    no_aa = True 
        if ok_flag == True:
            filtered_files.append(txtfile)
            # print("good file: ", txtfile)
        else:
            if no_aa:
                pass # print("no aa file: ", txtfile)
            else:
                pass # print("unused file: ", txtfile)
    print("NO agent array in {} files".format(no_aa_count))

    filtered_files.sort()
    print("%d filtered files found in %s" % (len(filtered_files), root_path))
    # print (filtered_files, start_file, end_file)
    #
    return filtered_files


def get_statistics(root_path, filtered_files):
    total_count = len(filtered_files)
    col_count = 0
    goal_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    eps_step = []
    goal_step = []
    ave_speeds = [] 
    dec_counts = []
    acc_counts = []
    mat_counts = []
    trav_dists= []
    for txtfile in filtered_files:
        #
        reach_goal_flag = False
        collision_flag = False
        cur_step = 0
        dec_count = 0
        acc_count = 0
        mat_count = 0
        speed = 0.0
        last_speed = 0.0
        ave_speed = 0.0
        dist = 0.0
        last_pos = None
        with open(txtfile, 'r') as f:
            for line in f:
                if 'executing step' in line:
                    line_1 = line.split('executing step ', 1)[1]
                    cur_step = int(line_1.split('=', 1)[0])
                elif 'Round 0 Step' in line:
                    line_1 = line.split('Round 0 Step ', 1)[1]
                    cur_step = int(line_1.split('-', 1)[0])
                elif 'goal reached at step' in line:
                    line_1 = line.split('goal reached at step ', 1)[1]
                    cur_step = int(line_1.split(' ', 1)[0])
                elif "porca" in folder and 'pos / yaw / speed /' in line:
                    speed = line.split(' ')[13]
                    if speed< last_speed:
                        dec_count += 1
                    last_speed = speed
                elif ("pomdp" in folder or "rollout" in folder) and 'action **=' in line:
                    acc = int(line.split(' ')[2]) % 3
                    # if acc == 2:
                        # dec_count += 1
                    if acc == 1:
                        acc_count += 1
                    elif acc == 0:
                        mat_count += 1
                elif ("pomdp" in folder or "gamma" in folder or "rollout" in folder) and "car pos / heading / vel" in line: 
                    # = (149.52, 171.55) / 1.3881 / 0.50245
                    speed = float(line.split(' ')[12])
                    pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                    pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                    if cur_step >= cap:
                        ave_speed += speed
                    pos = [pos_x, pos_y]

                    if last_pos:
                        dist += math.sqrt((pos[0]-last_pos[0])**2 + (pos[1]-last_pos[1])**2)
                    last_pos = pos
                    if "gamma" in folder or 'pomdp' in folder or "rollout" in folder:
                        if speed< last_speed - 0.2:
                            dec_count += 1
                        last_speed = speed

                elif "imitation" in folder and 'car pos / dist_trav / vel' in line:
                    speed = line.split(' ')[12]
                    if speed< last_speed:
                        dec_count += 1
                    last_speed = speed
                elif "lets-drive" in folder and 'car pos / dist_trav / vel' in line:
                    speed = line.split(' ')[12]
                    if speed< last_speed:
                        dec_count += 1
                    last_speed = speed

                if 'goal reached' in line:
                    reach_goal_flag = True
                    break    
                if ('collision = 1' in line or 'INININ' in line or 'in real collision' in line) and reach_goal_flag == False:
                    collision_flag = True
                    col_count += 1
                    break

        eps_step.append(cur_step)
        if cur_step > cap:
            ave_speed = ave_speed / (cur_step - cap)
            ave_speeds.append(ave_speed)
            dec_count  = dec_count / float(cur_step)
            acc_count  = acc_count / float(cur_step)
            mat_count  = mat_count / float(cur_step)
            dec_counts.append(dec_count)
            acc_counts.append(acc_count)
            mat_counts.append(mat_count)
            trav_dists.append(dist)
            if reach_goal_flag == True:
                goal_count+=1
                assert(cur_step != 0)
                goal_step.append(cur_step)
            else:
                pass # print("fail file: ", txtfile)
            if collision_flag == True:
                pass #col_count += 1
                # print("col file: ", txtfile)
    print("%d filtered files found in %s" % (len(filtered_files), root_path))

    # print("goal rate :", float(goal_count)/total_count)
    print("col rate :", float(col_count)/total_count)
    ave_speeds_np = np.asarray(ave_speeds)
    print("ave speed :", np.average(ave_speeds_np))
    freq = 3
    if 'porca' in folder:
        freq = 10

    # print('time to goal :', float(sum(goal_step))/len(goal_step)/freq)
    dec_np = np.asarray(dec_counts)
    acc_np = np.asarray(acc_counts)
    mat_np = np.asarray(mat_counts)
    print('dec_count:', np.average(dec_np))
    print('acc_count:', np.average(acc_np))
    print('mat_count:', np.average(mat_np))
    trav_np = np.asarray(trav_dists)
    print('travelled dist:', np.average(trav_np))
    print('travelled dist total:', np.sum(trav_np))
    print("col rate per meter:", float(col_count)/np.sum(trav_np))
    print("col rate per step:", float(col_count)/np.sum(eps_step))
    # print (filtered_files, start_file, end_file)
    #


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--flag',
        type=str,
        default='test',
        help='Folder name to track')
    parser.add_argument(
        '--ignore',
        type=str,
        default='map_test',
        help='folder flag to ignore')
    parser.add_argument(
        '--folder',
        type=str,
        default='./',
        help='Subfolder to check')

    flag = parser.parse_args().flag
    folder = parser.parse_args().folder
    ignore_flag = parser.parse_args().ignore

    files = collect_txt_files(folder, flag)
    filtered = filter_txt_files(folder, files)
    get_statistics(folder, filtered)




