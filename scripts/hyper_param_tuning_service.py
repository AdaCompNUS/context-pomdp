#!/usr/bin/env python3

import sys
import os

from pathlib import Path
import Pyro4
import subprocess
import numpy
import pdb
import random

# ws_root = Path(os.path.realpath(__file__)).parent.parent.parent
# sys.path.append(str(ws_root/'il_controller'/'src'))
# sys.path.append(str(ws_root/'reinforcement'/'src'))
# from params import *

from torch.utils.tensorboard import SummaryWriter


def error_handler(e):
    print(
        '\nError on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)
    # exit(-1)


def print_flush(msg):
    print(msg)
    sys.stdout.flush()


KP_RANGE = (0.2, 2.0)
KP_DELTA = 0.2
KD_RANGE = (0.2, 2.0)
KD_DELTA = 0.2
KI_RANGE = (0.2, 2.0)
KI_DELTA = 0.2

initial = True

BATCH = 10
RESTART_MAX = 0


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class HyperParamService():
    def __init__(self):
        # self.writer = SummaryWriter('runs/{}'.format(SUB_FOLDER))
        self.init_kp = 1.0
        self.init_ki = 0.5
        self.init_kd = 0.2
        self.cur_kp = self.init_kp
        self.cur_ki = self.init_ki
        self.cur_kd = self.init_kd
        self.kp_star = self.cur_kp
        self.ki_star = self.cur_ki
        self.kd_star = self.cur_kd
        
        self.neighbours = []
        self.evaluated_neighbours = []
        self.evaluated = {}
        self.cur_score_min = 1000000

        self.cmd_speeds_trial = []
        self.cur_speeds_trial = []

        self.iter = 0
        self.batch_max_error = 0
        self.batch_mean_error = 0
        self.vehicles = ['vehicle.volkswagen.t2',
                        'vehicle.bmw.isetta',
                        'vehicle.carlamotors.carlacola',
                        'vehicle.jeep.wrangler_rubicon',
                        'vehicle.nissan.patrol',
                        'vehicle.tesla.cybertruck',
                        'vehicle.yamaha.yzf',
                        'vehicle.bh.crossbike']
        self.vehicle_idx = 0
        self.seed_restart_count = 0

    def init_seed(self):
        self.init_kp = round(random.uniform(KP_RANGE[0], KP_RANGE[1]),1)
        self.init_ki = round(random.uniform(KI_RANGE[0], KI_RANGE[1]),1)
        self.init_kd = round(random.uniform(KD_RANGE[0], KD_RANGE[1]),1)

        self.cur_kp = self.init_kp
        self.cur_ki = self.init_ki
        self.cur_kd = self.init_kd

        self.neighbours = []
        self.evaluated_neighbours = []
        
        self.cmd_speeds_trial = []
        self.cur_speeds_trial = []

        self.iter = 0
        self.batch_max_error = 0
        self.batch_mean_error = 0

        self.kp_star = self.cur_kp
        self.ki_star = self.cur_ki
        self.kd_star = self.cur_kd
        self.cur_score_min = 1000000

    def initialize(self):
        self.init_kp = 1.0
        self.init_ki = 0.5
        self.init_kd = 0.2

        self.cur_kp = self.init_kp
        self.cur_ki = self.init_ki
        self.cur_kd = self.init_kd

        self.neighbours = []
        self.evaluated_neighbours = []

        self.cmd_speeds_trial = []
        self.cur_speeds_trial = []

        self.iter = 0
        self.batch_max_error = 0
        self.batch_mean_error = 0

        self.evaluated = {}
        self.kp_star = self.cur_kp
        self.ki_star = self.cur_ki
        self.kd_star = self.cur_kd
        self.cur_score_min = 1000000

    def propose_neighbours(self, kp, ki, kd):
        try:
            self.neighbours = []
            self.neighbours.append([kp + KP_DELTA, ki, kd])
            self.neighbours.append([kp - KP_DELTA, ki, kd])
            self.neighbours.append([kp, ki + KI_DELTA, kd])
            self.neighbours.append([kp, ki - KI_DELTA, kd])
            self.neighbours.append([kp, ki, kd + KD_DELTA])
            self.neighbours.append([kp, ki, kd - KD_DELTA])

            for i, neighbour in enumerate(self.neighbours):
                kp = neighbour[0]
                ki = neighbour[1]
                kd = neighbour[2]
                flag = '{}_{}_{}'.format(kp, ki, kd)
                if flag in self.evaluated:
                    del self.neighbours[i]
                    continue
                if kp < KP_RANGE[0] or kp > KP_RANGE[1]:
                    del self.neighbours[i]
                    continue
                if ki < KI_RANGE[0] or ki > KI_RANGE[1]:
                    del self.neighbours[i]
                    continue
                if kd < KD_RANGE[0] or kd > KD_RANGE[1]:
                    del self.neighbours[i]
                    continue

                self.neighbours[i][0] = round(neighbour[0],1)
                self.neighbours[i][1] = round(neighbour[1],1)
                self.neighbours[i][2] = round(neighbour[2],1)

            print_flush('proposed {} neighbours'.format(len(self.neighbours)))

        except Exception as e:
            error_handler(e)
            pdb.set_trace()
            sys.exit(0)


    def next_param(self, kp, ki, kd, result_max, result_mean):
        result = result_mean
        try:
            self.evaluated['{}_{}_{}'.format(kp, ki, kd)] = result
            self.evaluated_neighbours.append([kp, ki, kd, result])

            if len(self.neighbours) == 0:
                error_min = 10000
                candidate_best = []

                for evaluation in self.evaluated_neighbours:
                    error = evaluation[3]
                    if error < error_min:
                        candidate_best = evaluation[0:3]
                        error_min = error

                if error_min <= self.cur_score_min:
                    self.kp_star = candidate_best[0]
                    self.ki_star = candidate_best[1]
                    self.kd_star = candidate_best[2]
                    self.cur_score_min = error_min
                    print_flush('==> best params {} {} {}'.format(self.kp_star, self.ki_star, self.kd_star))

                else: # termination
                    print_flush('==> termination for vehicle {} at {} {} {}!'.format(self.vehicles[self.vehicle_idx],
                        self.kp_star, self.ki_star, self.kd_star))
                    return None

                self.evaluated_neighbours = []
                self.propose_neighbours(self.kp_star, self.ki_star, self.kd_star)

            if len(self.neighbours) == 0:
                print_flush('==> termination for vehicle {} at {} {} {}!'.format(self.vehicles[self.vehicle_idx],
                    self.kp_star, self.ki_star, self.kd_star))
                return None
            else:
                print_flush('neighbours remaining for test: {}'.format(len(self.neighbours)))
                return self.neighbours.pop(0)
        except Exception as e:
            error_handler(e)
            pdb.set_trace()
            sys.exit(0)

    def next_vehicle(self):
        try:
            if self.seed_restart_count < RESTART_MAX:
                print_flush('==> resetting intial params!')

                self.init_seed()
                self.seed_restart_count += 1
            else:
                self.vehicle_idx += 1
                if self.vehicle_idx >= len(self.vehicles):
                    exit(-1)
                print_flush('==> tuning for vehicle {}!'.format(self.vehicles[self.vehicle_idx]))

                self.initialize()
                self.seed_restart_count = 0

            return self.vehicles[self.vehicle_idx]    
        except Exception as e:
            error_handler(e)
            pdb.set_trace()
            sys.exit(0)    

    def get_pid_params(self):
        try:
            file_name = "{0}_{1:1.1f}_{2:1.1f}_{3:1.1f}.txt".format(self.vehicles[self.vehicle_idx].replace('.', '_'), round(self.cur_kp, 1), round(self.cur_ki, 1), round(self.cur_kd, 1))
            global initial

            while self.iter == 0 and os.path.isfile(file_name):
                with open(file_name, 'r') as file:
                    initial = False
                    lines = file.readlines()
                    self.batch_max_error = float(lines[0])
                    self.batch_mean_error = float(lines[1])
                    print_flush('=> min_error load from file {} {}'.format(file_name, self.batch_max_error))
                    print_flush('=> mean_error load from file {} {}'.format(file_name, self.batch_mean_error))
                next_params = self.next_param(self.cur_kp, self.cur_ki, self.cur_kd, self.batch_max_error, self.batch_mean_error)
                if next_params is None:  # termination
                    self.next_vehicle()
                else:
                    self.cur_kp = next_params[0]
                    self.cur_ki = next_params[1]
                    self.cur_kd = next_params[2]

                file_name = "{0}_{1:1.1f}_{2:1.1f}_{3:1.1f}.txt".format(self.vehicles[self.vehicle_idx].replace('.', '_'), round(self.cur_kp, 1), round(self.cur_ki, 1), round(self.cur_kd, 1))

            if self.iter == 0:
                print_flush('==> tuning for vehicle {}!'.format(self.vehicles[self.vehicle_idx]))
                print_flush('=> testing new params {} {} {}'.format(self.cur_kp, self.cur_ki, self.cur_kd))

            if initial:
                initial = False
                return self.cur_kp, self.cur_ki, self.cur_kd

            if len(self.cmd_speeds_trial) == 0 and len(self.cur_speeds_trial) == 0:
                # sys.exit(0)
                return self.cur_kp, self.cur_ki, self.cur_kd

            if len(self.cmd_speeds_trial) < 200:
                return self.cur_kp, self.cur_ki, self.cur_kd

            self.iter += 1

            max_error = numpy.absolute(numpy.asarray(self.cmd_speeds_trial) - numpy.asarray(self.cur_speeds_trial)).max()
            mean_error = numpy.absolute(numpy.asarray(self.cmd_speeds_trial) - numpy.asarray(self.cur_speeds_trial)).mean()

            self.batch_max_error = (self.batch_max_error * (self.iter - 1) + max_error) / self.iter 
            self.batch_mean_error = (self.batch_mean_error * (self.iter - 1) + mean_error) / self.iter 

            print('trial error max {} mean {} with {} data points'.format(max_error, mean_error, len(self.cmd_speeds_trial)))
            print_flush('=> min_error after trial {} {}'.format(self.iter, self.batch_max_error))
            print_flush('=> mean_error after trial {} {}'.format(self.iter, self.batch_mean_error))
            self.cmd_speeds_trial= []
            self.cur_speeds_trial = []
                
            if self.iter < BATCH:
                return self.cur_kp, self.cur_ki, self.cur_kd
            else:
                if not os.path.isfile(file_name):
                    with open(file_name, 'w') as file:
                        file.write("{}\n".format(self.batch_max_error))
                        file.write("{}".format(self.batch_mean_error))

                self.iter = 0

                next_params = self.next_param(self.cur_kp, self.cur_ki, self.cur_kd, self.batch_max_error, self.batch_mean_error)
                if next_params is None:  # termination
                    self.next_vehicle()
                else:
                    self.cur_kp = next_params[0]
                    self.cur_ki = next_params[1]
                    self.cur_kd = next_params[2]
                    self.batch_max_error = 0
                    self.batch_mean_error = 0
                    return self.cur_kp, self.cur_ki, self.cur_kd
        except Exception as e:
            error_handler(e)
            pdb.set_trace()
            sys.exit(0)

    def get_vehicle_model(self):
        return self.vehicles[self.vehicle_idx]

    def record_vels(self, cmd_speed, cur_speed):
        self.cmd_speeds_trial.append(cmd_speed)
        self.cur_speeds_trial.append(cur_speed)


def main():
    print_flush('Hyper-parameter tuning service running.')

    Pyro4.Daemon.serveSimple(
            {
                HyperParamService: "hyperparamservice.warehouse"
            },
            port=7104,
            ns=False)


if __name__ == "__main__":
    main()