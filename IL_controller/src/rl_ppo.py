

import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Components import mdn


def gae(rewards, masks, values, gamma, lambd):
    
    # rewards: T x N x 1
    # masks:   T x N x 1
    # values:  (T + 1) x N x 1

    T, N, _ = rewards.size()

    advantages = torch.zeros(T, N, 1)
    advantage_t = torch.zeros(N, 1)

    for t in reveresed(range(T)):
        delta = rewards[t] = values[t + 1] * gamma * masks[t] - values[t]
        advantage_t = delta + advantage_t * gamma * lambd * masks[t]
        advantages[t] = advantage_t

    return advantages, values[: T, :] + advantages


class PPO(object):
    def __init__(self, 
                 drive_net,     # neural network
                 optimizer,     # optimizer, default: Adam
                 clip,          # clip parameter, default: 0.05
                 gamma,         # discount factor, default 0.99
                 lambd,         # GAE lambda parameter, default: 0.95
                 value_coef,    # value loss parameter, default: 1.
                 entropy_coef,  # policy entropy loss, default: 0.01
                 epoch,         # 
                 timesteps      #
                 workers_num    #
                 minibatch_num  #
                 max_grad_norm  #
                 config         # drive_net config
                 ):
        self.drive_net = drive_net
        self.drive_net_old = copy.deepcopy(self.drive_net)
        self.optimizer = optimizer
        self.clip = clip
        self.gamma = gamma
        self.lambd = lambd
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epoch = epoch
        self.horizon = horizon
        self.minibatch_num = minibatch_num
        self.minibatch_timesteps = self.timesteps / self.minibatch_num
        self.max_grad_norm = max_grad_norm
        self.timesteps = timesteps
        self.workers_num = workers_num
        self.config = config

    def updata(self, rollouts):

        advantages, returns = gae(rollouts['rewards'], rollouts['masks'], rollouts['values'], self.gamma, self.lambd)
        advantages_mean = advantages.mean(0, True)
        advantages_std = advantages.std(0, True)
        # 
        acc = rollouts['actions']['acc']
        ang_idx = rollouts['actions']['ang_idx']

        for e in range(self.epoch):

            for minibatch_start in range(0, self.timesteps, self.minibatch_timesteps):

                # input_data: (self.minibatch_timesteps x N) x 1 x 9 x 32 x 32
                input_data_minibatch = rollouts['input_data'][minibatch_start: minibatch_start + \ 
                    self.minibatch_timesteps, :].view(self.workers_num * self.minibatch_timesteps, 1, 9, 32, 32)

                value, acc_pi, acc_mu, acc_sigma, ang_prob, _, _, _, _, _ = self.drive_net.forward(input_data_minibatch, self.config)
                _, acc_pi_old, acc_mu_old, acc_sigma_old, ang_old, _, _, _, _, _ = self.drive_net_old.forward(input_data_minibatch, self.config)

                acc_prob = mdn_accuracy(pi = acc_pi, sigma = acc_sigma, mu = \ 
                    acc_mu, acc[minibatch_start: minibatch_start + self.minibatch_timesteps].view(self.workers_num * self.minibatch_timesteps, -1))
                acc_prob_old = mdn_accuracy(pi = acc_pi_old, sigma = acc_sigma_old, mu = \ 
                    acc_mu_old, acc[minibatch_start: minibatch_start + self.minibatch_timesteps].view(self.workers_num * self.minibatch_timesteps, -1))

                ang_prob = torch.softmax(ang_prob, dim = 1)
                ang_prob_old = torch.softmax(ang_prob_old, dim = 1)

                ang_episode_idx = ang_idx[minibatch_start: minibatch_start + self.minibatch_timesteps, :]
                ang_episode_idx = ang_episode_idx.view(self.workers_num * self.minibatch_timesteps, -1)

                ratio = (acc_prob / acc_prob_old) * (ang_prob[ang_episode_idx] / ang_prob_old[ang_episode_idx])
                advantage = advantages[minibatch_start: minibatch_start + self.minibatch_timesteps, :]
                advantage = advantage.view(self.minibatch_timesteps * self.workers_num, -1)

                advantage = (advantage - advantages_mean) / (advantages_std + 1e-10)

                surr1 = ratio * advantages.view(self.workers_num * self.timesteps, 1)
                surr2 = torch.clamp(ratio, min = 1. - clip, max = 1. + clip) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (.5 * (value - returns[minibatch_start: minibatch_start + \ 
                    self.minibatch_timesteps, :].view(self.workers_num * self.minibatch_timesteps), -1) ** 2.).mean()

                ang_X_dim, ang_Y_dim = ang.size()
                acc_X_dim, acc_Y_dim = ang_X_dim, 3
                acc_candidate = torch.ones(acc_X_dim, acc_Y_dim)
                acc_candidate[:, 0], acc_candidate[:, 1], acc_candidate[:, 2] = -1., 0., 1.
  
                acc_prob = mdn_accuracy(pi = acc_pi, sigma = acc_sigma, mu = acc_mu, acc_candidate.view(ang_X_dim * 3, 1))
                acc_prob = acc_prob.view(ang_X_dim, 3)

                action_prob = torch.ones(ang_X_dim, ang_Y_dim * acc_Y_dim)
                action_prob[:, 0: ang_Y_dim] = acc_candidate_prob[:, 0] * ang_prob
                action_prob[:, ang_Y_dim : 2 * ang_Y_dim] = acc_candidate_prob[:, 1] * ang_prob
                action_prob[:, 2 * ang_Y_dim : 3 * ang_Y_dim] = acc_candidate_prob[:, 2] * ang_prob
                entropy_loss = (action_prob * F.log(action_prob, dim = 1)).sum(1).mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.drive_net.parameters(), self.max_grad_norm)
                self.optimizer.step()




