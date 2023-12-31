import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING, LIDAR_DISTANCE_CAP, MAX_GOAL_DISTANCE, MAX_SUBGOAL_DISTANCE, THREHSOLD_GOAL

from ..common.ounoise import OUNoise
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES

from .off_policy_agent import OffPolicyAgent, Network
from ..common.utilities import filter_scan


# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

class Actor(Network):
    def __init__(self, name, state_size, goal_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        self.name = name

        # --- define layers here ---
        self.fa1 = nn.Linear(state_size-14, hidden_size[0])
        self.fa2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.fa3 = nn.Linear(hidden_size[1]+14+goal_size, hidden_size[1])
        self.acc_mean = nn.Linear(hidden_size[1], action_size)
        self.acc_logstd = nn.Linear(hidden_size[1], action_size)

        self.LOG_SIG_MIN, self.LOG_SIG_MAX = -20, 2
        # --- define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, goals):
        # --- define forward pass here ---
        scan = states[:,:-14]
        other = states[:,-14:]

        # scan
        x1 = torch.relu(self.fa1(scan))
        scan_out = torch.tanh(self.fa2(x1))

        # concat
        concat = torch.cat([scan_out, other, goals],dim=-1)
        x3 = torch.relu(self.fa3(concat))
        acc_mean = self.acc_mean(x3)
        acc_logstd = self.acc_logstd(x3)
        acc_std = torch.clamp(acc_logstd, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
        # print("acc_mean", acc_mean)
        # print("acc_std", acc_std)
        normal = torch.distributions.Normal(acc_mean, acc_std)

        return normal

    # [-1, 1]
    def sample(self, state, goal):
        normal = self.forward(state, goal)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(normal.mean)
        return action, log_prob, mean

class Critic(Network):
    def __init__(self, name, state_size, goal_size, action_size, hidden_size):
        super(Critic, self).__init__(name)

        # --- define layers here ---
        self.l1 = nn.Linear(state_size-14, int(hidden_size[0] / 2))
        self.l2 = nn.Linear(int(hidden_size[0] / 2), 32)
        self.l3 = nn.Linear(32+14+goal_size+action_size, int(hidden_size[1] / 2))
        self.l4 = nn.Linear(int(hidden_size[1] / 2), 1)
        # --- define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, actions, goals):
        # --- define forward pass here ---
        scan = states[:,:-14]
        other = states[:,-14:]

        # scan
        xs = torch.relu(self.l1(scan))
        xss = torch.tanh(self.l2(xs))

        # concat
        concat = torch.cat([xss, other, goals, actions], dim=-1)
        x = torch.relu(self.l3(concat))
        x = self.l4(x)
        return x

class EnsembleCritic(nn.Module):
    def __init__(self, name, state_size, goal_size, action_size, hidden_size, n_Q=2):
        super(EnsembleCritic, self).__init__()
        self.name = name
        ensemble_Q = [Critic(name=name + str(i), state_size=state_size, goal_size=goal_size, action_size=action_size, \
                             hidden_size=hidden_size) for i in range(n_Q)]			
        self.ensemble_Q = nn.ModuleList(ensemble_Q)
        self.n_Q = n_Q


    def forward(self, state, action, goal):
        Q = [self.ensemble_Q[i](state, action, goal) for i in range(self.n_Q)]
        Q = torch.cat(Q, dim=-1)
        return Q
     

class DDPG(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        self.noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        self.actor = self.create_network(Actor, "actor")
        self.actor_target = self.create_network(Actor, "actor_target")
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(EnsembleCritic, "critic")
        self.critic_target = self.create_network(EnsembleCritic, "critic_target")
        self.critic_optimizer = self.create_optimizer(self.critic)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.new_subgoal = True
        self.subgoal_xy = [0.0, 0.0]

    def calculate_map(self, acc_bound):
        bias = (acc_bound[1] + acc_bound[0])/2
        mult = (acc_bound[1] - acc_bound[0])/2
        return mult, bias

    def cal_safe(self, scan, threshold=0.2):
        robot_safe = False
        tmp_index = []
        for i in range(len(scan)-20):
            ds = np.mean(scan[i:i+20])
            if ds > threshold:
                tmp_index.append(i+10)
        if len(tmp_index) > 0:
            robot_safe = True
        return robot_safe, tmp_index

    def cal_subgoal(self, robot_xy, subgoal_xy, robot_heading):
        subgoal_dis = math.sqrt((robot_xy[0]-subgoal_xy[0])**2 + (robot_xy[1]-subgoal_xy[1])**2)
        heading_to_goal = math.atan2((subgoal_xy[1] - robot_xy[1]), (subgoal_xy[0] - robot_xy[0]))
        subgoal_angle = heading_to_goal - robot_heading
        return [subgoal_dis, subgoal_angle]


    def get_action(self, state, goal, step, is_training, total_step, warm_step, play_in_rule):

        goal_angle = state[-18]*math.pi
        goal_distance = state[-19]*MAX_GOAL_DISTANCE
        goal_x = state[-2]
        goal_y = state[-1]
        robot_x = state[-5]
        robot_y = state[-4]
        robot_heading = state[-3]
        # acc_w_bound = [state[-2], state[-1]]
        pre_vel = state[-17:-14]
        scan = state[0:480]
        pre_scan = filter_scan(scan)
        # print("state", state)
        # goal_distance = state[-5]*MAX_GOAL_DISTANCE
        # ahead_distance = np.mean(state[230:250]*LIDAR_DISTANCE_CAP)
        # min_obst_distance = min(state[:480])*LIDAR_DISTANCE_CAP

        # trun around at begin if necessary
        acc = [0.0, 0.0, 0.0]
        action = [0.0, 0.0, 0.0]
        vel = [0.0, 0.0, 0.0]
        turn = False

        acc_x_b = [state[-11], state[-10]]
        acc_y_b = [state[-9], state[-8]]
        acc_w_b = [state[-7], state[-6]]

        acc_x_mult, acc_x_bias = self.calculate_map(acc_x_b)
        acc_y_mult, acc_y_bias = self.calculate_map(acc_y_b)
        acc_w_mult, acc_w_bias = self.calculate_map(acc_w_b)
        # print("acc_x_mult", acc_x_mult, "acc_x_bias", acc_x_bias)

        # update subgoal

        self.subgoal = self.cal_subgoal([robot_x,robot_y], self.subgoal_xy, robot_heading)

        # is safe or not
        robot_safe, _ = self.cal_safe(pre_scan, 0.2)
        _, tmp_index = self.cal_safe(pre_scan, 0.5)
        lengths = len(tmp_index)
        s = int(lengths/2)

        if goal_distance < 4:
            self.new_subgoal = False
            self.subgoal = [goal_distance, goal_angle]
        elif self.subgoal[0] < THREHSOLD_GOAL:
            self.new_subgoal = True
        elif self.subgoal[0] > 5:
            self.new_subgoal = True
        else:
            self.new_subgoal = False
        
        if self.new_subgoal:
            if abs(goal_angle) > math.pi/20:
                if goal_angle > 0:
                    vel = [0.0, 0.0, 0.3]
                else:
                    vel = [0.0, 0.0, -0.3]
            else:
                if not robot_safe:
                    vel = [-0.1, 0.0, 0.0]
                else:
                    angle = (tmp_index[s]/360 - 0.5)*math.pi + robot_heading
                    print("angle:", angle)
                    # distance = np.clip(pre_scan[s]*LIDAR_DISTANCE_CAP/MAX_GOAL_DISTANCE,0,1)
                    self.subgoal_xy = [robot_x + math.cos(angle)*LIDAR_DISTANCE_CAP, robot_y + math.sin(angle)*LIDAR_DISTANCE_CAP]
                    self.subgoal = self.cal_subgoal([robot_x, robot_y], self.subgoal_xy, robot_heading)
                    self.new_subgoal = False
                    print("subgoal_xy:", self.subgoal_xy)
                    print("subgoal:", self.subgoal)
            for i in range(len(vel)):
                acc[i] = (vel[i] - float(pre_vel[i]))/0.1
            action[0] = (acc[0] - acc_x_bias)/acc_x_mult
            action[1] = (acc[1] - acc_y_bias)/acc_y_mult
            action[2] = (acc[2] - acc_w_bias)/acc_w_mult

        if not play_in_rule:
            if total_step < warm_step:
                action = self.get_action_random()
                acc[0] = action[0]*acc_x_mult + acc_x_bias
                acc[1] = action[1]*acc_y_mult + acc_y_bias
                acc[2] = action[2]*acc_w_mult + acc_w_bias
                # print("random acc", acc)
            else:
                state = np.asarray(state, np.float32)
                goal = np.asarray(goal, np.float32)
                d_state = state.shape[-1]
                states = state.reshape(-1,d_state)
                d_goal = goal.shape[-1]
                goal = goal.reshape(-1, d_goal)
                with torch.no_grad():
                    state = torch.FloatTensor(states).to(self.device)
                    goal = torch.FloatTensor(goal).to(self.device)
                    action, _, mean = self.actor.sample(state, goal)
                    action = action.squeeze().detach().cpu().numpy()
                    mean = mean.squeeze().detach().cpu().numpy()
                    # print("model output", action)
                    # print("model mean", mean)

                if is_training:
                    acc[0] = float(action[0]*acc_x_mult + acc_x_bias)
                    acc[1] = float(action[1]*acc_y_mult + acc_y_bias)
                    acc[2] = float(action[2]*acc_w_mult + acc_w_bias)
                else:
                    acc[0] = float(mean[0]*acc_x_mult + acc_x_bias)
                    acc[1] = float(mean[1]*acc_y_mult + acc_y_bias)
                    acc[2] = float(mean[2]*acc_w_mult + acc_w_bias)
        else:
            if abs(self.subgoal[1]) > math.pi/20:
                if self.subgoal[1] > 0:
                    vel = [0.0, 0.0, 0.3]
                else:
                    vel = [0.0, 0.0, -0.3]
            elif not robot_safe:
                vel = [-0.1, 0.0, 0.0]
            else:
                vel = [1.0, 0.0, 0.0]
            for i in range(len(vel)):
                acc[i] = (vel[i] - float(pre_vel[i]))/0.1
            action[0] = (acc[0] - acc_x_bias)/acc_x_mult
            action[1] = (acc[1] - acc_y_bias)/acc_y_mult
            action[2] = (acc[2] - acc_w_bias)/acc_w_mult
                
        return acc, action, vel

    # TODO:随机加速度
    def get_action_random(self):
        random_x = np.random.uniform(-1.0, 1.0)
        random_y = np.random.uniform(-1.0, 1.0)
        random_yaw = np.random.uniform(-1.0, 1.0)
        random_action = [random_x, random_y, random_yaw]
        return random_action

    def sample_action_and_KL(self, state, goal):
        # Sample action and KL-divergence
        action_dist = self.actor(state, goal)
        action = action_dist.rsample()
        D_KL = action_dist.log_prob(action).sum(-1, keepdim=True)
        action = torch.tanh(action)
        return action, D_KL

    def train(self, state, goal, action, reward, state_next, done):
        # optimize critic
        with torch.no_grad():
            next_action, _, _ = self.actor_target.sample(state_next, goal)
            target_Q = self.critic_target(state_next, next_action, goal)
            target_Q = torch.min(target_Q, -1, keepdim=True)[0]
            target_Q = reward + (1.0-done) * self.discount_factor*target_Q

        Q = self.critic(state, action, goal)
        critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # self.critic_optimizer.step()

        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        action, D_KL = self.sample_action_and_KL(state, goal)

        Q = self.critic(state, action, goal)
        Q = torch.min(Q, -1, keepdim=True)[0]

        actor_loss = (self.alpha*D_KL - Q).mean()


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # self.actor_optimizer.step()

        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        return [critic_loss.mean().detach().cpu(), actor_loss.mean().detach().cpu()]
