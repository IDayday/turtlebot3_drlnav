import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING, LIDAR_DISTANCE_CAP, MAX_GOAL_DISTANCE, MAX_SUBGOAL_DISTANCE

from ..common.ounoise import OUNoise
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES

from .off_policy_agent import OffPolicyAgent


# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

class Actor(nn.Module):
    def __init__(self, name, state_size, goal_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.name = name

        # --- define layers here ---
        self.fa1 = nn.Linear(state_size-5, hidden_size[0])
        self.fa2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.fa3 = nn.Linear(hidden_size[1]+5+goal_size, hidden_size[1])
        self.acc_mean = nn.Linear(hidden_size[1], action_size)
        self.acc_logstd = nn.Linear(hidden_size[1], action_size)

        self.LOG_SIG_MIN, self.LOG_SIG_MAX = -20, 2
        # --- define layers until here ---

    def forward(self, states, goals):
        # --- define forward pass here ---
        scan = states[:,:-5]
        other = states[:,-5:]

        # scan
        x1 = torch.relu(self.fa1(scan))
        scan_out = torch.tanh(self.fa2(x1))

        # concat
        concat = torch.cat([scan_out, other, goals],dim=-1)
        x3 = torch.relu(self.fa3(concat))
        acc_mean = self.acc_mean(x3)
        acc_logstd = self.acc_logstd(x3)
        acc_std = torch.clamp(acc_logstd, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
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

class Critic(nn.Module):
    def __init__(self, state_size, goal_size, action_size, hidden_size):
        super(Critic, self).__init__()

        # --- define layers here ---
        self.l1 = nn.Linear(state_size-5, int(hidden_size[0] / 2))
        self.l2 = nn.Linear(int(hidden_size[0] / 2), 32)
        self.l3 = nn.Linear(32+5+goal_size+action_size, int(hidden_size[1] / 2))
        self.l4 = nn.Linear(int(hidden_size[1] / 2), 1)
        # --- define layers until here ---


    def forward(self, states, actions, goals):
        # --- define forward pass here ---
        scan = states[:,:-5]
        other = states[:,-5:]

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
        ensemble_Q = [Critic(state_size=state_size, goal_size=goal_size, action_size=action_size, \
                             hidden_size=hidden_size) for _ in range(n_Q)]			
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

    def get_action(self, state, goal, step, is_training, total_step, warm_step):
        goal_angle = state[-4]*math.pi
        # goal_distance = state[-5]*MAX_GOAL_DISTANCE
        # ahead_distance = np.mean(state[230:250]*LIDAR_DISTANCE_CAP)
        # min_obst_distance = min(state[:480])*LIDAR_DISTANCE_CAP

        # trun around
        action = [0.0, 0.0, 0.0]
        if abs(goal_angle) > math.pi/3 and step < 20:
            if goal_angle > 0:
                action = [0.0, 0.0, 0.5]
            else:
                action = [0.0, 0.0, -0.5]
            model_output_ = action
        else:
            if total_step < warm_step:
                model_output_ = self.get_action_random()
                action[0] = model_output_[0]*1.5
                action[1] = model_output_[1]*0.1
                action[2] = model_output_[2]*0.5
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
                    model_output_, _, model_output = self.actor.sample(state, goal)
                    model_output_ = model_output_.squeeze().detach().cpu().numpy()
                    model_output = model_output.squeeze().detach().cpu().numpy()

                if is_training:
                    action[0] = float(model_output_[0]*1.5)
                    action[1] = float(model_output_[1]*0.1)
                    action[2] = float(model_output_[2]*0.5)
                else:
                    action[0] = float(model_output[0]*1.5)
                    action[1] = float(model_output[1]*0.1)
                    action[2] = float(model_output[2]*0.5)

        # state = np.asarray(state, np.float32)
        # goal = np.asarray(goal, np.float32)
        # d_state = state.shape[-1]
        # states = state.reshape(-1,d_state)
        # d_goal = goal.shape[-1]
        # goal = goal.reshape(-1, d_goal)
        # with torch.no_grad():
        #     state = torch.FloatTensor(states).to(self.device)
        #     goal = torch.FloatTensor(goal).to(self.device)
        #     model_output_, _, model_output = self.actor.sample(state, goal)
        # action = model_output.squeeze()
        # print("action", action)
        return action, model_output_

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
            next_action, _, _ = self.actor.sample(state_next, goal)
            target_Q = self.critic_target(state_next, next_action, goal)
            target_Q = torch.min(target_Q, -1, keepdim=True)[0]
            target_Q = reward + (1.0-done) * self.discount_factor*target_Q

        Q = self.critic(state, action, goal)
        critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        # self.critic_optimizer.step()

        action, D_KL = self.sample_action_and_KL(state, goal)

        Q = self.critic(state, action, goal)
        Q = torch.min(Q, -1, keepdim=True)[0]

        actor_loss = (self.alpha*D_KL - Q).mean()


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        # self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        return [critic_loss.mean().detach().cpu(), actor_loss.mean().detach().cpu()]
