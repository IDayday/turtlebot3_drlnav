import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING

from ..common.ounoise import OUNoise
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES

from .off_policy_agent import OffPolicyAgent, Network

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---
        self.fa1 = nn.Linear(state_size-5, hidden_size)
        self.fa2 = nn.Linear(hidden_size, 32)
        self.fa3 = nn.Linear(32+5, int(hidden_size/2))
        self.fa4 = nn.Linear(int(hidden_size/2), action_size)

        self.ln = nn.LayerNorm(32)
        # --- define layers until here ---

        self.apply(super().init_weights)

    # TODO: 输入预处理
    # TODO: x速度后处理
    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        scan = states[:,:-5]
        other = states[:,-5:]
        x1 = torch.relu(self.fa1(scan))
        x2 = self.ln(self.fa2(x1))
        concat = torch.cat([x2, other],dim=-1)
        x3 = torch.relu(self.fa3(concat))
        action = torch.tanh(self.fa4(x3))

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, action, [x1, x2], [self.fa1.bias, self.fa2.bias])
        # -- define layers to visualize until here ---
        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)

        # --- define layers here ---
        self.l1 = nn.Linear(state_size-5, int(hidden_size / 2))
        self.l2 = nn.Linear(int(hidden_size / 2), 32)
        self.l3 = nn.Linear(32+5+action_size, int(hidden_size / 2))
        self.l4 = nn.Linear(int(hidden_size / 2), 1)

        self.ln = nn.LayerNorm(32)
        # --- define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, actions):
        # --- define forward pass here ---
        scan = states[:,:-5]
        other = states[:,-5:]
        x1 = torch.relu(self.l1(scan))
        x2 = self.ln(self.l2(x1))
        concat = torch.cat([x2, other, actions], dim=-1)
        x3 = torch.relu(self.l3(concat))
        value = self.l4(x3)
        return value


class DDPG(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        self.noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        d = state.shape[-1]
        states = state.reshape(-1,d)
        action = self.actor(states, visualize).squeeze()
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        return action.detach().cpu().data.numpy().tolist()

    # TODO:随机动作
    def get_action_random(self):
        random_x = np.random.uniform(-1.0, 1.0)
        random_y = np.array([0.0])
        random_yaw = np.random.uniform(-1.0, 1.0)
        random_action = [random_x, random_y, random_yaw]

        # action_list = [[0.5, 0.0, 0.0], [0.4, 0.2, 0.0], [0.2, 0.4, 0.0], [0.0, 0.5, 0.0], 
        #                [0.4, -0.2, 0.0], [0.2, -0.4, 0.0], [0.0, -0.5, 0.0], 
        #                [0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]
        # action_list = [[0.5, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0], 
        #                [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2],
        #                [0.0, 0.0, -0.1], [0.0, 0.0, -0.2]]
        # num = len(action_list)
        # random_action = action_list[np.random.randint(num)]

        # random_xy = [np.random.uniform(-0.2,1.0)]*2
        # if random_xy[0] > 0:
        #     random_yaw = np.abs(random_yaw)
        # else:
        #     random_yaw = -np.abs(random_yaw)
        # random_action = random_xy + [random_yaw]
        # print(random_action)
        return random_action

    def train(self, state, action, reward, state_next, done):
        # optimize critic
        action_next = self.actor_target(state_next)
        Q_next = self.critic_target(state_next, action_next)
        Q_target = reward + (1 - done) * self.discount_factor * Q_next
        Q = self.critic(state, action)

        loss_critic = self.loss_function(Q, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        pred_a_sample = self.actor(state)
        loss_actor = -1 * (self.critic(state, pred_a_sample)).mean()
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        return [loss_critic.mean().detach().cpu(), loss_actor.mean().detach().cpu()]
