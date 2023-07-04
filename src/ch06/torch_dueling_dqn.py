import base64
import glob
import io
import math
import os
import random
import sys
import time
from collections import namedtuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import HTML, clear_output
from scipy.signal import convolve, gaussian
from tqdm import trange

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import make_env, torch_fix_seed

ENV_NAME = "CartPole-v1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        state_dim = state_shape[0]
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.q = nn.Linear(32, n_actions)

    def forward(self, state_t):
        # pass the state at time t through the network to get Q(s,a)
        x = F.relu(self.fc1(state_t))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        qvalues = self.q(x)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and output is Q values as numpy array
        states = torch.tensor(np.array(states), device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for _ in range(n_games):
        s = env.reset()[0]
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            if greedy:
                action = qvalues.argmax(axis=-1)[0]
            else:
                action = agent.sample_actions(qvalues)[0]
            s, r, done, _, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


class ReplayBuffer:
    def __init__(self, size):
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to hold buffer
        self.next_id = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)

    def play_and_record(start_state, agent, env, exp_replay, n_steps=1):
        s = start_state
        sum_rewards = 0

        # Play the game for n_steps and record transitions in buffer
        for _ in range(n_steps):
            qvalues = agent.get_qvalues([s])
            a = agent.sample_actions(qvalues)[0]
            next_s, r, done, _, _ = env.step(a)
            sum_rewards += r
            exp_replay.add(s, a, r, next_s, done)
            if done:
                s = env.reset()
            else:
                s = next_s

        return sum_rewards, s
    
def td_loss_dqn(agent, target_network, states, actions, rewards, next_states, done_flags, gamma=0.99, device):
    
    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), device=device, dtype=torch.float)
    actions = torch.tensor(np.array(actions), device=device, dtype=torch.long)
    rewards = torch.tensor(np.array(rewards), device=device, dtype=torch.float)
    next_states = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
    done_flags = torch.tensor(np.array(done_flags.astype('float32')), device=device, dtype=torch.float)
    
    # get q-values for all actions in current states
    # use agent network    
    q_s = agent(states)
    
    # select q-values for chosen actions
    q_s_a = q_s[range(len(actions)), actions]
    
    with torch.no_grad():
        # compute q-values for all actions in next states
        # use target network
        q_s1 = target_network(next_states)
        
        # compute Qmax(next_states, actions) using predicted next q-values
        q_s1_a1max, _ = torch.max(q_s1, dim=1)
        
        # compute "target q-values" 
        target_q = rewards + gamma * q_s1_a1max * (1 - done_flags)
        
    # mean squared error loss to minimize
    loss = torch.mean((q_s_a - target_q.detach()) ** 2)
    
    return loss



def show_env():
    env = make_env(ENV_NAME)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(device)
    print(evaluate(env, agent, n_games=1))
    env.close()
    target_network = DQNAgent(agent.state_shape, agent.n_actions, epsilon=0.5).to(device)
    target_network.load_state_dict(agent.start_dict())


def main():
    pass


if __name__ == "__main__":
    torch_fix_seed(42)
    show_env()
