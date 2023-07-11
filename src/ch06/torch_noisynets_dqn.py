import base64
import glob
import io
import math
import os
import random
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

from utils import display_animation, generate_animation, make_env, torch_fix_seed

ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        states = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float32)
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


def show_env():
    torch_fix_seed(42)

    env = make_env(ENV_NAME)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(DEVICE)
    print(evaluate(env, agent, n_games=1))
    env.close()


def main():
    xxx


if __name__ == "__main__":
    show_env()
