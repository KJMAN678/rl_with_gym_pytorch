import base64
import glob
import io
import itertools
import os
import sys
import time
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import HTML
from torch.optim import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import display_animation, make_env, torch_fix_seed  # type:ignore

ENV_NAME = "Pendulum-v1"


class MLPActor(nn.Module):
    def __init__(self, state_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.factor = nn.Linear(256, act_dim)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.actor(x)
        x = torch.tanh(x)
        x = self.act_limit * x
        return x


class MLPQFunction(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.Q = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = self.rc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q = self.Q(x)
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.state_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # build Q and policy functions
        self.q1 = MLPQFunction(self.state_dim, self.act_dim)
        self.q2 = MLPQFunction(self.state_dim, self.act_dim)
        self.policy = MLPActor(self.state_dim, self.act_dim, self.act_limit)

    def act(self, state):
        with torch.no_grad():
            return self.policy(state).numpy()

    def get_action(self, s, noise_scale):
        a = self.act(torch.as_tensor(s, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)


def main():
    torch_fix_seed(42)

    env = make_env(ENV_NAME)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    state_shape, action_shape = env.observation_space.shape, env.action_space.shape
    print(f"State shape: {state_shape}")
    print(f"Action shape: {action_shape}")
    env.close()


if __name__ == "__main__":
    main()
