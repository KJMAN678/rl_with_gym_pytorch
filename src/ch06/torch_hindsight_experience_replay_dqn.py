import base64
import glob
import io
import math
import os
import random
import shutil
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from scipy.signal import convolve, gaussian
from tqdm import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
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
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(done_flags),
        )


def td_loss_dqn(
    agent,
    target_network,
    states,
    actions,
    rewards,
    next_states,
    done_flags,
    gamma=0.99,
    device=DEVICE,
):
    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float)
    actions = torch.tensor(np.array(actions), device=DEVICE, dtype=torch.long)
    rewards = torch.tensor(np.array(rewards), device=DEVICE, dtype=torch.float)
    next_states = torch.tensor(np.array(next_states), device=DEVICE, dtype=torch.float)
    done_flags = torch.tensor(
        np.array(done_flags.astype("float32")), device=DEVICE, dtype=torch.float
    )

    # get q-values for all actions in current states use agent network
    q_s = agent(states)

    # select q-values for chosen actions
    q_s_a = q_s[range(len(actions)), actions]

    with torch.no_grad():
        # compute q-values for all actions in next states use target network
        q_s1 = target_network(next_states)

        # compute Qmax(next_states, actions) using predicted next q-values
        q_s1_a1max, _ = torch.max(q_s1, dim=1)

        # compute "target q-values"
        target_q = rewards + gamma * q_s1_a1max * (1 - done_flags)

    # mean squared error loss to minimize
    loss = torch.mean((q_s_a - target_q.detach()) ** 2)

    return loss


class BitFlipEnvironment:
    def __init__(self, bits):
        self.bits = bits
        self.state = np.zeros((self.bits,))
        self.goal = np.zeros((self.bits,))
        self.reset()

    def reset(self):
        self.state = np.random.randint(2, size=self.bits).astype(np.float32)
        self.goal = np.random.randint(2, size=self.bits).astype(np.float32)
        if np.allclose(self.state, self.goal):
            self.reset()
        return self.state.copy(), self.goal.copy()

    def step(self, action):
        self.state[action] = 1 - self.state[action]
        reward, done = self.compute_reward(self.state, self.goal)
        return self.state.copy(), reward, done

    def render(self):
        print(f"State: {self.state.tolist()}")
        print(f"Goal: {self.goal.tolist()}\n")

    @staticmethod
    def compute_reward(state, goal):
        done = np.allclose(state, goal)
        return 0.0 if done else -1.0, done


# a simplified version of DuelingDQN with lesser number of layers
class DuelingMLP(nn.Module):
    def __init__(self, state_size, n_actions, epsilon=1.0):
        super().__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.linear = nn.Linear(state_size, 256)
        self.v = nn.Linear(256, 1)
        self.adv = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.linear(x))
        v = self.v(x)
        adv = self.adv(x)
        qvalues = v + adv - adv.mean(dim=1, keepdim=True)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float)
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


def train_her(
    env,
    agent,
    target_network,
    optimizer,
    num_epochs,
    exploration_fraction,
    eps_max,
    eps_min,
    num_cycles,
    num_episodes,
    num_bits,
    future_k,
    num_opt_steps,
    batch_size,
    td_loss_fn,
):
    success_rate = 0.0
    success_rates = []

    exp_replay = ReplayBuffer(10**6)

    for epoch in range(num_epochs):
        # Decay epsilon linearly from eps_max to eps_min
        eps = max(
            eps_max
            - epoch * (eps_max - eps_min) / int(num_epochs * exploration_fraction),
            eps_min,
        )
        print(
            f"Epoch: {epoch + 1}, exploration: {100 * eps: .0f}%, success rate: {success_rate: .2f}"
        )
        agent.epsilon = eps
        target_network.epsilon = eps

        successes = 0
        for cycle in range(num_cycles):
            for episode in range(num_episodes):
                # Run episode and cache trajectory
                episode_trajectory = []
                state, goal = env.reset()

                for step in range(num_bits):
                    state_ = np.concatenate([state, goal])
                    qvalues = agent.get_qvalues([state_])
                    action = agent.sample_actions(qvalues)[0]
                    next_state, reward, done = env.step(action)

                    episode_trajectory.append((state, action, reward, next_state, done))
                    state = next_state
                    if done:
                        successes += 1
                        break

                # Fill up replay memory
                steps_taken = step
                for t in range(steps_taken):
                    # Usual experience replay
                    state, action, reward, next_state, done = episode_trajectory[t]
                    state_, next_state_ = np.concatenate((state, goal)), np.concatenate(
                        (next_state, goal)
                    )
                    exp_replay.add(state_, action, reward, next_state_, done)

                    # Hindsight experience replay
                    for _ in range(future_k):
                        future = random.randint(t, steps_taken)
                        new_goal = episode_trajectory[future][3]
                        new_reward, new_done = env.compute_reward(next_state, new_goal)
                        state_, next_state_ = np.concatenate(
                            (state, new_goal)
                        ), np.concatenate((next_state, new_goal))
                        exp_replay.add(
                            state_, action, new_reward, next_state_, new_done
                        )

            # Optimize DQN
            for opt_step in range(num_opt_steps):
                # train by sampling batch_size of data from experience replay
                states, actions, rewards, next_states, done_flags = exp_replay.sample(
                    batch_size
                )
                # loss = <compute TD loss>
                optimizer.zero_grad()
                loss = td_loss_fn(
                    agent,
                    target_network,
                    states,
                    actions,
                    rewards,
                    next_states,
                    done_flags,
                    gamma=0.99,
                    device=DEVICE,
                )
                loss.backward()
                optimizer.step()

            target_network.load_state_dict(agent.state_dict())

        success_rate = successes / (num_episodes * num_cycles)
        success_rates.append(success_rate)

    # print graph
    plt.plot(success_rates, label="HER-DQN")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.title(f"Number of bits: {num_bits}")
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def main():
    num_bits = 50
    n_actions = num_bits
    state_size = 2 * num_bits

    future_k = 4
    num_epochs = 40
    num_cycles = 50
    num_episodes = 16
    num_opt_steps = 40
    eps_max = 0.2
    eps_min = 0.0
    exploration_fraction = 0.5
    batch_size = 128

    env = BitFlipEnvironment(num_bits)

    agent = DuelingMLP(state_size, n_actions, epsilon=1).to(DEVICE)
    target_network = DuelingMLP(state_size, n_actions, epsilon=1).to(DEVICE)
    target_network.load_state_dict(agent.state_dict())
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    train_her(
        env,
        agent,
        target_network,
        optimizer,
        num_epochs,
        exploration_fraction,
        eps_max,
        eps_min,
        num_cycles,
        num_episodes,
        num_bits,
        future_k,
        num_opt_steps,
        batch_size,
        td_loss_fn=td_loss_dqn,
    )


if __name__ == "__main__":
    main()
