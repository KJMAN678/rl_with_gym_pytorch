import glob
import os
import shutil
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import display_animation, make_env, torch_fix_seed  # type:ignore

ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        self.state_dim = state_dim
        self.n_actions = n_actions

        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        logits = self.actor(x)
        state_value = self.critic(x)
        return logits, state_value


def sample_action(state, model, n_actions):
    """
    Play a session and genrate a trajectory
    returns: arrays of states, actions, rewards
    """
    state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = model(state)
    action_probs = F.softmax(logits, -1).detach().numpy()[0]
    action = np.random.choice(n_actions, p=action_probs)
    return action


def generate_trajectory(env, model, n_actions, n_steps=1000):
    """
    Play a session and genrate a trajectory
    returns: arrays of states, actions, rewards
    """
    states, actions, rewards = [], [], []
    # initialize the environment
    s = env.reset()[0]

    # generate n_steps of trajectory:
    for t in range(n_steps):
        # sample action based on action_probs
        a = sample_action(np.array([s]), model, n_actions)
        next_state, r, done, _, _ = env.step(a)

        # update arrays
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = next_state
        if done:
            break

    return states, actions, rewards


def get_rewards_to_go(rewards, gamma=0.99):
    T = len(rewards)
    # empty array to return the rewards to go
    rewards_to_go = [0] * T
    rewards_to_go[T - 1] = rewards[T - 1]

    for i in range(T - 2, -1, -1):
        rewards_to_go[i] = gamma * rewards_to_go[i + 1] + rewards[i]

    return rewards_to_go


def train_one_episode(
    states, actions, rewards, model, optimizer, gamma=0.99, entropy_coef=1e-2
):
    # get rewards to go
    rewards_to_go = get_rewards_to_go(rewards, gamma)

    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float)
    actions = torch.tensor(np.array(actions), device=DEVICE, dtype=torch.long)
    rewards_to_go = torch.tensor(
        np.array(rewards_to_go), device=DEVICE, dtype=torch.float
    )

    # get action probabilities from states
    logits, state_values = model(states)
    probs = F.softmax(logits, -1)
    log_probs = F.log_softmax(logits, -1)

    log_probs_for_actions = log_probs[range(len(actions)), actions]

    advantage = rewards_to_go - state_values.squeeze(-1)

    # Compute loss to be minimized
    J = torch.mean(log_probs_for_actions * (advantage))
    H = -(probs * log_probs).sum(-1).mean()

    loss = -(J + entropy_coef * H)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return np.sum(rewards)


def generate_animation(env, n_actions, model, save_dir):
    try:
        env = RecordVideo(env, save_dir, episode_trigger=lambda id: True)
    except gym.error.Error as e:
        print(e)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generate_trajectory(env, model, n_actions)


def main():
    torch_fix_seed(42)

    env = make_env(ENV_NAME)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    state_dim = state_shape[0]
    env.close()

    model = ActorCritic(state_dim, n_actions)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_rewards = []
    for i in range(10000):
        states, actions, rewards = generate_trajectory(env, model, n_actions)
        reward = train_one_episode(states, actions, rewards, model, optimizer)
        total_rewards.append(reward)
        if i != 0 and i % 100 == 0:
            mean_reward = np.mean(total_rewards[-100:-1])
            print(f"mean rewards:{mean_reward:.3f}")
            if mean_reward > 700:
                break
    env.close()

    if os.path.exists("./videos/"):
        shutil.rmtree("./videos/")

    save_dir = "./videos/pytorch/actor_critic/"
    env = make_env(ENV_NAME)
    generate_animation(env, n_actions, model, save_dir=save_dir)
    [filepath] = glob.glob(os.path.join(save_dir, "*.mp4"))
    display_animation(filepath)
    env.close()


if __name__ == "__main__":
    main()
