import glob
import os
import shutil
import sys
from typing import Any, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import torch
import torch.nn as nn
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers import RecordVideo
from nptyping import NDArray
from torch.nn.modules.container import Sequential

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import display_animation, make_env, torch_fix_seed  # type:ignore

ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_probs(states: NDArray, model: Sequential):
    """
    params: states: [batch, state_dim]
    returns: probs: [batch, n_actions]
    """
    states_tensor: torch.Tensor = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        logits = model(states_tensor)
    probs = nn.functional.softmax(logits, -1).detach().numpy()
    return probs


def generate_trajectory(
    env: Any[CartPoleEnv, RecordVideo], model: Sequential, n_actions: np.int64, n_steps: int = 1000
) -> Tuple[list, list, list]:
    """
    Play a session and genrate a trajectory
    returns: arrays of states, actions, rewards
    """
    states, actions, rewards = [], [], []

    # initialize the environment
    s = env.reset()[0]

    # generate n_steps of trajectory:
    for t in range(n_steps):
        action_probs = predict_probs(np.array([s]), model)[0]
        # sample action based on action_probs
        a = np.random.choice(n_actions, p=action_probs)
        next_state, r, done, _, _ = env.step(a)

        # update arrays
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = next_state
        if done:
            break

    return states, actions, rewards


def get_rewards_to_go(rewards: list, gamma: float = 0.99) -> list:
    T = len(rewards)
    # empty array to return the rewards to go
    rewards_to_go = [0] * T
    rewards_to_go[T - 1] = rewards[T - 1]

    for i in range(T - 2, -1, -1):
        rewards_to_go[i] = gamma * rewards_to_go[i + 1] + rewards[i]

    return rewards_to_go


def train_one_episode(
    states: list,
    actions: list,
    rewards: list,
    model: Sequential,
    optimizer,
    gamma: float = 0.99,
    entropy_coef: float = 1e-2,
) -> NDArray:
    # get rewards to go
    rewards_to_go = get_rewards_to_go(rewards, gamma)

    # onvert numpy array to torch tensors
    states_tensor: torch.Tensor = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float)
    actions_tensor: torch.Tensor = torch.tensor(np.array(actions), device=DEVICE, dtype=torch.long)
    rewards_to_go_tensor: torch.Tensor = torch.tensor(np.array(rewards_to_go), device=DEVICE, dtype=torch.float)

    # get action probabilities from states
    logits = model(states_tensor)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    log_probs_for_actions = log_probs[range(len(actions_tensor)), actions_tensor]

    # Compute loss to be minimized
    J = torch.mean(log_probs_for_actions * rewards_to_go_tensor)
    H = -(probs * log_probs).sum(-1).mean()

    loss = -(J + entropy_coef * H)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return np.sum(rewards)


def generate_animation(env: CartPoleEnv, model: Sequential, n_actions: np.int64, save_dir: str):
    try:
        # env = RecordVideo(env, save_dir, episode_trigger=lambda id: True)
        env_rec: RecordVideo = RecordVideo(env, save_dir, episode_trigger=lambda id: True)
    except gym.error.Error as e:
        print(e)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generate_trajectory(env_rec, model, n_actions)


def main():
    torch_fix_seed(42)

    env = make_env(ENV_NAME)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    print(f"n_actions: {type(n_actions)}")
    state_dim = state_shape[0]

    model = nn.Sequential(
        nn.Linear(state_dim, 192),
        nn.ReLU(),
        nn.Linear(192, n_actions),
    )
    model = model.to(DEVICE)

    total_rewards = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(10000):
        states, actions, rewards = generate_trajectory(env, model, n_actions)
        reward = train_one_episode(states, actions, rewards, model, optimizer)
        total_rewards.append(reward)
        if i != 0 and i % 100 == 0:
            mean_reward = np.mean(total_rewards[-100:-1])
            print(f"mean reward:{mean_reward:.3f}")
            if mean_reward > 300:
                break
    env.close()

    if os.path.exists("./videos/"):
        shutil.rmtree("./videos/")

    save_dir = "./videos/pytorch/reinforce/"
    env = make_env(ENV_NAME)
    generate_animation(env, model, n_actions, save_dir=save_dir)
    [filepath] = glob.glob(os.path.join(save_dir, "*.mp4"))
    display_animation(filepath)
    env.close()


if __name__ == "__main__":
    main()
