import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStack, TransformReward

# from gymnasium.utils.play import play
from scipy.signal import convolve, gaussian
from utils import make_env, torch_fix_seed


def make_env_atari(env_name, clip_rewards=True, seed=None):
    env = gym.make(env_name, render_mode="rgb_array")
    env.metadata["render_fps"] = 30
    if seed is not None:
        env.seed(seed)
    env = AtariPreprocessing(env, screen_size=84, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    if clip_rewards:
        env = TransformReward(env, lambda r: np.sign(r))
    return env


def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride + 1


if __name__ == "__main__":
    seed = 42
    torch_fix_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    env_name = "BreakoutNoFrameskip-v4"

    env = make_env(env_name)
    env.reset(seed=seed)

    n_cols = 4
    n_rows = 2
    fig = plt.figure(figsize=(16, 9))

    for row in range(n_rows):
        for col in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            ax.set_title(f"row:{row} col:{col}")
            ax.imshow(env.render())
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            env.step(env.action_space.sample())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # play(env=gym.make(env_name), render_mode="rgb_array", zoom=2, fps=10)

    env = make_env_atari(env_name)
    env.reset()
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    print(n_actions)
    print(env.get_action_meanings())

    for _ in range(12):
        obs, _, _, _, _ = env.step(env.action_space.sample())

    plt.figure(figsize=(12, 10))
    plt.title("Game Image")
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    obs = obs[:]
    obs = np.transpose(obs, [1, 0, 2])
    obs = obs.reshape((obs.shape[0], -1))

    plt.figure(figsize=(15, 15))
    plt.title("Agent observation(4 frmaes left to right)")
    plt.imshow(obs, cmap="gray")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    conv1 = conv2d_size_out(84, 8, 4)
    print("Conv1", conv1)
    conv2 = conv2d_size_out(conv1, 4, 2)
    print("Conv2", conv2)

    print("Input to Dense layer:", conv2 * conv2 * 32)
