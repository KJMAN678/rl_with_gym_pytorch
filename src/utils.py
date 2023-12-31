import base64
import io
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from IPython.display import HTML


def plot_rewards(env_name, rewards, label):
    """
    報酬を可視化する
    """
    plt.title(f"env={env_name}, Mean reward = {np.mean(rewards[-20:])}")
    plt.plot(rewards, label=label)
    plt.grid()
    plt.legend()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def plot_rewards_compare(env_name, rewards, labels):
    """
    複数の報酬を可視化
    """
    for i in range(len(rewards)):
        reward_list = rewards[i]
        label = labels[i] + f".mean={np.mean(reward_list[-20])}"
        plt.title(f"env={env_name}")
        plt.plot(rewards[i], label=label)
    plt.grid()
    plt.legend()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def print_policy(env, agent):
    """CliffWorldの方策を出力するためのヘルパー関数"""
    nR, nC = env._cliff.shape

    actions = "^>v<"

    for y in range(nR):
        for x in range(nC):
            if env._cliff[y, x]:
                print(" C ", end="")
            elif (y * nC + x) == env.start_state_index:
                print(" X ", end="")
            elif (y * nC + x) == nR * nC - 1:
                print(" T ", end="")
            else:
                print(f" {actions[agent.max_action(y * nC + x)]}", end="")
        print()


def torch_fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def make_env(env_name, seed=None):
    env = gym.make(env_name, render_mode="rgb_array").unwrapped
    if seed is not None:
        env.seed(seed)
    return env


def generate_animation(env, agent, save_dir):
    try:
        env = RecordVideo(env, save_dir, episode_trigger=lambda id: True)
    except gym.error.Error as e:
        print(e)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state = env.reset()[0]
    reward = 0
    while True:
        qvalues = agent.get_qvalues([state])
        action = qvalues.argmax(axis=-1)[0]
        state, r, done, _, _ = env.step(action)
        reward += r
        if done or reward >= 1000:
            print(f"Got reward: {reward}")
            break


def display_animation(filepath):
    video = io.open(filepath, "r+b").read()
    encoded = base64.b64encode(video)
    return HTML(
        data=f"""<video alt="test controls>
                <source src=data: video/mp4; base64, {encoded.decode('ascii')} type="video/mp4" />
                </video>"""
    )
