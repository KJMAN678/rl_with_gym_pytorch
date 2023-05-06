import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium.utils.play import play
from scipy.signal import convolve, gaussian
from utils import make_env, torch_fix_seed


def CLD_DUMPED():
    pass


if __name__ == "__main__":
    seed = 42
    torch_fix_seed(seed)

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
    plt.pause(4)
    plt.close()

    # play(env=gym.make(env_name), zoom=2, fps=30)
