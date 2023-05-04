import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(env_name, rewards, label):
    """
    報酬を可視化する
    """
    plt.title(f"env={env_name}, Mean reward = {np.mean(rewards[-20:])}")
    plt.plot(rewards, label=label)
    plt.grid()
    plt.legend()
    plt.ylim(-300, 0)
    plt.show()


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
    plt.ylim(-300, 0)
    plt.show()


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
