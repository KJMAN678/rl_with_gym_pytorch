import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

from gridworld import GridworldEnv

env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    # すべて 0 の value function で始める
    V = np.zeros(env.nS)
    V_new = np.copy(V)

    while True:
        delta = 0
        # 各 state ごとにバックアップを行う
        for s in range(env.nS):
            v = 0
            # 可能な次の actions を見る
            for a, pi_a in enumerate(policy[s]):
                # 各 action ごとに、次の state を見る
                for prob, next_state, reward, done in env.P[s][a]:
                    # バックアップ図ごとに期待値を計算する
                    v += pi_a * prob * (reward + discount_factor * V[next_state])

            # value function はいくらにかわったか
            V_new[s] = v
            delta = max(delta, np.abs(V_new[s] - V[s]))
        V = np.copy(V_new)
        # 変化が閾値以下になったら停止
        if delta < theta:
            break
    return np.array(V)


def grid_print(V, k=None):
    ax = sns.heatmap(
        V.reshape(env.shape),
        annot=True,
        square=True,
        cbar=False,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
    )
    if k:
        ax.set(title=f"K = {k}")
    plt.show()


random_policy = np.ones([env.nS, env.nA]) / env.nA

V_pi = policy_eval(random_policy, env, discount_factor=1.0, theta=0.00001)

grid_print(V_pi.reshape(env.shape))
