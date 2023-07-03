import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gridworld import GridworldEnv


def mc_policy_eval(policy, env, discount_factor=1.0, episode_count=100):
    """
    環境を与えられたポリシーを評価する。

    Args:
        ポリシー： 方策を表している [S, A] 形状のマトリックス. この例ではランダム
        env: OpenAIのenv。モデルフリーでは、環境の遷移ダイナミクスであるenv.Pにアクセスすることはできない。
             step(a)を使って行動を起こし、(s', r, done, info)のタプルを受け取る。
             env.nSは、環境中の状態の数です。
             env.nAは、環境中のアクションの数である。
        episode_count: エピソードの数：
        discount_factor: ガンマ割引係数。

    戻り値：
        価値関数を表す長さenv.nSのベクトル.
    """
    # 全てゼロの 状態価値の配列と、ゼロの訪問回数でスタートする
    V = np.zeros(env.nS)
    N = np.zeros(env.nS)
    i = 0

    # 複数エピソードで実行する
    while i < episode_count:
        # 1エピソードごとにサンプルを集める
        episode_states = []
        episode_returns = []
        state = env.reset()[0]
        episode_states.append(state)

        while True:
            action = np.random.choice(env.nA, p=policy[state])
            (state, reward, done, _, _) = env.step(action)
            episode_returns.append(reward)
            if not done:
                episode_states.append(state)
            else:
                break
        # 状態価値をアップデートする
        G = 0
        count = len(episode_states)
        for t in range(count - 1, -1, -1):
            s, r = episode_states[t], episode_returns[t]
            G = discount_factor * G + r
            if s not in episode_states[:t]:
                N[s] += 1
                V[s] = V[s] + 1 / N[s] * (G - V[s])

        i = i + 1

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
    plt.show(block=False)
    plt.pause(2)
    plt.close()


if __name__ == "__main__":
    env = GridworldEnv()

    # ランダムな方策を作成する
    random_policy = np.ones([env.nS, env.nA]) / env.nA

    # ランダム方策ごと、100 エピソードごとに MC方策予測を実行
    V_pi = mc_policy_eval(random_policy, env, discount_factor=1.0, episode_count=100)
    # 方策を出力する
    grid_print(V_pi.reshape(env.shape), 100)

    # ランダム方策ごと、10000 エピソードごとに MC方策予測を実行
    V_pi = mc_policy_eval(random_policy, env, discount_factor=1.0, episode_count=10000)
    # 方策を出力する
    grid_print(V_pi.reshape(env.shape), 10000)

    # ランダム方策ごと、100000 エピソードごとに MC方策予測を実行
    V_pi = mc_policy_eval(random_policy, env, discount_factor=1.0, episode_count=100000)
    # 方策を出力する
    grid_print(V_pi.reshape(env.shape), 100000)
