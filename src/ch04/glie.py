import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from gridworld import GridworldEnv


def mc_policy_eval(policy, env, discount_factor=1.0, episode_count=100):
    """
        環境を与えられたポリシーを評価する。

        Args:
            policy:[S, A] ポリシーを表す行列の形をしている。この場合、ランダム
            env: OpenAIのenv.Pにアクセスできる。モデルフリーでは、env.P.にアクセスできない、
    step(a)でアクションを起こし、(s', r, done, info)のタプルを受け取る。
                 env.nSは、環境における状態の数です。
                 env.nAは、環境中のアクションの数である。
            episode_count: エピソードの数：
            discount_factor: ガンマ割引係数。

        Returns:
            価値関数を表す長さenv.nSのベクトル.
    """

    # 全てゼロの状態価値の配列と、訪問回数ゼロでスタートå
    V = np.zeros(env.nS)
    N = np.zeros(env.nS)
    i = 0

    # 複数エピソードを実行
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

        # 状態価値を更新する
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


def GLIE(env, discount_factor=1.0, episode_count=10):
    """
    環境が与えられたときの最適な方針を見つける。

    Args:
        env: OpenAIの環境。モデルフリーでは、環境の遷移ダイナミクスであるenv.Pにアクセスできない。 step(a)を使って行動を起こし、(s', r, done, info)のタプルを受け取る。
             env.nSは、環境における状態の数である。
             env.nAは、環境中のアクションの数である。
        episode_count: エピソードの数：
        discount_factor: ガンマ割引係数。

    戻り値：
        価値関数を表す長さ env.nS のベクトル.
        policy:[S, A] 方針を表す整形行列. この場合，ランダム
    """
    # 全てゼロの状態価値の配列と状態-行動マトリックスで開始する
    # 状態-価値の訪問回数ごとに訪問回数をゼロで初期化する
    V = np.zeros(env.nS)
    N = np.zeros((env.nS, env.nA))
    Q = np.zeros((env.nS, env.nA))
    # ランダム方策
    policy = [np.random.randint(env.nA) for _ in range(env.nS)]
    k = 1
    eps = 1

    def argmax_a(arr):
        """
        配列の最大要素のインデックスを返す。
        結びつきを一律に破る。
        """
        max_idx = []
        max_val = float("-inf")
        for idx, elem in enumerate(arr):
            if elem == max_val:
                max_idx.append(idx)
            elif elem > max_val:
                max_idx = [idx]
                max_val = elem
        return np.random.choice(max_idx)

    def get_action(state):
        if np.random.random() < eps:
            return np.random.choice(env.nA)
        else:
            return argmax_a(Q[state])

    # 複数エピソードを実行
    pbar = tqdm(total=episode_count)
    while k <= episode_count:
        # 1エピソードごとにサンプルを収集する
        episode_states = []
        episode_actions = []
        episode_returns = []
        state = env.reset()[0]
        while True:
            action = get_action(state)
            episode_actions.append(action)
            (state, reward, done, _, _) = env.step(action)
            episode_returns.append(reward)
            if not done:
                episode_states.append(state)
            else:
                break

        # 状態-行動価値を更新
        G = 0
        count = len(episode_states)
        for t in range(count - 1, -1, -1):
            s, a, r = episode_states[t], episode_actions[t], episode_returns[t]
            G = discount_factor * G + r
            N[s, a] += 1
            Q[s, a] = Q[s, a] + 1 / N[s, a] * (G - Q[s, a])

        # 方策と最適な価値を更新する
        k = k + 1
        eps = 1 / k
        # if "をコメントアウトすることで、初期に高い探査を行い、5000エピソード後にイプシロンが減衰するようにする。
        if (k >= 20) & (k <= 1000):
            eps = 0.05

        for s in range(env.nS):
            best_action = argmax_a(Q[s])
            policy[s] = best_action
            V[s] = Q[s, best_action]

        pbar.update(1)

    pbar.close()

    return np.array(V), np.array(policy)


def grid_print(
    V,
    k=None,
    shade_cell=True,
    annot=True,
    square=True,
    cbar=False,
    cmap="Blues",
    xticklabels=False,
    yticklabels=False,
):
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
    sns.set()
    env = GridworldEnv()

    # MC 方策管理 GLIE を実行
    V_pi, policy = GLIE(env, discount_factor=1.0, episode_count=2000)

    action_labels = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    # 方策を出力
    optimal_actions = [action_labels[policy[s]] for s in range(env.nS)]
    optimal_actions[0] = "*"
    optimal_actions[-1] = "*"

    print("policy\n\n", np.array(optimal_actions).reshape(env.shape))

    # 状態価値を出力
    grid_print(V_pi.reshape(env.shape))
