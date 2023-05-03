import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..gridworld import GridworldEnv


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
        ax.set(title=f"State values after K = {k}")
    else:
        ax.set(title=f"State Values")
    plt.show()


def policy_evaluation(policy, env, discount_factor=1.0, theta=0.00001):
    """
    環境を与えられた方策を評価し 環境のダイナミクスを完全に説明する。

    Args:
        policy:[S, A] 方針を表す定形行列. 我々の場合、ランダム
        env: OpenAI env. env.P -> transition dynamics of the environment.
            env.P[s][a] [(prob, next_state, reward, done)].
            env.nSは環境中の状態の数です。
            env.nAは環境中のアクションの数である。
        theta: すべての状態において、価値関数の変化がθより小さくなった時点で評価を停止する。
        discount_factor: ガンマ割引係数.

    Returns:
        価値関数を表す長さenv.nSのベクトル.
    """

    # 価値関数を全てゼロで開始する
    V = np.zeros(env.nS)
    V_new = np.copy(V)

    while True:
        delta = 0
        # 各 state ごとにバックアップをとる
        for s in range(env.nS):
            v = 0
            # 次の可能な行動を見る
            for a, pi_a in enumerate(policy[s]):
                # 各行動ごとに、とりうる次の状態を見る
                for prob, next_state, reward, done in env.P[s][a]:
                    # バックアップしたダイアグラムごとに期待値を計算する
                    v += pi_a * prob * (reward + discount_factor * V[next_state])
            # 価値関数は、各状態を経ていくらに変化したか
            V_new[s] = v
            delta = max(delta, np.abs(V_new[s] - V[s]))
        V = np.copy(V_new)
        # 閾値を下回った場合は停止する
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(policy, V, env, discount_factor=1.0):
    """
    環境と、環境のダイナミクスと状態値Vの完全な記述を与えられたポリシーを改善する。

    Args:
        ポリシー： [S, A] 方針を表す定形行列.
        V: 与えられた方策に対する現在の状態値
        env: OpenAI env. env.P -> 環境の遷移ダイナミクス。
            env.P[s][a] [(prob, next_state, reward, done)]
            env.nSは環境中の状態の数です。
            env.nAは環境中のアクションの数である。
        discount_factor: ガンマ割引係数.

    戻り値：
        政策： [S, A] 改善された方策を表す整形行列.
        policy_changed: 方策に変更があった場合に `True` の値を持つ boolean値。
    """

    def argmax_a(arr):
        """
        各配列の最大値のインデックスを返す
        """
        max_idx = []
        max_val = float("-inf")
        for idx, elem in enumerate(arr):
            if elem == max_val:
                max_idx.append(idx)
            elif elem > max_val:
                max_idx = [idx]
                max_val = elem
        return max_idx

    policy_changed = False
    Q = np.zeros([env.nS, env.nA])
    new_policy = np.zeros([env.nS, env.nA])

    # 各状態ごとに、貪欲法による改善を行う
    for s in range(env.nS):
        old_action = np.array(policy[s])
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                # 各バックアップしたダイアグラムごとに期待値を計算する
                Q[s, a] += prob * (reward + discount_factor * V[next_state])

        # 最も価値の高い行動を獲得し、状態ごとに新しい方策を設定する
        best_actions = argmax_a(Q[s])
        new_policy[s, best_actions] = 1.0 / len(best_actions)

    if not np.allclose(new_policy[s], policy[s]):
        policy_changed = True

    return new_policy, policy_changed


def policy_iteration(env, discount_factor=1.0, theta=0.00001):
    # ランダムな方策で初期化
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        V = policy_evaluation(policy, env, discount_factor, theta)
        policy, changed = policy_improvement(policy, V, env, discount_factor)
        if not changed:  # 一段落ついたら反復終了
            V_optimal = policy_evaluation(policy, env, discount_factor, theta)
            print("Optimal Policy\n", policy)
            return np.array(V_optimal)


if __name__ == "__main__":
    sns.set()
    env = GridworldEnv()

    # グリッド上で方策ごとのイテレーションを実行
    V_star = policy_iteration(env)

    # 最適な方策、状態、値を出力
    grid_print(V_star.reshape(env.shape))
