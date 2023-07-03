import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gridworld import GridworldEnv


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    環境を与えられた方策を評価する。

    Args:
        policy:[S, A]形式の方策を表す行列。この場合、ランダム
        env: OpenAI env. env.P -> 環境の遷移ダイナミクス。
            env.P[s][a] [(prob, next_state, reward, done)].
            env.nS は環境中の状態の数。
            env.nA は環境中のアクションの数。
        theta: すべての状態において、価値関数の変化がθより小さくなった時点で、反復を停止する。
        discount_factor: ガンマ割引係数.

    Returns:
        価値関数を表す長さenv.nSのベクトル.
    """

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
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def policy_eval_withprint(policy, env, discount_factor=1.0, theta=0.00001, print_at=[]):
    """
        環境を与えられた政策を評価し 環境のダイナミクスを完全に記述する。

    Args:
        policy: [S, A] 方策を表す定形行列．この場合はランダム
        env: OpenAI env. env.P -> 環境の遷移ダイナミクス
            env.P[s][a] [(prob, next_state, reward, done)].
            env.nSは環境の状態数
            env.nAは、環境におけるアクションの数
        theta: すべての状態において、価値関数の変化量がθより小さくなったら評価を中止する。
        discount_factor: ガンマ割引係数。

    Returns:
        価値関数を表す長さenv.nSのベクトル。
    """

    # 価値関数はすべて 0 から開始する
    k = 0
    V = np.zeros(env.nS)
    V_new = np.copy(V)

    while True:
        k += 1
        delta = 0
        # 各 state がバックアップとして機能する
        for s in range(env.nS):
            v = 0
            # 次のとりうるアクションを見る
            for a, pi_a in enumerate(policy[s]):
                # 各アクションごとに、とりうる状態を見る
                for prob, next_state, reward, done in env.P[s][a]:
                    # バックアップダイアログごとに期待値を計算する
                    v += pi_a * prob * (reward + discount_factor * V[next_state])
            # 価値関数はどのくらい変わったか？
            V_new[s] = v
            delta = max(delta, np.abs(V_new[s] - V[s]))

        V = np.copy(V_new)
        # 各イテレーションのグリッドの値を出力
        if k in print_at:
            grid_print(V.reshape(env.shape), k=k)
        # 閾値以下に変化したら止める
        if delta < theta:
            break
    grid_print(V.reshape(env.shape), k=k)
    return np.array(V)


if __name__ == "__main__":
    sns.set()

    env = GridworldEnv()

    # ランダムな方策を作成
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    # ランダムな方策ごとに方策のイテレーションを実行
    V_pi = policy_eval(random_policy, env, discount_factor=1.0, theta=0.00001)
    # 方策のグリッドを出力
    grid_print(V_pi.reshape(env.shape))

    # ランダムな方策ごとに方策のイテレーションを実行し、暫定的な状態値を出力
    V_pi = policy_eval_withprint(
        random_policy,
        env,
        discount_factor=1.0,
        theta=0.000001,
        print_at=[1, 2, 3, 10, 50, 100],
    )
