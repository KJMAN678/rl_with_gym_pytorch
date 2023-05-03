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
        ax.set(title=f"State values after K = {k}.")
    else:
        ax.set(title="State Values.")
    plt.show()


def value_iteration(env, discount_factor=1.0, theta=0.00001):
    """
    環境と環境のダイナミクスの完全な記述があれば、価値反復処理を行う。

    Args:
        env: OpenAI env. env.P -> 環境の遷移ダイナミクス。
            env.P[s][a] [(prob, next_state, reward, done)].
            env.nSは環境の状態数です。
            env.nAは環境中のアクションの数である。
        discount_factor: ガンマ割引係数。
        theta: 反復を停止させるための許容範囲レベル

    リターンズ
        ポリシー： [S, A] 最適な方針を表す定形行列.
        value : [S] 最適値を表す長さベクトル
    """

    def argmax_a(arr):
        """
        配列の最大となる要素を返す
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

    optimal_policy = np.zeros([env.nS, env.nA])
    V = np.zeros(env.nS)
    V_new = np.copy(V)

    while True:
        delta = 0
        # 状態ごとに貪欲なバックアップを行う
        for s in range(env.nS):
            q = np.zeros(env.nA)
            # 次に取りうる行動を見る
            for a in range(env.nA):
                # 行動ごとに、次の取りうる状態を見る
                # q[s, a] を計算する
                for prob, next_state, reward, done in env.P[s][a]:
                    # バックアップダイアグラムごとに各行動ごとの価値を計算する
                    if not done:
                        q[a] += prob * (reward + discount_factor * V[next_state])
                    else:
                        q[a] += prob * reward
            # 取りうる行動すべてのうち、最大となる価値を検索する
            # 更新した状態の価値をアップデートする
            V_new[s] = q.max()
            # 各行動を通じて価値関数はどのくらい変わったか
            delta = max(delta, np.abs(V_new[s] - V[s]))

        V = np.copy(V_new)

        # 閾値を下回ったら処理を停止する
        if delta < theta:
            break

    # V(s) は最適値を持つ。これらの価値と最適な方策を計算するためのバックアップステップを使う
    for s in range(env.nS):
        q = np.zeros(env.nA)
        # とりうる次の行動を見る
        for a in range(env.nA):
            # q[s, a] を計算するために、各行動ごとに、とりうる各状態をみる
            for prob, next_state, reward, done in env.P[s][a]:
                # バックアップダイアグラムごとに各行動の価値を計算する
                if not done:
                    q[a] += prob * (reward + discount_factor * V[next_state])
                else:
                    q[a] += prob * reward
        # 最適な行動を探索する
        # 最大値に等しい行動を全て等しい確率で行う確率論的方策を返す。
        best_actions = argmax_a(q)
        optimal_policy[s, best_actions] = 1.0 / len(best_actions)

    return optimal_policy, V


if __name__ == "__main__":
    sns.set()
    env = GridworldEnv()

    # グリッド上で反復方策を実行
    pi_star, V_star = value_iteration(env)

    # 最適な方策を出力
    print("Optimal Policy\n", pi_star)

    # 最適な状態価値を出力する
    grid_print(V_star.reshape(env.shape))
