import os
import sys
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tiles3 import IHT, tiles
from tqdm import tqdm
from utils import plot_rewards

warnings.filterwarnings("ignore")


class QEstimator:
    def __init__(
        self, step_size, num_of_tilings=8, tiles_per_dim=8, max_size=2048, epsilon=0.0
    ):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.tiles_per_dim = tiles_per_dim
        self.epsilon = epsilon
        self.step_size = step_size / num_of_tilings

        self.table = IHT(max_size)

        self.w = np.zeros(max_size)

        self.pos_scale = self.tiles_per_dim / (
            env.observation_space.high[0] - env.observation_space.low[0]
        )
        self.vel_scale = self.tiles_per_dim / (
            env.observation_space.high[1] - env.observation_space.low[1]
        )

    def get_active_features(self, state, action):
        pos, vel = state
        active_features = tiles(
            self.table,
            self.num_of_tilings,
            [
                self.pos_scale * (pos - env.observation_space.low[0]),
                self.vel_scale * (vel - env.observation_space.low[1]),
            ],
            [action],
        )
        return active_features

    def q_predict(self, state, action):
        pos, vel = state
        if pos == env.observation_space.high[0]:  # ゴールに到達したら
            return 0.0
        else:
            active_features = self.get_active_features(state, action)
            return np.sum(self.w[active_features])

    def q_update(self, state, action, target):
        """与えられた状態、行動とターゲットを学習する"""
        active_features = self.get_active_features(state, action)
        q_s_a = np.sum(self.w[active_features])
        delta = target - q_s_a
        self.w[active_features] += self.step_size * delta

    def get_eps_greedy_action(self, state):
        pos, vel = state
        if np.random.rand() < self.epsilon:
            return np.random.choice(env.action_space.n)
        else:
            qvals = np.array(
                [self.q_predict(state, action) for action in range(env.action_space.n)]
            )
            return np.argmax(qvals)


def sarsa_n(qhat, step_size=0.5, epsilon=0.0, n=1, gamma=1.0, episode_cnt=10000):
    episode_rewards = []
    for _ in tqdm(range(episode_cnt)):
        state = env.reset()[0]
        action = qhat.get_eps_greedy_action(state)
        T = float("inf")
        t = 0
        states = [state]
        actions = [action]
        rewards = [0.0]
        while True:
            if t < T:
                next_state, reward, done, _, _ = env.step(action)
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1
                else:
                    next_action = qhat.get_eps_greedy_action(next_state)
                    actions.append(next_action)
            tau = t - n + 1

            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += gamma ** (i - tau - 1) * rewards[i]
                if tau + n < T:
                    G += gamma**n * qhat.q_predict(states[tau + n], actions[tau + n])
                qhat.q_update(states[tau], actions[tau], G)

            if tau == T - 1:
                episode_rewards.append(np.sum(rewards))
                break
            else:
                t += 1
                state = next_state
                action = next_action

    return np.array(episode_rewards)


if __name__ == "__main__":
    # この環境でのみ動作するように設計されています。他の環境では変更が必要です。
    env = gym.make("MountainCar-v0", render_mode="rgb_array")

    # 200~4000の範囲で設定可能
    env._max_episode_steps = 4000

    np.random.seed(13)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # n-SARSA学習のエージェントを作成する
    step_size = 0.8
    episode_cnt = 1000
    n = 4
    epsilon = 0.0
    gamma = 1.0

    estimator = QEstimator(step_size, epsilon=epsilon)
    rewards = sarsa_n(
        estimator,
        step_size=step_size,
        epsilon=epsilon,
        n=n,
        gamma=gamma,
        episode_cnt=episode_cnt,
    )

    # 報酬の可視化
    plot_rewards("Mountain Car World", rewards, "Semi Grad n-step SARSA")

    env.close()
