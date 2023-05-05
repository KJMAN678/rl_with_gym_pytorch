import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gymnasium.envs import toy_text
from utils import plot_rewards, print_policy


class SARSAAgent:
    def __init__(self, alpha, epsilon, gamma, get_possible_actions):
        self.get_possible_actions = get_possible_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self._Q = defaultdict(lambda: defaultdict(lambda: 0))

    def get_Q(self, state, action):
        return self._Q[state][action]

    def set_Q(self, state, action, value):
        self._Q[state][action] = value

    def update(self, state, action, reward, next_state, next_action, done):
        """
        (S, A, R, S', A') のサンプルに基づき更新されるSARSAを実行
        """

        if not done:
            td_error = (
                reward
                + self.gamma * self.get_Q(next_state, next_action)
                - self.get_Q(state, action)
            )
        else:
            td_error = reward - self.get_Q(state, action)

        new_value = self.get_Q(state, action) + self.alpha * td_error
        self.set_Q(state, action, new_value)

    def max_action(self, state):
        """
        q(s, a) ごとの最大値のインデックスを取得
        """
        actions = self.get_possible_actions(state)
        best_action = []
        best_q_value = float("-inf")

        for action in actions:
            q_s_a = self.get_Q(state, action)
            if q_s_a > best_q_value:
                best_action = [action]
                best_q_value = q_s_a
            elif q_s_a == best_q_value:
                best_action.append(action)
        return np.random.choice(np.array(best_action))

    def get_action(self, state):
        """
        epsilon-greedy 方策ごととして行動を選択する
        """
        actions = self.get_possible_actions(state)

        if len(actions) == 0:
            return None

        if np.random.random() < self.epsilon:
            a = np.random.choice(actions)
            return a
        else:
            a = self.max_action(state)
            return a


def train_agent(env, agent, episode_cnt=10000, tmax=10000, anneal_eps=True):
    """
    学習アルゴリズム
    """
    episode_rewards = []
    for i in range(episode_cnt):
        G = 0
        state = env.reset()[0]
        action = agent.get_action(state)
        for t in range(tmax):
            next_state, reward, done, _, _ = env.step(action)
            next_action = agent.get_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            G += reward
            if done:
                episode_rewards.append(G)
                # 学習期間を超えて探索確率 epsilon を減らすために
                if anneal_eps:
                    agent.epsilon = agent.epsilon * 0.99
                break
            state = next_state
            action = next_action
    return np.array(episode_rewards)


if __name__ == "__main__":
    env = gym.envs.toy_text.CliffWalkingEnv()
    print(env.__doc__)

    # SARSA エージェントを作成
    agent = SARSAAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.nA),
    )

    # エージェントを学習し、各エピソードごとの報酬を獲得する
    rewards = train_agent(env, agent, episode_cnt=5000)

    # 報酬を可視化する
    plot_rewards("Cliff World", rewards, "SARSA")

    # 方策を出力
    print_policy(env, agent)

    # タクシーの環境を作成
    env = gym.make("Taxi-v3")

    # SARSAエージェントを作成
    agent = SARSAAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.action_space.n),
    )

    # エージェントを学習して各エピソードの報酬を獲得
    rewards = train_agent(env, agent, episode_cnt=5000)

    # 報酬のグラフを可視化する
    plot_rewards("Taxi", rewards, "SARSA")
