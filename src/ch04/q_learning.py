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


class QLearningAgent:
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

    def update(self, state, action, reward, next_state, done):
        """Q learning の ステップ更新"""
        if not done:
            best_next_action = self.max_action(next_state)
            td_error = (
                reward
                + self.gamma * self.get_Q(next_state, best_next_action)
                - self.get_Q(state, action)
            )
        else:
            td_error = reward - self.get_Q(state, action)

        new_value = self.get_Q(state, action) + self.alpha * td_error
        self.set_Q(state, action, new_value)

    def max_action(self, state):
        """状態 s の行動ごとに Q(S, a) を最大化するベストな Q(S, A) を獲得"""
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
        """探索のための epsilon-greedy な方策ごとに行動を選択する"""
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
    """学習アルゴリズム"""
    episode_rewards = []
    for i in range(episode_cnt):
        G = 0
        state = env.reset()[0]
        for t in range(tmax):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            G += reward
            if done:
                episode_rewards.append(G)
                # 学習期間全体で epsilon 活用確率を減らすために
                if anneal_eps:
                    agent.epsilon = agent.epsilon * 0.99
                break
            state = next_state
    return episode_rewards


if __name__ == "__main__":
    env = toy_text.CliffWalkingEnv()
    print(env.__doc__)

    # Q learning agent の初期化
    agent = QLearningAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.nA),
    )

    # エージェントの学習とエピソードごとの報酬の獲得
    rewards = train_agent(env, agent, episode_cnt=5000)

    # 報酬のプロット
    plot_rewards("Cliff World", rewards, "Q-learning")

    # 方策を出力
    print_policy(env, agent)

    # タクシー環境の作成
    env = gym.make("Taxi-v3")

    # Q learning agent の初期化
    agent = QLearningAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.action_space.n),
    )

    # エージェントの学習とエピソードごとの報酬の獲得
    rewards = train_agent(env, agent, episode_cnt=5000)

    # 報酬のプロット
    plot_rewards("Taxi", rewards, "Q-learning")
