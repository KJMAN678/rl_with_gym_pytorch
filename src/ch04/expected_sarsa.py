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


class ExpectedSARSAAgent:
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

    # Expected SARSA の更新
    def update(self, state, action, reward, next_state, done):
        if not done:
            best_next_action = self.max_action(next_state)
            actions = self.get_possible_actions(next_state)
            next_q = 0
            for next_action in actions:
                if next_action == best_next_action:
                    next_q += (
                        1 - self.epsilon + self.epsilon / len(actions)
                    ) * self.get_Q(next_state, next_action)
                else:
                    next_q += (self.epsilon / len(actions)) * self.get_Q(
                        next_state, next_action
                    )

            td_error = reward + self.gamma * next_q - self.get_Q(state, action)
        else:
            td_error = reward - self.get_Q(state, action)

        next_value = self.get_Q(state, action) + self.alpha * td_error
        self.set_Q(state, action, next_value)

    def max_action(self, state):
        """state S ごとの 行動 Q(s, a) を最大化するベストなQ()S, A)を獲得"""
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
        """活用のための epsilon-greedy 法ごとに行動を選択"""
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
    """エージェントの学習"""
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
                # トレーニング期間中の探索確率εを小さくすることができます。
                # このフラグをFalseに設定して、エピソード報酬に与える影響を見ることができます。
                if anneal_eps:
                    agent.epsilon = agent.epsilon * 0.99
                break
            state = next_state
    return np.array(episode_rewards)


if __name__ == "__main__":
    env = gym.envs.toy_text.CliffWalkingEnv()
    print(env.__doc__)

    # Expected SARSA エージェントの初期化
    agent = ExpectedSARSAAgent(
        alpha=1.0,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.action_space.n),
    )

    # エピソードごとにエージェントを学習し、報酬を獲得
    rewards = train_agent(env, agent, episode_cnt=5000)

    # エピソードごとの報酬をプロット
    plot_rewards("Cliff World", rewards, "Expected SARSA")

    # Cliff World の最適な方策を表示
    print_policy(env, agent)

    # Taxi の環境を作成
    env = gym.make("Taxi-v3")

    # Expected SARSA エージェントの初期化
    agent = ExpectedSARSAAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.action_space.n),
    )

    # エピソードごとにエージェントを学習し、報酬を獲得
    rewards = train_agent(env, agent, episode_cnt=5000)

    # 報酬グラフをプロット
    plot_rewards("Taxi", rewards, "Expected SARSA")
