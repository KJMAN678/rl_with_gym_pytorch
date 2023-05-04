from collections import defaultdict

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
        """Q学習の更新"""
        if not done:
            best_next_action = self.max_action(next_state)
            td_error = (
                reward
                + self.gamma * self.get_Q(next_state, best_next_action)
                - self.get_Q(state, action)
            )
        else:
            td_error = reward - self.get_Q(state, action)

        next_value = self.get_Q(state, action) + self.alpha * td_error
        self.set_Q(state, action, next_value)

    def max_action(self, state):
        """状態sにおける行動Q(s, a)を最大化するベストなQ(s, a)を獲得"""
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
