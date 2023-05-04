from collections import defaultdict

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
    
        

if __name__ == "__main__":
    