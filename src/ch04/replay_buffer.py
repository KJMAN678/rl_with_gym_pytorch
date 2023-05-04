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

    def get_action(self, state):
        """探索のためのepsilon-greedy方策ごとに行動を選択する"""
        actions = self.get_possible_actions(state)

        if len(actions) == 0:
            return None

        if np.random.random() < self.epsilon:
            a = np.random.choice(actions)
            return a
        else:
            a = self.max_action(state)
            return a


class ReplayBuffer:
    def __init__(self, size):
        self.size = size  # バッファ内のアイテムの最大数
        self.buffer = []  # バッファを持つための配列

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, rewawrd, next_state, done):
        item = (state, action, reward, next_state, done)
        self.buffer = self.buffer[-self.size :] + [item]

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*sample))
        return states, actions, rewards, next_states, done_flags


def train_agent(
    env,
    agent,
    episode_cnt=10000,
    tmax=10000,
    anneal_eps=True,
    replay_buffer=None,
    batch_size=16,
):
    episode_rewards = []
    for i in range(episode_cnt):
        G = 0
        state = env.reset()[0]
        for t in range(tmax):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            if replay_buffer:
                replay_buffer.add(state, action, reward, next_state, done)
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    done_flags,
                ) = replay_buffer.sample(batch_size)
                for i in range(batch_size):
                    agent.update(
                        states[i], actions[i], rewards[i], next_states[i], done_flags[i]
                    )
            else:
                agent.update(state, action, reward, next_state, done)

            G += reward
            if done:
                episode_rewards.append(G)
                # 学習期間中に探索確率 epsilon を減らすこと
                if anneal_eps:
                    agent.epsilon = agent.epsilon * 0.99
                break
            state = next_state
    return np.array(episode_rewards)


if __name__ == "__main__":
    env = toy_text.CliffWalkingEnv()
    print(env.__doc__)

    # Q学習エージェントの初期化
    agent = QLearningAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.nA),
    )

    # リプレイバッファを使ったエージェント学習を行、エピソードごとの報酬を獲得する
    rewards = train_agent(env, agent, episode_cnt=5000, replay_buffer=ReplayBuffer(512))

    # エピソードごとの報酬を可視化
    plot_rewards("Cliff World", rewards, "Q-Learning(Replay Buffer)")

    # エージェントによる方策の学習結果を可視化
    print_policy(env, agent)

    # タクシー環境を作成
    env = gym.make("Taxi-v3")

    # Q学習エージェントの初期化
    agent = QLearningAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.action_space.n),
    )

    # リプレイバッファによりエージェントを学習し、エピソードごとの報酬を獲得する
    rewards = train_agent(env, agent, episode_cnt=5000, replay_buffer=ReplayBuffer(512))

    # 報酬のグラフを可視化する
    plot_rewards("Taxi", rewards, "QAgent(replay)")
