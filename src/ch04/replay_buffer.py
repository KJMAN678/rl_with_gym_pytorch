import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gymnasium.envs import toy_text
from q_learning import QLearningAgent
from utils import plot_rewards, print_policy


class ReplayBuffer:
    def __init__(self, size):
        self.size = size  # バッファ内のアイテムの最大数
        self.buffer = []  # バッファを持つための配列

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.buffer = self.buffer[-self.size :] + [item]

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
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
    print(rewards.shape)
    print(rewards)
    # 報酬のグラフを可視化する
    plot_rewards("Taxi", rewards, "QAgent(replay)")
