import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gymnasium.core import ObservationWrapper
from q_learning import QLearningAgent
from replay_buffer import ReplayBuffer
from tqdm import tqdm
from utils import plot_rewards


def train_agent(
    env,
    agent,
    episode_cnt=10000,
    tmax=10000,
    anneal_eps=True,
    replay_buffer=None,
    batch_size=16,
):
    """リプレイバッファーによる学習アルゴリズム"""
    episode_rewards = []
    for i in tqdm(range(episode_cnt)):
        G = 0
        state = env.reset()[0]
        for t in range(tmax):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            if replay_buffer:
                replay_buffer.add(state, action, reward, next_state, done)
                (states, actions, rewards, next_states, done_flags) = replay_buffer(
                    batch_size
                )

                for i in range(batch_size):
                    agent.update(
                        states[i], actions[i], rewards[i], next_states[i], done_flags[i]
                    )
            else:
                agent.update(state, action, reward, next_state, done)
            G += reward
            if done:
                episode_rewards.append(G)
                # 学習期間中に探索確率εを下げる
                if anneal_eps:
                    agent.epsilon = agent.epsilon * 0.99
                break
            state = next_state
    return np.array(episode_rewards)


class Discretizer(ObservationWrapper):
    """
    環境をラップするために gymのObservationWrapperクラスを使用します
    基礎となる環境からオリジナルの状態値を受け取る observation() メソッドを実装する必要があります。
    envによって外界に渡される状態値を離散化し、この離散化された状態値を用いて、エージェントはq-learningによって効果的な政策を学習する。
    """

    def observation(self, state):
        discrete_x_pos = round(state[0], 1)
        discrete_x_vel = round(state[1], 1)
        discrete_pole_angle = round(state[2], 1)
        discrete_pole_ang_vel = round(state[3], 1)

        return (
            discrete_x_pos,
            discrete_x_vel,
            discrete_pole_angle,
            discrete_pole_ang_vel,
        )


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    print("Cart Pole Environment")
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Cart Poleの環境を作る .envを使用すると、200ステップの時間制限の終了が解除されます。
    env = gym.make("CartPole-v1").env

    # Discretizerでenvをラップする
    env = Discretizer(env)

    # Q学習エージェントを作成する
    agent = QLearningAgent(
        alpha=0.5,
        epsilon=0.20,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.action_space.n),
    )

    # エピソードごとの報酬を獲得する
    rewards = train_agent(env, agent, episode_cnt=50000)
    print(rewards.shape)
    print(rewards)
    # 報酬を可視化する
    plot_rewards("CartPole", rewards, "Q-learning")

    env.close()
