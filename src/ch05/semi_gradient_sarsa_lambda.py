import base64
import glob
import io
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from tiles3 import IHT, tiles
from tqdm import tqdm
from utils import plot_rewards


def accumulating_trace(trace, active_features, gamma, lambd):
    trace *= gamma * lambd
    trace[active_features] += 1
    return trace


def replacing_trace(trace, active_features, gamma, lambd):
    trace *= gamma * lambd
    trace[active_features] = 1
    return trace


class QEstimator:
    def __init__(
        self,
        step_size,
        num_of_tilings=8,
        tiles_per_dim=8,
        max_size=2048,
        epsilon=0.0,
        trace_fn=replacing_trace,
        lambd=0,
        gamma=1.0,
    ):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.tiles_per_dim = tiles_per_dim
        self.epsilon = epsilon
        self.lambd = lambd
        self.gamma = gamma

        self.step_size = step_size / num_of_tilings
        self.trace_fn = trace_fn

        self.table = IHT(max_size)

        self.w = np.zeros(max_size)
        self.trace = np.zeros(max_size)

        self.pos_scale = self.num_of_tilings / (
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
        if pos == env.observation_space.high[0]:
            return 0.0
        else:
            active_features = self.get_active_features(state, action)
            return np.sum(self.w[active_features])

    def q_update(self, state, action, reward, next_state, next_action):
        """与えられた状態、行動、ターゲットから学習する"""
        active_features = self.get_active_features(state, action)

        q_s_a = self.q_predict(state, action)
        target = reward + self.gamma * self.q_predict(next_state, next_action)
        delta = target - q_s_a

        if self.trace_fn == accumulating_trace or self.trace_fn == replacing_trace:
            self.trace = self.trace_fn(
                self.trace, active_features, self.gamma, self.lambd
            )
        else:
            self.trace = self.trace_fn(self.trace, active_features, self.gamma, 0)

        self.w += self.step_size * delta * self.trace

    def get_eps_greedy_action(self, state):
        pos, vel = state
        if np.random.rand() < self.epsilon:
            return np.random.choice(env.action_space.n)
        else:
            qvals = np.array(
                [self.q_predict(state, action) for action in range(env.action_space.n)]
            )
            return np.argmax(qvals)


def sarsa_lambda(qhat, episode_cnt=10000, max_size=2048, gamma=1.0):
    episode_rewards = []
    for i in tqdm(range(episode_cnt)):
        state = env.reset()[0]
        action = qhat.get_eps_greedy_action(state)
        qhat.trace = np.zeros(max_size)
        episode_reward = 0
        while True:
            next_state, reward, done, _, _ = env.step(action)
            next_action = qhat.get_eps_greedy_action(next_state)
            episode_reward += reward
            qhat.q_update(state, action, reward, next_state, next_action)
            if done:
                episode_rewards.append(episode_reward)
                break
            state = next_state
            action = next_action
    return np.array(episode_rewards)


def generate_animation(env, estimator, save_path):
    try:
        video = VideoRecorder(env, save_path)
        state = env.reset()[0]
        t = 0
        while True:
            time.sleep(0.01)
            action = estimator.get_eps_greedy_action(state)
            state, _, done, _, _ = env.step(action)
            env.render()
            video.capture_frame()
            t += 1
            if done:
                print(f"Solved in {t} steps.")
                video.close()
                break

    except gym.error.Error as e:
        print(e)


def display_animation(file_path):
    """読み込んだ mp4 ファイルを再生"""
    cap = cv2.VideoCapture(file_path)

    if cap.isOpened() == False:
        print("ビデオファイルを開くとエラーが発生しました")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("Video", frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        else:
            break

    cap.release()


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="rgb_array")  # この環境専用の実装となっている
    env._max_episode_steps = 500

    np.random.seed(13)
    env.reset()
    plt.title("Mountain Car Environment")
    plt.imshow(env.render())
    plt.show()
    env.close()

    # n-SARSA学習エージェントを作成
    step_size = 0.8
    episode_cnt = 1000
    epsilon = 0.0
    gamma = 1.0
    lambd = 0.5
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env._max_episode_steps = 500
    np.random.seed(13)

    # この2つのトレースを使って、その挙動を確認することができます。
    # 限界値には大きな違いはありませんが、パラメータstep_size、lambdaの挙動は、特に解析の初期サイクルにおいて異なります。
    # 詳細な説明と導出はSutton and Barto bookの12章を参照してください。
    estimator = QEstimator(
        step_size=step_size,
        epsilon=epsilon,
        lambd=lambd,
        gamma=gamma,
        trace_fn=accumulating_trace,
    )
    rewards = sarsa_lambda(estimator, episode_cnt=episode_cnt, gamma=gamma)

    # 報酬の可視化
    plot_rewards("Mountain Car World", rewards, "Semi Grad SARSA(λ). Replacing trace")

    path = "src/ch05/video/mountain_car_semi_grad_sarsa_lambda.mp4"

    generate_animation(
        env,
        estimator,
        save_path=path,
    )

    display_animation(path)
