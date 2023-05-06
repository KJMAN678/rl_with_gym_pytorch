import base64
import glob
import io
import os
import random
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import convolve, gaussian
from tqdm import trange
from utils import torch_fix_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array").unwrapped
    return env


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()

        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        state_dim = state_shape[0]

        self.network = nn.Sequential()
        self.network.add_module("layer1", nn.Linear(state_dim, 192))
        self.network.add_module("relu1", nn.ReLU())
        self.network.add_module("layer2", nn.Linear(192, 256))
        self.network.add_module("relu2", nn.ReLU())
        self.network.add_module("layer3", nn.Linear(256, 64))
        self.network.add_module("relu3", nn.ReLU())
        self.network.add_module("layer4", nn.Linear(64, n_actions))

        self.parameters = list(self.network.parameters())

    def forward(self, state_t):
        """時刻tの状態をネットワークを通してQ(s,a)を得るために渡す。す"""
        qvalues = self.network(state_t)
        return qvalues

    def get_qvalues(self, states):
        """inputはnumpyの状態の配列、outputはnumpyの配列としてのQvalsです。"""
        states = np.array(states)  # 高速化のため np.array に変更
        states = torch.tensor(states, device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """q_valueのバッチからε-greedyポリシーでサンプルアクションを作成する。"""
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for _ in range(n_games):
        s = env.reset()[0]
        reward = 0
        for _ in range(t_max):
            s = s.tolist()  # 高速化のため np.darray を list に変更
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _, _ = env.step(action)
            reward += r
            if done:
                break
        rewards.append(reward)
    return np.mean(rewards)


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.next_id = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)


def play_and_record(start_state, agent, env, exp_replay, n_steps=1):
    s = start_state
    sum_rewards = 0

    # バッファー中のステップと遷移の記録ごとにゲームをプレイする
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)[0]
        next_s, r, done, _, _ = env.step(a)
        sum_rewards += r
        exp_replay.add(s, a, r, next_s, done)
        if done:
            s = env.reset()[0]
        else:
            s = next_s
    return sum_rewards, s


def compute_td_loss(
    agent, target_network, states, actions, rewards, next_states, done_flags, gamma=0.99, device=device
):
    """numpy の array を Tensor に変換する"""
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype("float32"), device=device, dtype=torch.float)

    # 現在の状態の全ての行動ごとにQ値を獲得する
    # エージェントのネットワークを使う
    predicted_qvalues = agent(states)

    # 次の状態の全ての行動ごとにQ値を計算する
    # エージェントのネットワークを使う
    predicted_next_qvalues = target_network(next_states)

    # 選んだ行動ごとにQ値を選択する
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # 予測された次のQ値を使って Qmax(next_states, actions) を計算する
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    # 対象のQ値を計算する
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1 - done_flags)

    # 最小化のため平均二乗誤差を計算する
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    return loss


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps - start_eps) * min(step, final_step) / final_step


def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, "valid")


def main():
    seed = 42
    torch_fix_seed(seed)

    torch.backends.cudnn.benchmark = True

    env_name = "CartPole-v1"
    env = make_env(env_name)
    env.reset(seed=seed)
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    state_shape, n_actions = env.observation_space.shape, env.action_space.n

    print(device)

    agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(device)

    print(f"score: {evaluate(env, agent, n_games=1)}")
    env.close()

    target_network = DQNAgent(agent.state_shape, agent.n_actions, epsilon=0.5).to(device)
    target_network.load_state_dict(agent.state_dict())

    # Main Loop
    env_name = "CartPole-v1"
    env = make_env(env_name)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset(seed=seed)[0]

    agent = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
    target_network = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())

    # フルランダムポリシーを使用して、いくつかのサンプルで経験リプレイを満たしてみましょう。
    exp_replay = ReplayBuffer(10**4)
    for i in range(100):
        play_and_record(state, agent, env, exp_replay, n_steps=10**2)
        if len(exp_replay) == 10**4:
            break
    print(f"Experienc Replay Size: {len(exp_replay)}")

    # パラメータ
    timesteps_per_epoch = 1
    batch_size = 256
    total_steps = 5 * 10**4

    opt = torch.optim.Adam(list(agent.parameters), lr=1e-4)

    start_epsilon = 1
    end_epsilon = 0.05
    eps_decay_final_step = 2 * 10**4

    loss_freq = 20
    refresh_target_network_freq = 100
    eval_freq = 1000

    max_grad_norm = 5000

    mean_rw_history = []
    td_loss_history = []

    state = env.reset(seed=seed)[0]

    for step in trange(total_steps + 1):
        # 進歩として探索を減らす
        agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)

        # timesteps_per_epochを取得し、experience replay バッファを更新する。
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # experience replay からバッチサイズのデータをサンプリングして学習させる
        states, actions, rewards, next_states, done_flags = exp_replay.sample(batch_size)

        # TD誤差を計算
        loss = compute_td_loss(
            agent, target_network, states, actions, rewards, next_states, done_flags, gamma=0.99, device=device
        )

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters, max_grad_norm)
        opt.step()
        opt.zero_grad()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())

        if step % refresh_target_network_freq == 0:
            # エージェントの重みをtarget_networkに読み込む
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            # エージェントを評価する
            mean_rw_history.append(evaluate(make_env(env_name), agent, n_games=3, greedy=True, t_max=1000))

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[16, 9])

            axes[0].set_title("Mean reward per episode")
            axes[0].plot(mean_rw_history)
            axes[0].grid()

            assert not np.isnan(td_loss_history[-1])
            axes[1].set_title("TD loss history (smoothened)")
            axes[1].plot(smoothen(td_loss_history))
            axes[1].grid()

            fig.suptitle(
                f"step = {step}/{total_steps}, buffer size = {len(exp_replay)}, epsilon = {agent.epsilon: .5f}"
            )

            plt.show(block=False)
            if step == total_steps:
                plt.savefig("src/ch06/result.png")
            plt.pause(0.5)
            plt.close()

    final_score = evaluate(make_env(env_name), agent, n_games=30, greedy=True, t_max=1000)
    print(f"final score: {final_score: .2f}")
    print("Well done")


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    elasped_time = time_end - time_start
    print(f"{elasped_time: .2f} second elasped.")
