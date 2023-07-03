import base64
import glob
import io
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import (AtariPreprocessing, FrameStack, RecordVideo,
                                TransformReward)
from IPython.display import HTML, clear_output
# from gymnasium.utils.play import play
from scipy.signal import convolve, gaussian
from tqdm import trange

from utils import make_env, torch_fix_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env_atari(env_name, clip_rewards=True, seed=None):
    env = gym.make(env_name, render_mode="rgb_array")
    env.metadata["render_fps"] = 30
    if seed is not None:
        env.seed(seed)
    env = AtariPreprocessing(env, screen_size=84, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    if clip_rewards:
        env = TransformReward(env, lambda r: np.sign(r))
    return env


def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride + 1


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        state_dim = state_shape[0]
        self.network = nn.Sequential()
        self.network.add_module("conv1", nn.Conv2d(4, 16, kernel_size=8, stride=4))
        self.network.add_module("relu1", nn.ReLU())
        self.network.add_module("conv2", nn.Conv2d(16, 32, kernel_size=4, stride=2))
        self.network.add_module("relu2", nn.ReLU())
        self.network.add_module("flatten", nn.Flatten())
        self.network.add_module("linear3", nn.Linear(2592, 256))
        self.network.add_module("relu3", nn.ReLU())
        self.network.add_module("linear4", nn.Linear(256, n_actions))

        self.parameters = self.network.parameters()

    def forward(self, states_t):
        # pass the state at time t through the network to get Q(s,a)
        qvalues = self.network(states_t)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and output is Q values as numpy array
        states = torch.tensor(np.array(states), device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
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
            qvalues = agent.get_qvalues([s])

            if greedy:
                action = qvalues.argmax(axis=-1)[0]
            else:
                action = agent.sample_actions(qvalues)[0]

            s, r, done, _, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


class ReplayBuffer:
    def __init__(self, size):
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to hold buffer
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

    # Play the game for n_steps and record transitions in buffer
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
    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype("float32"), device=device, dtype=torch.float)

    # get q-values for all actions in current states use agent network
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states use target network
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute Qmax(next_states, actions) using predicted next q-values
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    # compute "target q-values"
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1 - done_flags)

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    return loss


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps - start_eps) * min(step, final_step) / final_step


def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, "valid")


def generate_animation(env, agent, save_dir):
    try:
        env = RecordVideo(env, save_dir, episode_trigger=lambda id: True)
    except gym.error.Error as e:
        print(e)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state = env.reset()[0]
    reward = 0
    t = 0
    while True:
        qvalues = agent.get_qvalues([state])
        action = qvalues.argmax(axis=-1)[0]
        state, r, done, _, _ = env.step(action)
        reward += r
        t += 1
        if done or t >= 1000:
            print(f"Got reward: {reward}")
            break


def display_animation(filepath):
    video = io.open(filepath, "r+b").read()
    encoded = base64.b64encode(video)
    return HTML(
        data=f"""<video alt="test controls>
                <source src=data: video/mp4; base64, {encoded.decode('ascii')} type="video/mp4" />
                </video>"""
    )


def main():
    seed = 42
    torch_fix_seed(seed)

    env_name = "BreakoutNoFrameskip-v4"

    env = make_env(env_name)
    env.reset(seed=seed)[0]

    n_cols = 4
    n_rows = 2
    fig = plt.figure(figsize=(16, 9))

    for row in range(n_rows):
        for col in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            ax.set_title(f"row:{row} col:{col}")
            ax.imshow(env.render())
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            env.step(env.action_space.sample())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # play(env=gym.make(env_name), render_mode="rgb_array", zoom=2, fps=10)

    env = make_env_atari(env_name)
    env.reset()[0]
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    print(n_actions)
    print(env.get_action_meanings())

    for _ in range(12):
        obs, _, _, _, _ = env.step(env.action_space.sample())

    plt.figure(figsize=(12, 10))
    plt.title("Game Image")
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    obs = obs[:]
    obs = np.transpose(obs, [1, 0, 2])
    obs = obs.reshape((obs.shape[0], -1))

    plt.figure(figsize=(15, 15))
    plt.title("Agent observation(4 frmaes left to right)")
    plt.imshow(obs, cmap="gray")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    conv1 = conv2d_size_out(84, 8, 4)
    print("Conv1", conv1)
    conv2 = conv2d_size_out(conv1, 4, 2)
    print("Conv2", conv2)

    print("Input to Dense layer:", conv2 * conv2 * 32)

    agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(device)

    evaluate(env, agent, n_games=1)
    env.close()

    target_network = DQNAgent(agent.state_shape, agent.n_actions, epsilon=0.5).to(device)
    target_network.load_state_dict(agent.state_dict())

    env = make_env_atari(env_name, seed)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()[0]

    agent = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
    target_network = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())

    # let us fill experience replay with some samples using full random policy
    exp_replay = ReplayBuffer(10**4)
    for i in range(100):
        play_and_record(state, agent, env, exp_replay, n_steps=10**2)
        if len(exp_replay) == 10**4:
            break
    print(len(exp_replay))

    # setup some parameters for training
    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = 100

    # init Optimizer
    opt = torch.optim.Adam(agent.parameters, lr=1e-4)

    # set exploration epsilon
    start_epsilon = 1
    end_epsilon = 0.05
    eps_decay_final_step = 1 * 10**6

    # setup spme frequency for logging and updating target network
    loss_freq = 20
    refresh_target_network_freq = 100
    eval_freq = 1000

    # to clip the gradients
    max_grad_norm = 5000

    mean_rw_history = []
    td_loss_history = []

    state = env.reset()[0]
    for step in trange(total_steps + 1):
        # reduce exploration as we progress
        agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)

        # take timesteps_per_epoch and update experience replay buffer
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train by sampling batch_size of data from experience replay
        states, actions, rewards, next_states, done_flags = exp_replay.sample(batch_size)

        # loss = <compute TD loss>
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
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            # eval the agent
            mean_rw_history.append(
                evaluate(make_env_atari(env_name, seed=step), agent, n_games=3, greedy=True, t_max=1000)
            )

            clear_output(True)
            print(f"buffer size = {len(exp_replay)}, epsilon = {agent.epsilon: .5f}")

            plt.figure(figsize=[16, 5])
            plt.subplot(1, 2, 1)
            plt.title("Mean reward per episode")
            plt.plot(mean_rw_history)
            plt.grid()

            assert not np.isnan(td_loss_history[-1])
            plt.subplot(1, 2, 2)
            plt.title("TD loss history (smoothened)")
            plt.plot(smoothen(td_loss_history))
            plt.grid()

            plt.show(block=False)
            plt.pause(2)
            plt.close()

    final_score = evaluate(make_env_atari(env_name), agent, n_games=1, greedy=True, t_max=1000)
    print(f"final score: {final_score}")

    # Animate learned policy
    if os.path.exists("./videos/"):
        shutil.rmtree("./videos/")

    save_dir = "./videos/pytorch/6_2/"
    env = make_env_atari(env_name)
    generate_animation(env, agent, save_dir=save_dir)
    # [filepath] = glob.glob(os.path.join(save_dir, "*.mp4"))
    # display_animation(filepath)
    env.close()


if __name__ == "__main__":
    main()
