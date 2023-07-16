import glob
import math
import os
import shutil
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from scipy.signal import convolve, gaussian
from tqdm import trange

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import display_animation, generate_animation, make_env, torch_fix_seed

ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        state_dim = state_shape[0]
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.q = nn.Linear(32, n_actions)

    def forward(self, state_t):
        # pass the state at time t through the network to get Q(s,a)
        x = F.relu(self.fc1(state_t))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        qvalues = self.q(x)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and output is Q values as numpy array
        states = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float32)
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
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(done_flags),
        )


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


def td_loss_dqn(
    agent,
    target_network,
    states,
    actions,
    rewards,
    next_states,
    done_flags,
    gamma=0.99,
    device=DEVICE,
):
    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float)
    actions = torch.tensor(np.array(actions), device=DEVICE, dtype=torch.long)
    rewards = torch.tensor(np.array(rewards), device=DEVICE, dtype=torch.float)
    next_states = torch.tensor(np.array(next_states), device=DEVICE, dtype=torch.float)
    done_flags = torch.tensor(
        np.array(done_flags.astype("float32")), device=DEVICE, dtype=torch.float
    )

    # get q-values for all actions in current states use agent network
    q_s = agent(states)

    # select q-values for chosen actions
    q_s_a = q_s[range(len(actions)), actions]

    with torch.no_grad():
        # compute q-values for all actions in next states use target network
        q_s1 = target_network(next_states)

        # compute Qmax(next_states, actions) using predicted next q-values
        q_s1_a1max, _ = torch.max(q_s1, dim=1)

        # compute "target q-values"
        target_q = rewards + gamma * q_s1_a1max * (1 - done_flags)

    # mean squared error loss to minimize
    loss = torch.mean((q_s_a - target_q.detach()) ** 2)

    return loss


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps - start_eps) * min(step, final_step) / final_step


def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, "valid")


def train_agent(
    env,
    agent,
    target_network,
    optimizer,
    total_steps,
    start_epsilon,
    end_epsilon,
    eps_decay_final_step,
    timesteps_per_epoch,
    batch_size,
    max_grad_norm,
    loss_freq,
    refresh_target_network_freq,
    eval_freq,
    td_loss_fn,
):
    state = env.reset()[0]
    # let us fill experience replay with some samples using full random policy
    exp_replay = ReplayBuffer(10**4)
    for i in range(100):
        play_and_record(state, agent, env, exp_replay, n_steps=10**2)
        if len(exp_replay) == 10**4:
            break
    print(f"Finished filling buffer with: {len(exp_replay)} samples.")

    mean_rw_history = []
    td_loss_history = []
    state = env.reset()[0]
    for step in trange(total_steps + 1):
        # reduce exploration as we progress
        agent.epsilon = epsilon_schedule(
            start_epsilon, end_epsilon, step, eps_decay_final_step
        )

        # take timesteps_per_epoch and update experience replay buffer
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train by sampling batch_size of data from experience replay
        states, actions, rewards, next_states, done_flags = exp_replay.sample(
            batch_size
        )

        # loss = <compute TD loss>
        optimizer.zero_grad()
        loss = td_loss_fn(
            agent,
            target_network,
            states,
            actions,
            rewards,
            next_states,
            done_flags,
            gamma=0.99,
            device=DEVICE,
        )

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            # eval the agent
            mean_rw_history.append(
                evaluate(make_env(ENV_NAME), agent, n_games=3, greedy=True, t_max=1000)
            )

        clear_output(True)

        if step % 20 == 0:
            print(
                f"step is {step}/{total_steps} {step/total_steps: .5f}  buffer size = {len(exp_replay)}, epsilon = {agent.epsilon: .5f}"
            )

        plt.figure(figsize=[16, 5])
        plt.subplot(1, 2, 1)
        plt.title("Mean reward per episode")
        plt.plot(mean_rw_history)
        plt.grid()

        assert not np.isnan(td_loss_history[-1])
        plt.subplot(1, 2, 2)
        plt.title("TD loss history (smoothended)")
        plt.plot(smoothen(td_loss_history))
        plt.grid()

        plt.show(block=False)
        plt.pause(2)
        plt.close()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_0=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        self.mu_w = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_b = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_b = nn.Parameter(torch.FloatTensor(out_features))

        self.register_buffer("epsilon_w", torch.FloatTensor(out_features, in_features))
        self.register_buffer("epsilon_b", torch.FloatTensor(out_features))

        self.reset_noise()
        self.reset_params()

    def forward(self, x):
        if self.training:
            w = self.mu_w + self.sigma_w * self.epsilon_w
            b = self.mu_b + self.sigma_b * self.epsilon_b
        else:
            w = self.mu_w
            b = self.mu_b
        return F.linear(x, w, b)

    def reset_params(self):
        k = 1 / self.in_features
        k_sqrt = math.sqrt(k)
        self.mu_w.data.uniform_(-k_sqrt * self.sigma_0)
        self.sigma_w.data.fill_(k_sqrt * self.sigma_0)
        self.mu_b.data.uniform_(-k_sqrt, k_sqrt)
        self.sigma_b.data.fill_(k_sqrt * self.sigma_0)

    def reset_noise(self):
        eps_in = self._noise(self.in_features)
        eps_out = self._noise(self.out_features)
        self.epsilon_w.copy_(eps_out.ger(eps_in))
        self.epsilon_b.copy_(self._noise(self.out_features))

    def _noise(self, size):
        x = torch.randn(size)
        x = torch.sign(x) * torch.sqrt(torch.abs(x))
        return x


class NoisyDQN(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(NoisyDQN, self).__init__()
        self.n_actons = n_actions
        self.state_shape = state_shape
        state_dim = state_shape[0]
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.fc1 = NoisyLinear(state_dim, 64)
        self.fc2 = NoisyLinear(64, 128)
        self.fc3 = NoisyLinear(128, 32)
        self.q = NoisyLinear(32, n_actions)

    def forward(self, state_t):
        # pass the state at time t through the network to get Q(s,a)
        x = F.relu(self.fc1(state_t))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        qvalues = self.q(x)
        return qvalues

    def get_qvalues(self, states):
        # pass the state at time t through the network to get Q(s,a)
        states = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using greedy policy
        batch_size, n_actions = qvalues.shape
        best_actions = qvalues.argmax(axis=-1)
        return best_actions

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.q.reset_noise()


def train_agent_noisy(
    env,
    agent,
    target_network,
    optimizer,
    total_steps,
    start_epsilon,
    end_epsilon,
    eps_decay_final_step,
    timesteps_per_epoch,
    batch_size,
    max_grad_norm,
    loss_freq,
    refresh_target_network_freq,
    eval_freq,
    td_loss_fn,
):
    state = env.reset()[0]
    # let us fill experience replay with some samples using full random policy
    exp_replay = ReplayBuffer(10**4)
    for i in range(100):
        play_and_record(state, agent, env, exp_replay, n_steps=10**2)
        if len(exp_replay) == 10**4:
            break
    print(f"Finished filling buffer with: {len(exp_replay)} samples.")

    mean_rw_history = []
    td_loss_history = []
    state = env.reset()[0]
    for step in trange(total_steps + 1):
        # reduce exploration as we progress
        agent.epsilon = epsilon_schedule(
            start_epsilon, end_epsilon, step, eps_decay_final_step
        )

        # take timesteps_per_epoch and update experience replay buffer
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train by sampling batch_size of data from experience replay
        states, actions, rewards, next_states, done_flags = exp_replay.sample(
            batch_size
        )

        # loss = <compute TD loss>
        optimizer.zero_grad()
        loss = td_loss_fn(
            agent,
            target_network,
            states,
            actions,
            rewards,
            next_states,
            done_flags,
            gamma=0.99,
            device=DEVICE,
        )

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            # eval the agent
            mean_rw_history.append(
                evaluate(make_env(ENV_NAME), agent, n_games=3, greedy=True, t_max=1000)
            )

            clear_output(True)
            print(f"buffer size = {len(exp_replay)}, epsilon = {agent.epsilon :.5f}")

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


def show_env():
    torch_fix_seed(42)

    env = make_env(ENV_NAME)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(DEVICE)
    print(evaluate(env, agent, n_games=1))
    env.close()

    target_network = DQNAgent(agent.state_shape, agent.n_actions, epsilon=0.5).to(
        DEVICE
    )
    target_network.load_state_dict(agent.state_dict())


def main():
    torch_fix_seed(42)

    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = 3 * 10**4

    start_epsilon = 1
    end_epsilon = 0.05
    eps_decay_final_step = 0.05
    eps_decay_final_step = 2 * 10**4

    loss_freq = 50
    refresh_target_network_freq = 100
    eval_freq = 1000

    max_grad_norm = 5000

    env = make_env(ENV_NAME)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim, n_actions, epsilon=1).to(DEVICE)
    target_network = DQNAgent(state_dim, n_actions, epsilon=1).to(DEVICE)
    target_network.load_state_dict(agent.state_dict())
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)

    train_agent(
        env,
        agent,
        target_network,
        optimizer,
        total_steps,
        start_epsilon,
        end_epsilon,
        eps_decay_final_step,
        timesteps_per_epoch,
        batch_size,
        max_grad_norm,
        loss_freq,
        refresh_target_network_freq,
        eval_freq,
        td_loss_fn=td_loss_dqn,
    )

    final_score = evaluate(
        make_env(ENV_NAME), agent, n_games=30, greedy=True, t_max=1000
    )
    print(f"final score] {final_score}")

    # Animate learned policy
    if os.path.exists("./videos/"):
        shutil.rmtree("./videos/")

    save_dir = "./videos/pytorch/6_6/"
    env = make_env(ENV_NAME)
    generate_animation(env, agent, save_dir=save_dir)
    [filepath] = glob.glob(os.path.join(save_dir, "*.mp4"))
    display_animation(filepath)
    env.close()


def main_noisy():
    torch_fix_seed(42)

    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = 3 * 10**4

    start_epsilon = 1
    end_epsilon = 0.05
    eps_decay_final_step = 2 * 10**4

    loss_freq = 50
    refresh_target_network_freq = 100
    eval_freq = 1000

    max_grad_norm = 5000

    env = make_env(ENV_NAME)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n

    agent = NoisyDQN(state_dim, n_actions).to(DEVICE)
    target_network = NoisyDQN(state_dim, n_actions).to(DEVICE)
    target_network.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

    train_agent_noisy(
        env,
        agent,
        target_network,
        optimizer,
        total_steps,
        start_epsilon,
        end_epsilon,
        eps_decay_final_step,
        timesteps_per_epoch,
        batch_size,
        max_grad_norm,
        loss_freq,
        refresh_target_network_freq,
        eval_freq,
        td_loss_fn=td_loss_dqn,
    )

    final_score = evaluate(
        make_env(ENV_NAME), agent, n_games=30, greedy=True, t_max=1000
    )
    print(f"final score(Noisy DQN): {final_score:.0f}")


if __name__ == "__main__":
    show_env()
    main()
    main_noisy()
