import base64
import glob
import io
import os
import shutil
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import RecordVideo
from IPython.display import HTML, clear_output
from scipy.signal import convolve, gaussian
from tqdm import trange

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import make_env, torch_fix_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        state_dim = state_shape[0]
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.network = nn.Sequential()
        self.network.add_module("layer1", nn.Linear(state_dim, 192))
        self.network.add_module("relu1", nn.ReLU())
        self.network.add_module("layer2", nn.Linear(192, 256))
        self.network.add_module("relu2", nn.ReLU())
        self.network.add_module("layer3", nn.Linear(256, 64))
        self.network.add_module("relu3", nn.ReLU())
        self.network.add_module("layer4", nn.Linear(64, n_actions))

        self.parameters = self.network.parameters()

    def forward(self, state_t):
        # pass the state at time t through the network to get Q(s,a)
        qvalues = self.network(state_t)
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


class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6, beta=0.4):
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to holde buffer
        self.next_id = 0
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.ones(size)
        self.epsilon = 1e-5

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        max_priority = self.priorities.max()
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.priorities[self.next_id] = max_priority
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size):
        priorities = self.priorities[: len(self.buffer)]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()
        N = len(self.buffer)
        weights = (N * probabilities) ** (-self.beta)
        weights /= weights.max()

        idxs = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        weights = weights[idxs]

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(done_flags),
            np.array(weights),
            np.array(idxs),
        )

    def update_priorities(self, idxs, new_priorities):
        self.priorities[idxs] = new_priorities + self.epsilon


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


def compute_td_loss_priority_replay(
    agent,
    target_network,
    replay_buffer,
    states,
    actions,
    rewards,
    next_states,
    done_flags,
    weights,
    buffer_idxs,
    gamma=0.99,
    device=device,
):
    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), device=device, dtype=torch.float)
    actions = torch.tensor(np.array(actions), device=device, dtype=torch.long)
    rewards = torch.tensor(np.array(rewards), device=device, dtype=torch.float)
    next_states = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
    done_flags = torch.tensor(np.array(done_flags.astype("float32")), device=device, dtype=torch.float)
    weights = torch.tensor(np.array(weights), device=device, dtype=torch.float)

    # get q-values for all actions in current states use agent network
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states use target network
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute Q max(next_states, actions) using predicted next q-values
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    # compute "target q-values"
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1 - done_flags)

    # compute each sample TD error
    loss = ((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2) * weights

    # mean squared error loss to minimize
    loss = loss.mean()

    # calculate new priorities and update buffer
    with torch.no_grad():
        new_priorities = predicted_qvalues_for_actions.detach() - target_qvalues_for_actions.detach()
        new_priorities = np.absolute(new_priorities.detach().numpy())
        replay_buffer.update_priorities(buffer_idxs, new_priorities)

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
        if done or reward >= 1000:
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


def sample():
    seed = 42
    torch_fix_seed(seed)

    env_name = "CartPole-v1"

    env = make_env(env_name)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    print(device)

    state_shape, n_actions = env.observation_space.shape, env.action_space.n

    agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(device)

    print(evaluate(env, agent, n_games=1))
    env.close()


def main():
    seed = 42
    torch_fix_seed(seed)

    env_name = "CartPole-v1"

    env = make_env(env_name)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()[0]

    agent = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
    target_network = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())

    # let us fill experience replay with some samples using full random policy
    exp_replay = PrioritizedReplayBuffer(10**4)
    for i in range(100):
        play_and_record(state, agent, env, exp_replay, n_steps=10**2)
        if len(exp_replay) == 10**4:
            break
    print(len(exp_replay))

    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = 45000

    # init Optimizer
    opt = torch.optim.Adam(agent.parameters, lr=1e-4)

    # set exploration epsilon
    start_epsilon = 1.0
    end_epsilon = 0.05
    eps_decay_final_step = 10**4

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
        states, actions, rewards, next_states, done_flags, weights, idxs = exp_replay.sample(batch_size)

        # loss = <compute TD loss>
        loss = compute_td_loss_priority_replay(
            agent,
            target_network,
            exp_replay,
            states,
            actions,
            rewards,
            next_states,
            done_flags,
            weights,
            idxs,
            gamma=0.99,
            device=device,
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
            mean_rw_history.append(evaluate(make_env(env_name), agent, n_games=3, greedy=True, t_max=1000))

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

    final_score = evaluate(make_env(env_name), agent, n_games=30, greedy=True, t_max=1000)
    print(f"final score: {final_score}")
    print("Well done")

    # Animate learned policy
    if os.path.exists("./videos/"):
        shutil.rmtree("./videos/")

    save_dir = "./videos/pytorch/6_3/"
    env = make_env(env_name)
    generate_animation(env, agent, save_dir=save_dir)
    [filepath] = glob.glob(os.path.join(save_dir, "*.mp4"))
    display_animation(filepath)
    env.close()


if __name__ == "__main__":
    sample()
    main()
