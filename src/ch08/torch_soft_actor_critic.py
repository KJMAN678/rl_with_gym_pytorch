import glob
import itertools
import os
import shutil
import sys
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from torch.distributions.normal import Normal
from torch.optim import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import display_animation, make_env, torch_fix_seed  # type:ignore

ENV_NAME_1 = "Pendulum-v1"
ENV_NAME_2 = "LunarLanderContinuous-v2"
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, state_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)
        self.act_limit = act_limit

    def forward(self, s, deterministric=False, with_logprob=True):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministric:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=-1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.Q = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q = self.Q(x)
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.state_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # build Q and policy functions
        self.q1 = MLPQFunction(self.state_dim, self.act_dim)
        self.q2 = MLPQFunction(self.state_dim, self.act_dim)
        self.policy = SquashedGaussianMLPActor(
            self.state_dim, self.act_dim, self.act_limit
        )

    def act(self, state, deterministric=False):
        with torch.no_grad():
            a, _ = self.policy(state, deterministric, False)
            return a.numpy()

    def get_action(self, state, deterministric=False):
        return self.act(torch.as_tensor(state, dtype=torch.float32), deterministric)


class ReplayBuffer:
    def __init__(self, size=1e6):
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

    def sample(self, batch_size=32):
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


def compute_q_loss(
    agent,
    target_network,
    states,
    actions,
    rewards,
    next_states,
    done_flags,
    gamma,
    alpha,
    act_limit,
):
    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), dtype=torch.float)
    actions = torch.tensor(np.array(actions), dtype=torch.float)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float)
    done_flags = torch.tensor(np.array(done_flags.astype("float32")), dtype=torch.float)

    # get q-values for all actions in current states use agent network
    q1 = agent.q1(states, actions)
    q2 = agent.q2(states, actions)

    # Bellman backup for Q function
    with torch.no_grad():
        action_target, action_target_logp = target_network.policy(next_states)

        # Target Q
        q1_target = target_network.q1(next_states, action_target)
        q2_target = target_network.q2(next_states, action_target)
        q_target = torch.min(q1_target, q2_target)
        target = rewards + gamma * (1 - done_flags) * (
            q_target - alpha * action_target_logp
        )

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - target) ** 2).mean()
    loss_q2 = ((q2 - target) ** 2).mean()
    loss_q = loss_q1 + loss_q2

    return loss_q


def compute_policy_loss(agent, states, alpha):
    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), dtype=torch.float)

    actions, actions_logp = agent.policy(states)

    q1_values = agent.q1(states, actions)
    q2_values = agent.q2(states, actions)
    q_values = torch.min(q1_values, q2_values)

    # Entropy regularised
    loss_policy = -(q_values - alpha * actions_logp).mean()
    return loss_policy


def one_step_update(
    agent,
    target_network,
    q_params,
    q_optimizer,
    policy_optimizer,
    states,
    actions,
    rewards,
    next_states,
    done_flags,
    gamma,
    polyak,
    alpha,
    act_limit,
):
    # one step gradient for q-values
    q_optimizer.zero_grad()
    loss_q = compute_q_loss(
        agent,
        target_network,
        states,
        actions,
        rewards,
        next_states,
        done_flags,
        gamma,
        alpha,
        act_limit,
    )

    loss_q.backward()
    q_optimizer.step()

    # Freeze Q-network
    for params in q_params:
        params.requires_grad = False

    # one step gradient for policy network
    policy_optimizer.zero_grad()
    loss_policy = compute_policy_loss(agent, states, alpha)
    loss_policy.backward()
    policy_optimizer.step()

    # UnFreeze Q-network
    for params in q_params:
        params.requires_grad = True

    # update target networks with polyak averaging
    with torch.no_grad():
        for params, params_target in zip(
            agent.parameters(), target_network.parameters()
        ):
            params_target.data.mul_(polyak)
            params_target.data.add_((1 - polyak) * params.data)


def test_agent(env, agent, num_test_episodes, max_ep_len):
    ep_rets, ep_lens = [], []
    for j in range(num_test_episodes):
        state, done, ep_ret, ep_len = env.reset()[0], False, 0, 0
        while not (done or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            state, reward, done, _, _ = env.step(agent.get_action(state, True))
            ep_ret += reward
            ep_len += 1
        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)
    return np.mean(ep_rets), np.mean(ep_lens)


def sac(
    env_fn,
    seed=0,
    steps_per_epoch=4000,
    epochs=5,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    policy_lr=1e-3,
    q_lr=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=1000,
):
    torch_fix_seed(seed)

    env, test_env = env_fn(), env_fn()

    ep_rets, ep_lens = [], []

    state_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    agent = MLPActorCritic(env.observation_space, env.action_space)
    target_network = deepcopy(agent)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for params in target_network.parameters():
        params.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(replay_size)

    # List of parameters for both Q-networks
    q_params = itertools.chain(agent.q1.parameters(), agent.q2.parameters())

    # optimizers
    q_optimizer = Adam(q_params, lr=q_lr)
    policy_optimizer = Adam(agent.policy.parameters(), lr=policy_lr)

    total_steps = steps_per_epoch * epochs
    state, ep_ret, ep_len = env.reset()[0], 0, 0

    for t in range(total_steps):
        if t > start_steps:
            action = agent.get_action(state)
        else:
            action = env.action_space.sample()

        next_state, reward, done, _, _ = env.step(action)
        ep_ret += reward
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        if ep_len == max_ep_len:
            done = False
        else:
            done = done

        # Store experience to replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state

        # End of trajectory handling
        if done or (ep_len == max_ep_len):
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            state, ep_ret, ep_len = env.reset()[0], 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    done_flags,
                ) = replay_buffer.sample(batch_size)

                one_step_update(
                    agent,
                    target_network,
                    q_params,
                    q_optimizer,
                    policy_optimizer,
                    states,
                    actions,
                    rewards,
                    next_states,
                    done_flags,
                    gamma,
                    polyak,
                    alpha,
                    act_limit,
                )

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            avg_ret, avg_len = test_agent(
                test_env, agent, num_test_episodes, max_ep_len
            )
            print(
                f"End of epoch: {epoch:.0f}, Training Average Reward: {np.mean(ep_rets):.0f}, Training Average Length: {np.mean(ep_lens):.0f}"
            )
            print(
                f"End of epoch: {epoch:.0f}, Testing Average Reward: {avg_ret:.0f}, Test Average Length: {avg_len:.0f}"
            )
            ep_rets, ep_lens = [], []

    return agent


def generate_animation(env, agent, save_dir):
    try:
        env = RecordVideo(env, save_dir, episode_trigger=lambda id: True)
    except gym.error.Error as e:
        print(e)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state, done, ep_ret, ep_len = env.reset()[0], False, 0, 0
    while not done and ep_len <= 5000:
        # Take deterministic actions at test time (noise_scale=0)
        state, reward, done, _, _ = env.step(agent.get_action(state, 0))
        ep_ret += reward
        ep_len += 1

    print(f"Reward: {ep_ret}")
    env.close()


def main():
    env = make_env(ENV_NAME_1)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    state_shape, action_shape = env.observation_space.shape, env.action_space.shape
    print(f"State shape: {state_shape}")
    print(f"Action sahpe: {action_shape}")
    env.close()

    agent_1 = sac(lambda: make_env(ENV_NAME_1))

    if os.path.exists("./videos_1/"):
        shutil.rmtree("./videos_1/")

    save_dir_1 = "./videos_1/pytorch/sac/pendulum/"
    env_1 = make_env(ENV_NAME_1)

    generate_animation(env_1, agent_1, save_dir=save_dir_1)

    [filepath_1] = glob.glob(os.path.join(save_dir_1, "*.mp4"))
    display_animation(filepath_1)
    env_1.close()

    if os.path.exists("./videos_2/"):
        shutil.rmtree("./videos_2/")

    agent_2 = sac(lambda: make_env(ENV_NAME_2))

    # Animate learned policy
    save_dir_2 = "./videos_2/pytorch/sac/lunar"
    env_2 = make_env(ENV_NAME_2)
    generate_animation(env_2, agent_2, save_dir=save_dir_2)
    [filepath_2] = glob.glob(os.path.join(save_dir_2, "*.mp4"))
    display_animation(filepath_2)


if __name__ == "__main__":
    main()
