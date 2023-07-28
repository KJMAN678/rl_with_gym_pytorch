import random
import sys
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gymnasium.envs import toy_text


class DynaQAgent:
    def __init__(self, alpha, epsilon, gamma, get_possible_actions, n):
        self.get_possible_actions = get_possible_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.n = n
        self._Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.buffer = {}

    def get_Q(self, state, action):
        return self._Q[state][action]

    def set_Q(self, state, action, value):
        self._Q[state][action] = value

    # Q learning update step
    def update(self, state, action, reward, next_state, done):
        if not done:
            best_next_action = self.max_action(next_state)
            td_error = (
                reward
                + self.gamma * self.get_Q(next_state, best_next_action)
                - self.get_Q(state, action)
            )
        else:
            td_error = reward - self.get_Q(state, action)

        new_value = self.get_Q(state, action) + self.alpha * td_error
        self.set_Q(state, action, new_value)

    # get best A for Q(S, A) which maximizes Q(S, a) for actions in state S
    def max_action(self, state):
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

    # choose action as per ε-greedy policy for exploration
    def get_action(self, state):
        actions = self.get_possible_actions(state)

        if len(actions) == 0:
            return None

        if np.random.random() < self.epsilon:
            a = np.random.choice(actions)
            return a
        else:
            a = self.max_action(state)
            return a


# plot rewards
def plot_rewards(env_name, rewards, label):
    plt.title(f"env={env_name}, Mean reward = {np.mean(rewards[-20:]):.1f}")
    plt.plot(rewards, label=label)
    plt.grid()
    plt.legend()
    plt.ylim(-300, 0)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


# training algorithm
def train_agent(env, agent, episode_cnt=10000, tmax=10000, anneal_eps=True):
    episode_rewards = []
    for i in range(episode_cnt):
        G = 0
        state = env.reset()[0]
        for t in range(tmax):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            G += reward
            if done:
                episode_rewards.append(G)
                # to reduce the exploration probability epsilon over the training period.
                if anneal_eps:
                    agent.epsilon = agent.epsilon * 0.99
                break
            # add the experience to agent’s buffer (i.e. agent’s model estimate)
            agent.buffer[(state, action)] = (next_state, reward, done)
            state = next_state
            # plan n steps through simulated experience
            for j in range(agent.n):
                state_v, action_v = random.choice(list(agent.buffer))
                next_state_v, reward_v, done_v = agent.buffer[(state_v, action_v)]
                agent.update(state_v, action_v, reward_v, next_state_v, done_v)

    return np.array(episode_rewards)


# helper function to print policy for Cliff world
def print_policy(env, agent):
    nR, nC = env._cliff.shape

    actions = "^>v<"

    for y in range(nR):
        for x in range(nC):
            if env._cliff[y, x]:
                print(" C ", end="")
            elif (y * nC + x) == env.start_state_index:
                print(" X ", end="")
            elif (y * nC + x) == nR * nC - 1:
                print(" T ", end="")
            else:
                print(f" {actions[agent.max_action(y * nC + x)]} ", end="")
        print()


def main():
    # create cliff world environment
    # env = gym.envs.toy_text.CliffWalkingEnv()
    env = gym.envs.toy_text.CliffWalkingEnv()
    print(env.__doc__)
    # env.reset()

    # create a Dyna-Q Learning agent
    agent = DynaQAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.nA),
        n=20,
    )

    # train agent and get rewards for episodes
    rewards = train_agent(env, agent, episode_cnt=500)

    # Plot rewards
    plot_rewards("Cliff World", rewards, "Dyna Q")

    # print policy
    print_policy(env, agent)
    # env.close()

    # create taxi environment
    env = gym.make("Taxi-v3")
    # env.reset()

    # create a Q Learning agent
    agent = DynaQAgent(
        alpha=0.25,
        epsilon=0.2,
        gamma=0.99,
        get_possible_actions=lambda s: range(env.action_space.n),
        n=50,
    )

    # train agent and get rewards for episodes
    rewards = train_agent(env, agent, episode_cnt=500)

    # plot reward graph
    plot_rewards("Taxi", rewards, "Dyna Q Learning")
    # env.close()


if __name__ == "__main__":
    main()
