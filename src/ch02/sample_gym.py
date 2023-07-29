import gymnasium as gym
import numpy as np


def policy(env: gym.wrappers.time_limit.TimeLimit, observation: tuple) -> np.int64:
    print(type(env.action_space.sample()))
    return env.action_space.sample()


def main():
    env = gym.make("MountainCar-v0", render_mode="rgb_array")

    obsservation = env.reset()

    print("initial observation:", obsservation)
    print("possible actions:", env.action_space.n)

    for _ in range(5):
        env.render()

        action = policy(env, obsservation)
        print("\ntaking action:", action)

        obs, reward, done, truncated, info = env.step(action)
        print(f"got reward: {reward} New state/observation is: {obsservation}")

    env.close()


if __name__ == "__main__":
    main()
