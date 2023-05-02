import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="rgb_array")

obs = env.reset()
print("initial observation:", obs)

print("possible actions:", env.action_space.n)


def policy(observation):
    return env.action_space.sample()


for _ in range(5):
    env.render()

    action = policy(obs)
    print("\ntaking action:", action)

    obs, reward, done, truncated, info = env.step(action)
    print(f"got reward: {reward} New state/observation is: {obs}")

env.close()
