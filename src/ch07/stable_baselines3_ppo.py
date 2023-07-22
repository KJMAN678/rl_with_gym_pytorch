import glob
import os
import shutil
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.ppo.policies import MlpPolicy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import display_animation, make_env, torch_fix_seed

ENV_NAME = "CartPole-v1"


def generate_animation(env, model, video_length=500, save_dir="videos/baselines/ppo"):
    try:
        env = RecordVideo(env, save_dir, episode_trigger=lambda id: True)
    except gym.error.Error as e:
        print(e)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    obs = env.reset()[0]
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    env.close()


def main():
    torch_fix_seed(42)

    env = make_env(ENV_NAME)
    env.reset()
    plt.imshow(env.render())
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    state_dim = state_shape[0]
    env.close()

    model = PPO(MlpPolicy, env, verbose=0)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    print(f"mean_reward: {mean_reward :.2f} +/- {std_reward :.2f}")

    # Train the agent for 30000 steps
    model.learn(total_timesteps=3000)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    if os.path.exists("./videos/"):
        shutil.rmtree("./videos/")

    save_dir = "videos/baselines/ppo"
    env = make_env(ENV_NAME)
    generate_animation(env, model, video_length=500, save_dir="videos/baselines/ppo")
    [filepath] = glob.glob(os.path.join(save_dir, "*.mp4"))
    display_animation(filepath)
    env.close()

    # record_video("CartPole-v1", model, video_length=500, prefix="ppo2-cartpole")


if __name__ == "__main__":
    main()
