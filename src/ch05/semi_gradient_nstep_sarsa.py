import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    # この環境でのみ動作するように設計されています。他の環境では変更が必要です。
    env = gym.make("MountainCar-v0", render_mode="rgb_array")

    # 200~4000の範囲で設定可能
    env._max_episode_steps = 4000

    np.random.seed(13)
    env.reset()
    plt.imshow(env.render())
    plt.show()
