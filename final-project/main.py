import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
print(env.action_space.shape)

obs, info = env.reset()
episode_over = False 
while not episode_over:
    # placeholder. would replace with NN to select best next action
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
