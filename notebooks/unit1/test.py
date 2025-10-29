import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

model = PPO.load("ppo-LunarLander-v3")

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")
# vec_env = make_vec_env('LunarLander-v3', n_envs=3)

# Reset the environment to generate the first observation
obs, info = env.reset(seed=42)
# obs = vec_env.reset()
for _ in range(1000):
    
    # env.render()
    # vec_env.render("human")

    # this is where you would insert your policy
    action, _states = model.predict(obs)

    # step (transition) through the environment with the action    
    obs, rewards, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
