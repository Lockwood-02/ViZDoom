import gym
import torch
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from envs.env_setup import VizDoomEnv

# Initialize ViZDoom environment
env = VizDoomEnv()
check_env(env)  # Ensure the environment conforms to Gym's API

# Create TD3 policy with action noise (for exploration)
n_actions = env.action_space.n
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Define the TD3 model
model = TD3("CnnPolicy", env, action_noise=action_noise, verbose=1, device="cuda", buffer_size=100000, batch_size=64)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
# model.save("models/td3_vizdoom")

# Load the model (optional)
# model = TD3.load("models/td3_vizdoom", env=env)

# Test the trained model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
