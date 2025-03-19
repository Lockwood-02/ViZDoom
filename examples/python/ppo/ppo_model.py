# File path: models/ppo_model.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.vizdoom_env import ViZDoomEnv

def train_ppo_model(config_path, total_timesteps=10000):
    # Set up the environment and the PPO model
    env = DummyVecEnv([lambda: ViZDoomEnv(config_path)])
    model = PPO("CnnPolicy", env, verbose=1)
    
    episode_rewards = []
    for timestep in range(total_timesteps):
        # Step through environment and collect rewards
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        
        # Yield episode reward (and other metrics if needed)
        yield total_reward, None  # Placeholder for loss metric

    model.save("ppo_vizdoom")
