from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.vizdoom_env import ViZDoomEnv

# Create and initialize PPO model
def train_ppo_model(config_file, total_timesteps=10000):
    env = DummyVecEnv([lambda: ViZDoomEnv(config_file)])
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("saved_models/ppo_vizdoom")
    return model
