from models.ppo_model import train_ppo_model
from utils.track_metrics import track_training_metrics
from envs.vizdoom_env import ViZDoomEnv
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    # Set up the ViZDoom environment with your configuration
    config_path = "/../scenarios/basic.cfg"
    env = DummyVecEnv([lambda: ViZDoomEnv(config_path)])

    # Train the PPO model
    ppo_model = train_ppo_model(config_path, total_timesteps=10000)

    # Track the model's performance
    track_training_metrics(ppo_model, env, total_timesteps=10000)
