import os
import gym
import numpy as np
import vizdoom as vzd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Define Doom environment wrapper
class DoomEnv(gym.Env):
    def __init__(self, config_file_path):
        super(DoomEnv, self).__init__()
        # Load the Doom game and configure
        self.game = vzd.DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_window_visible(False)  # Disable rendering
        self.game.init()
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(self.game.get_available_buttons_size())
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)

    def reset(self):
        self.game.new_episode()
        return self.game.get_state().screen_buffer

    def step(self, action):
        reward = self.game.make_action([action])
        done = self.game.is_episode_finished()
        obs = self.game.get_state().screen_buffer if not done else np.zeros((120, 160, 3), dtype=np.uint8)
        return obs, reward, done, {}

    def close(self):
        self.game.close()

# Initialize Doom environment
config_file_path = os.path.join('configs', 'basic.cfg')
doom_env = DummyVecEnv([lambda: DoomEnv(config_file_path)])

# Initialize SAC model
model = SAC('CnnPolicy', doom_env, verbose=1)

# Train the model
total_timesteps = 10000
episode_rewards = []
current_episode_reward = 0
obs = doom_env.reset()

for t in range(total_timesteps):
    action, _ = model.predict(obs)
    obs, reward, done, info = doom_env.step(action)
    current_episode_reward += reward

    if done:
        episode_rewards.append(current_episode_reward)
        current_episode_reward = 0
        obs = doom_env.reset()

        # Print out average reward for the last 10 episodes
        if len(episode_rewards) > 10:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {len(episode_rewards)} - Average Reward: {avg_reward:.2f}")

# Save the model
model.save("sac_vizdoom")

# Test the trained model
obs = doom_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = doom_env.step(action)

# Close the environment
doom_env.close()
