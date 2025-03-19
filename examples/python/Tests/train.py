import csv
import gym
from stable_baselines3 import PPO
from doom_env import DoomEnv

# Initialize environment
env = DoomEnv(config="basic.cfg", render=True)

# Load PPO model
model = PPO("CnnPolicy", env, verbose=1) # type: ignore

# Prepare CSV file for logging
csv_filename = "training_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Total Timesteps", "Episode Length", "Episode Reward"])

# Training with logging
num_iterations = 10  # Change as needed
for i in range(num_iterations):
    model.learn(total_timesteps=2048)  # Train for 2048 timesteps
    ep_len = env.game.get_episode_time()  # Get episode length
    ep_rew = env.game.get_total_reward()  # Get total reward

    # Log the data
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([i + 1, (i + 1) * 2048, ep_len, ep_rew])

    print(f"Iteration {i+1}: Episode Length = {ep_len}, Episode Reward = {ep_rew}")

# Save trained model
model.save("ppo_doom")

env.close()
