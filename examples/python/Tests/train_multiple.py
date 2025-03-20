import csv
import gym
import argparse
from stable_baselines3 import PPO, DQN, A2C
from doom_env import DoomEnv

# Parse arguments
parser = argparse.ArgumentParser(description="Train RL models on ViZDoom")
parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN", "A2C"],
                    help="Choose an RL algorithm: PPO, DQN, A2C")
parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps to train")  # Increased default
args = parser.parse_args()

# Initialize environment
env = DoomEnv(config="basic.cfg", render=False)

# Select model
if args.algo == "PPO":
    model = PPO("CnnPolicy", env, verbose=1)
elif args.algo == "DQN":
    model = DQN("CnnPolicy", env, verbose=1)
elif args.algo == "A2C":
    model = A2C("CnnPolicy", env, verbose=1, ent_coef=0.01)

print(f"Training {args.algo} for {args.timesteps} timesteps...")

# CSV Logging
csv_filename = f"training_results_{args.algo}.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Total Timesteps", "Episode Length", "Episode Reward"])

# Ensure multiple iterations
num_iterations = max(args.timesteps // 2048, 10)  # Ensure at least 10 iterations
for i in range(num_iterations):
    model.learn(total_timesteps=2048)
    
    # Get stats
    ep_len = env.game.get_episode_time()
    ep_rew = env.game.get_total_reward()

    # Log data
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([i + 1, (i + 1) * 2048, ep_len, ep_rew])

    print(f"Iteration {i+1}: Total Timesteps = {(i+1) * 2048}, Episode Length = {ep_len}, Episode Reward = {ep_rew}")

# Save model
model.save(f"{args.algo}_doom")

env.close()
print(f"Training complete! Model saved as {args.algo}_doom.zip")
