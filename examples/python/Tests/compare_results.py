import pandas as pd
import matplotlib.pyplot as plt

# Load results from all algorithms
ppo_df = pd.read_csv("training_results_PPO.csv")
dqn_df = pd.read_csv("training_results_DQN.csv")
a2c_df = pd.read_csv("training_results_A2C.csv")

plt.figure(figsize=(10, 5))

# Plot Episode Reward
plt.plot(ppo_df["Total Timesteps"], ppo_df["Episode Reward"], marker='o', linestyle='-', label="PPO")
plt.plot(dqn_df["Total Timesteps"], dqn_df["Episode Reward"], marker='s', linestyle='-', label="DQN", color='red')
plt.plot(a2c_df["Total Timesteps"], a2c_df["Episode Reward"], marker='^', linestyle='-', label="A2C", color='green')

plt.xlabel("Total Timesteps")
plt.ylabel("Episode Reward")
plt.title("Comparison of RL Algorithms in ViZDoom")
plt.legend()
plt.grid(True)
plt.show()
