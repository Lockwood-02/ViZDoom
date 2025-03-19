import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file
df = pd.read_csv("training_results.csv")

# Plot Episode Reward Over Time
plt.figure(figsize=(10, 5))
plt.plot(df["Total Timesteps"], df["Episode Reward"], marker='o', linestyle='-', label="Episode Reward")
plt.xlabel("Total Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Progress: Reward Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot Episode Length Over Time
plt.figure(figsize=(10, 5))
plt.plot(df["Total Timesteps"], df["Episode Length"], marker='s', linestyle='-', color='red', label="Episode Length")
plt.xlabel("Total Timesteps")
plt.ylabel("Episode Length")
plt.title("Training Progress: Episode Length Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Display the DataFrame
print(df.head())  # Print first few rows of the DataFrame

# Save DataFrame as CSV for analysis
df.to_csv("training_results_processed.csv", index=False)
print("Training data saved as training_results_processed.csv")

