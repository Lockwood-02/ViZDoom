import pandas as pd
import time

# Evaluate the trained model
def evaluate_model(model, env, num_episodes=10):
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    avg_reward = sum(total_rewards) / num_episodes
    return avg_reward

# Track training metrics and save to CSV
def track_training_metrics(model, env, total_timesteps):
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    end_time = time.time()

    avg_reward = evaluate_model(model, env)
    training_time = end_time - start_time
    print(f"Training completed in {training_time} seconds.")
    print(f"Average reward: {avg_reward}")

    # Save results to CSV
    results_df = pd.DataFrame({
        'Algorithm': ['PPO'],
        'Avg_Reward': [avg_reward],
        'Training_Time': [training_time]
    })
    results_df.to_csv('results/ppo_performance_metrics.csv', index=False)
