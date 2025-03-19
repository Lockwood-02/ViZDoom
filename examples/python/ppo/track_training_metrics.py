# File path: utils/track_metrics.py

import time

def track_training_metrics(train_function):
    metrics = []
    start_time = time.time()
    episode = 0

    for episode_rewards, episode_loss in train_function():
        episode += 1
        metrics.append({
            "episode": episode,
            "reward": episode_rewards,
            "loss": episode_loss,
            "time_elapsed": time.time() - start_time
        })
    return metrics
