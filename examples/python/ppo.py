import cv2
from stable_baselines3 import ppo
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.policies import BasePolicy, ActorCriticCnnPolicy

import common
# from stable_baselines3.common.env_util import make_vec_env

if __name__ == '__main__':
    # Create training and evaluation environments.
    env_args = {
        'scenario': 'basic',
        'frame_skip': 4,
        'frame_processor': lambda frame: cv2.resize(
            frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    }

    # Create training and evaluation environments.
    training_env = common.create_vec_env(**env_args)
    eval_env = common.create_vec_env(**env_args)

    # Create an agent.
    agent = ppo.PPO(policy=ActorCriticCnnPolicy,
                    env=training_env,
                    learning_rate=1e-4,
                    tensorboard_log='logs/tensorboard')

    # Add an evaluation callback that will save the best model when new records are reached.
    evaluation_callback = EvalCallback(eval_env,
                                                 n_eval_episodes=10,
                                                 eval_freq=2500,
                                                 best_model_save_path='logs/models/basic')

    # Play!
    agent.learn(total_timesteps=25000, tb_log_name='ppo_basic', callback=evaluation_callback)

    training_env.close()
    eval_env.close()