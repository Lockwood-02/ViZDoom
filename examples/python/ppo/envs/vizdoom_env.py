import gym
import numpy as np
import vizdoom as vzd

# Create a ViZDoom environment
def create_vizdoom_env(config_path="config/basic.cfg"):
    game = vzd.DoomGame()
    game.load_config(config_path)

    # Enable available buttons (e.g., move left, move right, shoot)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)

    game.init()
    return game

class ViZDoomEnv(gym.Env):
    def __init__(self, config_path="config/basic.cfg"):
        super(ViZDoomEnv, self).__init__()
        self.game = create_vizdoom_env(config_path)

        # Define all available actions (e.g., move left, right, shoot)
        self.action_space = gym.spaces.Discrete(3)  # 3 discrete actions: move left, move right, shoot

        self.actions = [
            [1, 0, 0],  # move left
            [0, 1, 0],  # move right
            [0, 0, 1],  # shoot
        ]

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.game.get_screen_height(),
                   self.game.get_screen_width(),
                   self.game.get_screen_channels()),
            dtype=np.uint8
        )

    def reset(self):
        self.game.new_episode()
        obs = self.game.get_state().screen_buffer
        # Transpose the observation to HWC format (height, width, channels)
        obs = obs.transpose(1, 2, 0)
        return obs

    def step(self, action):
        # Execute the action chosen by the agent (mapped from self.actions)
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()

        if not done:
            obs = self.game.get_state().screen_buffer
            obs = obs.transpose(1, 2, 0)  # Transpose to HWC format
        else:
            obs = self.reset()

        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.game.close()
