import gym
import numpy as np
import cv2
from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat
from gym import spaces

class DoomEnv(gym.Env):
    def __init__(self, config="deathmatch.cfg", render=False):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(render)
        
        # Force RGB format for screen buffer
        self.game.set_screen_format(ScreenFormat.RGB24)  # Ensures (3, H, W) format
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)  # Higher resolution for better learning
        
        self.game.init()

        # Define action space (Move + Turn + Shoot)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)


    def step(self, action):
        reward = self.game.make_action(self._convert_action(action))
        state = self._get_state()
        done = self.game.is_episode_finished()
        return state, reward, done, {}

    def reset(self):
        self.game.new_episode()
        return self._get_state()

    def _get_state(self):
        game_state = self.game.get_state()
        if game_state is None:
            print("Warning: game.get_state() is None! Returning empty frame.")
            return np.zeros((84, 84, 1), dtype=np.uint8)  # Return a blank frame

        screen = game_state.screen_buffer  # Shape (240, 320, 3)

        # Convert to grayscale (H, W, C) -> (H, W)
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # Resize to 84x84 for Stable-Baselines3 compatibility
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # Expand dimensions to (84, 84, 1) for CNN input
        return np.expand_dims(resized, axis=-1)



    def _convert_action(self, action):
        actions = [
            [1, 0, 0, 0, 0, 0],  # Move Forward
            [0, 1, 0, 0, 0, 0],  # Move Backward
            [0, 0, 1, 0, 0, 0],  # Turn Left
            [0, 0, 0, 1, 0, 0],  # Turn Right
            [0, 0, 0, 0, 1, 0],  # Strafe Left
            [0, 0, 0, 0, 0, 1],  # Shoot
        ]
        return actions[action]

    def close(self):
        self.game.close()
