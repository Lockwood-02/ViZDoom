import gym
import numpy as np
import cv2
from vizdoom import DoomGame, Mode, ScreenResolution
from gym import spaces

class DoomEnv(gym.Env):
    def __init__(self, config="basic.cfg", render=True):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(render)
        self.game.init()
        
        self.action_space = spaces.Discrete(self.game.get_available_buttons_size())
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def step(self, action):
        reward = self.game.make_action(self._convert_action(action))
        state = self._get_state()
        done = self.game.is_episode_finished()
        return state, reward, done, {}

    def reset(self):
        print("Resetting environment...")

        self.game.new_episode()  # Start a new episode

        # Check if the game is running
        if self.game.is_episode_finished():
            print("Warning: Game episode ended immediately after reset!")

        # Wait until the game state is available
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        while self.game.get_state() is None and attempts < max_attempts:
            print(f"Waiting for game state... Attempt {attempts + 1}")
            self.game.make_action([0] * self.action_space.n)  # Take no action
            attempts += 1

        # If we exceeded max attempts, something is wrong
        if self.game.get_state() is None:
            print("Error: Game state is still None after multiple attempts!")
            return np.zeros((84, 84, 1), dtype=np.uint8)  # Return blank screen

        print("Game successfully started.")
        return self._get_state()



    # def _get_state(self):
    #     screen = self.game.get_state().screen_buffer
    #     gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    #     resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    #     return np.expand_dims(resized, axis=-1)

    def _get_state(self):
        game_state = self.game.get_state()
        
        if game_state is None:
            print("Warning: game.get_state() is None! Returning empty frame.")
            return np.zeros((84, 84, 1), dtype=np.uint8)  # Return a blank frame

        screen = game_state.screen_buffer

        # Debug print to confirm
        # print(f"Screen shape before transpose: {screen.shape}")

        # Fix the shape from (C, H, W) → (H, W, C)
        screen = np.transpose(screen, (1, 2, 0))

        # Convert to grayscale
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # Resize for stable-baselines3 compatibility
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        return np.expand_dims(resized, axis=-1)  # Add channel dimension for CNNs



    def _convert_action(self, action):
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Move forward, turn left, turn right
        return actions[action]

    def close(self):
        self.game.close()
