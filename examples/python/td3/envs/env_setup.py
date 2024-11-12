import gym
import vizdoom as vzd
from gym import spaces
import numpy as np
import cv2  # OpenCV for resizing

class VizDoomEnv(gym.Env):
    def __init__(self):
        super(VizDoomEnv, self).__init__()
        self.game = vzd.DoomGame()
        
        # Load basic scenario
        self.game.load_config("../scenarios/basic.cfg")  # Ensure correct path
        self.game.init()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.game.get_available_buttons_size())
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(120, 160, 3),  # Height: 120, Width: 160, Channels: 3 (RGB)
            dtype=np.uint8
        )
        
        # Define the available actions
        self.actions = [list(action) for action in np.eye(self.action_space.n, dtype=int)]

    def reset(self):
        self.game.new_episode()
        # Get the screen buffer and resize it
        obs = self.game.get_state().screen_buffer
        if obs is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        
        # Resize to (120, 160) and ensure it's transposed to HWC format
        obs = cv2.resize(obs.transpose(1, 2, 0), (160, 120), interpolation=cv2.INTER_LINEAR)
        
        return obs

    def step(self, action):
        # TD3 outputs continuous actions, we need to map them to discrete actions
        # Assuming the actions are output in a continuous range, we can round or discretize them.
        action = int(np.round(action))  # Convert continuous action to nearest integer

        # Ensure action is within the valid range of discrete actions
        action = np.clip(action, 0, len(self.actions) - 1)

        # Execute the chosen action in the environment
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()

        if not done:
            obs = self.game.get_state().screen_buffer
            obs = cv2.resize(obs.transpose(1, 2, 0), (160, 120), interpolation=cv2.INTER_LINEAR)
        else:
            obs = self.reset()

        return obs, reward, done, {}


    def get_observation(self):
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        
        # Get the screen buffer and resize it
        screen_buffer = state.screen_buffer
        resized_buffer = cv2.resize(screen_buffer.transpose(1, 2, 0), (160, 120), interpolation=cv2.INTER_LINEAR)
        
        return resized_buffer

    def close(self):
        self.game.close()
