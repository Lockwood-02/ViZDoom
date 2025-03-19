import numpy as np
import random
import time
import os

import vizdoom as vzd

def main():
    game = vzd.DoomGame()
    
    # Load the deathmatch scenario config
    game.load_config("../scenarios/deathmatch.cfg")

    # game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "deathmatch.wad"))
    
    # Initialize the game
    game.init()

    # Define possible actions
    # Adjust these based on what you'd like to allow in your scenario
    actions = [
        [1, 0, 0],  # Example: move left
        [0, 1, 0],  # move right
        [0, 0, 1]   # attack
    ]
    
    # Run a certain number of episodes
    episodes = 5
    for i in range(episodes):
        print("Episode #", i + 1)
        game.new_episode()
        
        while not game.is_episode_finished():
            # Get the current state
            state = game.get_state()
            
            # (Optional) Use the screen buffer, depth buffer, labels, etc.
            screen_buf = state.screen_buffer
            
            # Select a random action
            action = random.choice(actions)
            
            # Make the action
            reward = game.make_action(action)
            
            # (Optional) Sleep so you can see the action taking place
            time.sleep(0.02)
        
        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")

    game.close()

if __name__ == "__main__":
    main()
