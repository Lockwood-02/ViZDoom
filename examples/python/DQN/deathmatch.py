from vizdoom import DoomGame
import numpy as np
import torch
import random
import time

# Hyperparameters
frame_repeat = 8  # Number of frames to repeat per action
episodes = 2000
learning_rate = 1e-4
gamma = 0.99
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.0001  # Rate at which epsilon decays
update_target_every = 1000  # Steps between target net updates
memory_capacity = 10000

def create_game():
    game = DoomGame()
    game.load_config("../scenarios/deathmatch.cfg")  # Adjust path as needed
    # Optional: set window visible or not
    # game.set_window_visible(True)
    game.init()
    return game

def preprocess_frame(frame):
    # Convert from (H, W, C) to a grayscale or other representation if desired
    # e.g., Torch expects (C, H, W); you might also downsample the image to speed up training
    frame = np.mean(frame, axis=0)  # convert to grayscale
    frame = frame / 255.0
    return frame

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Neural Network for DQN
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    def __init__(self, n_channels=1, n_actions=3):
        super(DQNetwork, self).__init__()
        # Example CNN architecture for raw Doom frames
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64*7*7, 256)  # Adjust dims based on input
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        # Random action
        return random.randrange(n_actions)
    else:
        # Greedy action (max Q-value)
        with torch.no_grad():
            q_values = policy_net(state)
            return q_values.argmax().item()

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Each transition is (state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.tensor(states, dtype=torch.float).unsqueeze(1)  # shape: [B, 1, H, W]
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1) # shape: [B, 1]
    rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)

    # Pass through policy network
    q_values = policy_net(states).gather(1, actions)

    # Compute the target Q-values
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    game = create_game()

    # Suppose we define possible actions as [turn_left, turn_right, attack]
    # Each action is a list of button states. Example (doom marine):
    actions_list = [
        [1, 0, 0],  # turn left
        [0, 1, 0],  # turn right
        [0, 0, 1]   # attack
    ]
    n_actions = len(actions_list)

    policy_net = DQNetwork(n_channels=1, n_actions=n_actions)
    target_net = DQNetwork(n_channels=1, n_actions=n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(memory_capacity)

    steps_done = 0
    epsilon = epsilon_start

    for episode in range(episodes):
        game.new_episode()
        frame = game.get_state().screen_buffer
        frame = preprocess_frame(frame)
        state = frame

        episode_reward = 0
        done = False

        while not game.is_episode_finished():
            # Convert state to torch
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # shape: [1, H, W]
            # shape for CNN: [1, channels=1, H, W]
            state_t = state_t.unsqueeze(1)

            action_idx = select_action(state_t, policy_net, epsilon, n_actions)
            reward = game.make_action(actions_list[action_idx], frame_repeat)

            done = game.is_episode_finished()
            next_frame = None
            if not done:
                next_frame = game.get_state().screen_buffer
                next_frame = preprocess_frame(next_frame)
            else:
                next_frame = np.zeros_like(state)

            memory.push(state, action_idx, reward, next_frame, float(done))
            episode_reward += reward
            state = next_frame

            # DQN optimization
            optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma)

            # Update target network
            if steps_done % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon - epsilon_decay)

        print(f"Episode {episode} finished. Total reward: {episode_reward}")
    
    game.close()

if __name__ == "__main__":
    main()
