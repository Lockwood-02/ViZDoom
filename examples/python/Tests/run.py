from stable_baselines3 import PPO
from doom_env import DoomEnv

# Load environment
env = DoomEnv(config="basic.cfg", render=True)

# Load the trained model
model = PPO.load("ppo_doom")

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, _, done, _ = env.step(action)
    
env.close()
