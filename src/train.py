import gymnasium
import time

env = gymnasium.make("LunarLander-v3", render_mode = "human")  
state, info = env.reset()
for i in range(1, 300):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.02)
    if terminated or truncated:
        env.reset()
        