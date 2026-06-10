import gymnasium as np
import numpy as np
import torch
import time
from src.agent import DQNAgent

def train():
    """
    Main training loop that connects the Lunar Lander environment
    with the DQNAgent to train it over multiple episodes.
    """
    # 1. Initialize environment (render_mode = None trains at warp speed without graphics)
    # Change to "human" later if wanted to watch the agent play
    env = gymnasium.make("LunarLander-v3", render_mode = None)  
    
    # 2. Extract environment geometry dimensions
    state_size = env.observation_space.shape[0] # Should be 8 parameters
    action_size = env.action_space.n            # Should be 4 discrete actions
    
    # 3. Instantiate the expert DQN Agent with a capacity of 100,000 experiences
    agent = DQNAgent(state_size=state_size, action_size=action_size, memory_size=100000)
    
    # 4. Define training hyperparameters
    n_episodes = 2000   # Maximum number of games to play
    batch_size = 64     # Size of the memory mini-batches for learning
    
    print("Initializing training... The Lunar Lander is ready on the launchpad.")
    
    # Main loop over training episodes
    for episode in range(1, n_episodes + 1):
        # Reset the environment at the start of each game to get the initial state
        state, info = env.reset()
        total_reward = 0
        done = False  
        
        # Loop through environment timesteps until the episode terminates
        while not done:
            # Select action using the Epsilon-Greedy policy defined in agent.py
            chosen_action = agent.act(state)
            
            # Execute action in the environment to obtain the next transition tuple
            next_state, reward, terminated, truncated, info = env.step(chosen_action)
            
            # An episode ends if the agent reaches a terminal state or times out
            if terminated or truncated:
                done = True
                
            # Store the experience transition into the agent's memory buffer
            agent.replay_buffer.add(state, chosen_action, reward, next_state, done)
            
            # Transition to the next state and accumulate step reward
            state = next_state
            total_reward += reward
            
            # Trigger the learning process once the buffer has enough samples
            if len(agent.replay_buffer) > batch_size:
                agent.learn(batch_size)
                
        # 5. Print progress at the end of each training episode
        print(f"Episode {episode} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
        
        # 6. Decay epsilon to reduce exploration over time
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
if __name__ == "__main__":
    train()