import gymnasium as gym
import torch
import time
from src.agent import DQNAgent

def test():
    """
    Evaluation script to validate the performance of the trained DQN policy.
    Loads pre-trained neural network weights and executes deterministic actions.
    """
    # 1. Initialize environment in human rendering mode for visual inspection
    env = gym.make("LunarLander-v3", render_mode="human")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 2. Instantiate agent and load the serialized state dictionary (trained weights)
    agent = DQNAgent(state_size=state_size, action_size=action_size, memory_size=1000)
    agent.brain.load_state_dict(torch.load("lunar_lander_dqn.pth"))
    
    # 3. Enforce deterministic policy by disabling exploration (Epsilon = 0.0)
    agent.epsilon = 0.0
    
    print("Pre-trained model weights loaded successfully. Starting evaluation...")
    
    n_test_episodes = 5
    
    for episode in range(1, n_test_episodes + 1):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action greedily based on the learned Q-function
            chosen_action = agent.act(state)
            
            next_state, reward, terminated, truncated, info = env.step(chosen_action)
            
            if terminated or truncated:
                done = True
            
            state = next_state
            total_reward += reward
            
            # Frame rate regulation for human-readable evaluation pacing
            time.sleep(0.01)
            
        print(f"Test Episode {episode} - Evaluation Reward: {total_reward:.2f}")
        time.sleep(1.0)
        
    env.close()

if __name__ == "__main__":
    test()