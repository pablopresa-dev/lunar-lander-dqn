import gymnasium as gym
import numpy as np
import torch
import time
from src.agent import DQNAgent
import os
import csv
import matplotlib.pyplot as plt

def train():
    """
    Main training loop that connects the Lunar Lander environment
    with the DQNAgent to train it over multiple episodes.
    """
    # 1. Initialize environment (render_mode = None trains at warp speed without graphics)
    # Change to "human" later if wanted to watch the agent play
    env = gym.make("LunarLander-v3", render_mode = None)  
    
    # 2. Extract environment geometry dimensions
    state_size = env.observation_space.shape[0] # Should be 8 parameters
    action_size = env.action_space.n            # Should be 4 discrete actions
    
    # 3. Instantiate the expert DQN Agent with a capacity of 100,000 experiences
    agent = DQNAgent(state_size=state_size, action_size=action_size, memory_size=100000)
    
    # 4. Define training hyperparameters
    n_episodes = 2000   # Maximum number of games to play
    batch_size = 64     # Size of the memory mini-batches for learning
    
    # Track evaluation metrics throughout training episodes
    scores_history = [] # List to store total rewards per episode
    
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
        
        # Append reward to history for post-training metrics execution
        scores_history.append(total_reward) # Store the final reward of the episode
        
        # 6. Decay epsilon to reduce exploration over time
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
    # 7. Save the primary network's weights to disk
    torch.save(agent.brain.state_dict(), "lunar_lander_dqn.pth")
    print("Training complete! Optimal weights saved successfully to  'lunar_lander_dqn.pth'.")
    
    # 8. Export analytics files automatically
    save_training_metrics(scores_history) # Call the metric logging function here
        
# After training completes, assume 'scores_history' is your list of rewards per episode
def save_training_metrics(scores, output_dir="metrics"):
    """
    Logs training scores to a CSV file and exports a professional evaluation plot.
    """
    # Create metrics directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "training_scores.csv")
    plot_path = os.path.join(output_dir, "learning_curve.png")
    
    # 1. Save evaluation metrics to a clean CSV file
    print(f"Saving training metrics to {csv_path}...")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])
        for episode, reward in enumerate(scores, start=1):
            writer.writerow([episode, reward])
            
    # 2. Generate a professional learning curve plot
    print(f"Generating learning curve plot at {plot_path}...")
    plt.style.use("seaborn-v0_8-whitegrid")  # Clean corporate plotting style
    plt.figure(figsize=(10, 5))
    
    # Plot raw episode rewards with lower alpha for visual clarity
    plt.plot(scores, color="#1f77b4", alpha=0.3, label="Raw Episode Reward")
    
    # Calculate and plot a moving average (e.g., window of 50 episodes) to show convergence
    if len(scores) >= 50:
        import numpy as np
        moving_avg = np.convolve(scores, np.ones(50)/50, mode="valid")
        plt.plot(range(50, len(scores) + 1), moving_avg, color="#d62728", linewidth=2, label="50-Episode Moving Avg")
        
    plt.title("DQN Autonomous Pilot - LunarLander-v3 Training Convergence", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Total Reward Score", fontsize=12)
    plt.axhline(y=200, color="#2ca02c", linestyle="--", linewidth=1.5, label="Environment Solved Threshold (+200)")
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    # Save chart with high DPI resolution for publication quality
    plt.savefig(plot_path, dpi=300)
    plt.close()
        
if __name__ == "__main__":
    train()