import streamlit as st
import gymnasium as gym
import torch
import numpy as np
import time
from src.agent import DQNAgent

# 1. Page layout and styling configuration
st.set_page_config(page_title="DQN Lunar Lander Visualizer", page_icon="🚀", layout="centered")

st.title("🚀 Deep Q-Network: Lunar Lander Pilot")
st.write("This interactive web application demonstrates a reinforcement learning agent trained using a DQN architecture to safely land on the Moon.")

# 2. Optimized model loading cached by Streamlit
@st.cache_resource
def load_trained_agent():
    # Initialize a temporary environment to extract dimensions
    temp_env = gym.make("LunarLander-v3")
    state_size = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.n
    temp_env.close()
    
    # Instantiate the agent and load the serialized optimal weights
    agent = DQNAgent(state_size=state_size, action_size=action_size, memory_size=1000)
    agent.brain.load_state_dict(torch.load("lunar_lander_dqn.pth", map_location=torch.device('cpu')))
    agent.epsilon = 0.0 # Enforce fully deterministic evaluation policy
    return agent

try:
    agent = load_trained_agent()
    st.success("Trained model weights ('lunar_lander_dqn.pth') loaded successfully!")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    
# 3. User Interface Control Panel
st.subheader("Simulation Control Panel")
start_simulation = st.button("Launch Evaluation Flight")

# 4. Main simulation and web rendering loop
if start_simulation:
    # Initialize environment in rgb_array mode to capture frames as matrices
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    state, info = env.reset()
    done = False
    total_reward = 0
    
    # Placeholders for dynamic rendering
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    while not done:
        # Agent selects the optimal action greedily
        chosen_action = agent.act(state)
        
        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(chosen_action)
        
        if terminated or truncated:
            done = True
            
        state = next_state
        total_reward += reward
        
        # Capture current simulation frame matrix
        current_frame = env.render()
        
        # Render frame matrix directly onto the web interface
        frame_placeholder.image(current_frame, channels="RGB", use_container_width=True)
        status_placeholder.metric(label="Current Accumulated Reward", value=f"{total_reward:.2f}")
        
        # Frame pacing delay
        time.sleep(0.005)
        
    env.close()
    st.balloons()
    st.success(f"Flight complete! Final Evaluation Reward: {total_reward:.2f}")