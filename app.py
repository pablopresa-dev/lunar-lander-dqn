import streamlit as st
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import time
import os
import shutil
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

# 4. Main simulation, video recording and playback loop
if start_simulation:
    # Clean up any existing video directories from previous runs
    video_dir = "./videos"
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
        
    # Initialize environment in rgb_array mode to enable pixel rendering for recording
    base_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Wrap the environment to automatically save the flight as an MP4 file
    env = RecordVideo(base_env, video_folder=video_dir, episode_trigger=lambda e: True)
    
    state, info = env.reset()
    done = False
    total_reward = 0
    
    # Create a clean visual spinner for the background calculation phase
    with st.spinner("🛸 Autopilot is computing the optimal flight path... Please wait."):
        live_reward_placeholder = st.empty()
    
        while not done:
            # Agent selects the optimal action greedily
            chosen_action = agent.act(state)
            
            # Step the environment (frames are captured automatically by the wrapper)
            next_state, reward, terminated, truncated, info = env.step(chosen_action)
            
            if terminated or truncated:
                done = True
                
            state = next_state
            total_reward += reward
            
            # Clean text-only update during computation (avoids video placeholder glitch)
            live_reward_placeholder.markdown(f"**Current Telemetry Reward:** `{total_reward:.2f}`")
            time.sleep(0.005) # Hyper-fast update for fluid backend processing
    # Close the environment and finalize the video encoding
    env.close()
    base_env.close()
    
    # 1. Clear the temporary text tracker once the real video is ready
    live_reward_placeholder.empty()    
    
    # 2. Get the video path and verify it exists
    video_file_path = os.path.join(video_dir, "rl-video-episode-0.mp4")
    if os.path.exists(video_file_path):
        # 3. Render the clean video player first so it starts playing immediately (60 FPS)
        st.video(video_file_path, autoplay=True)
        
        # 4. Display the final score banner at the bottom as a closing summary
        st.success(f"🎯 Flight complete! Final Evaluation Reward: {total_reward:.2f}")
    else:
        st.error("Error: Video file could not be generated.")