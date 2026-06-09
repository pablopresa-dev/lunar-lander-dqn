import torch
import random
import numpy as np
import torch.optim as optim
from src.model import QNetwork
from src.buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent that interacts with the environment,
    stores experiences in memory, and optimizes its neural network
    """
    
    def __init__(self, state_size, action_size, memory_size):
        """
        Initialize the agent's brain, memory buffer and mathematical optimizer.
        
        Inputs:
            state_size (int): Dimension of each environmental state (8)
            action_size (int): Number of available discrete actions (4)
            memory_size (int): Maximum capacity of the experience replay buffer
        """
        # Initialize the primary neural network (The Brain)
        self.brain = QNetwork(state_size, action_size)
        
        # Initialize the replay memory structure to store past experiences
        self.ReplayBuffer = ReplayBuffer(memory_size)
        
        # Initialize the Adam optimizer to adjust network weights based on learning errors
        # 'lr' represents the learning rate (how fast the agent modifies its weights)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0005)
        
        self.epsilon, self.epsilon_decay, self.epsilon_min = 1.0, 0.995, 0.01
        
    def act(self, state):
        """
        Select an action using the Epsilon-Greedy policy (Exploration Vs Exploitation).
        
        Inputs:
            state (ndarray): The current 8-dimensional observation from the environment
        Returns:
            int: The chosen action index (0 to 3)
        """
        # Roll the dice to determine if the agent will explore
        random_num = random.random()
        
        if random_num < self.epsilon:
            # Exploration: choose a random action to discover the environment
            return random.choice([0, 1, 2, 3])
        
        else:
            # Exploitation: convert state to PyTorch Tensor and predict best action
            # 1. Transform the numpy state into a float tensor with a batch dimension
            state = torch.from_numpy(state).float().unsqueeze(0)
            
            # 2. Set network to evaluation mode and disable gradient tracking for speed
            self.brain.eval()
            with torch.no_grad():
                # Pass the prepared state through the neural network
                    q_values = self.brain.forward(state)
            self.brain.train() # Return network to training mode
            
            # 3. Return the raw integer index of the maximum Q-value.
            return torch.argmax(q_values).item()
        
    def learn(self, batch_size):
        
        """
        Sample a mini-batch of experiences from memory and optimize the brain's weights
        by minimizing the difference between predicted and target Q-values.
        
        Inputs:
            batch_size (int): Number of experiences to sample and process in this step
        """
        # 1. Extract a random mini-batch of structured experiences from the replay buffer
        experiences = self.replay_buffer.sample(batch_size)
        
        # 2. Extract and stack current states into a 2D float tensor (shape: batch_size x 8)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        
        # 3. Extract actions and convert them into a 2D long (integer) tensor (shape: batch_size x 1)
        # We use .long() because action indices (0, 1, 2, 3) are used as matrix indexes in PyTorch
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        
        # 4. Extract rewards and convert them into a 2D float tensor (shape: batch_size x 1)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        
        # 5. Extract next states and convert them into a 2D float tensor (shape: batch_size x 8)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float()
        
        # 6. Extract termination flags (dones) and convert True/False to 1.0/0.0 float tensor (shape: batch_size x 1)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float()