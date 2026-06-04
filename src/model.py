import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Deep Q-Network (DQN) classifier for estimating action-value functions.
    Matches the architecture needed for the Gymnasium LunarLander environment.
    """
    
    def __init__(self, state_size, action_size):
        """
        Initialize parameters and build the neural network layers.
        
        Inputs:
            state_size (int): Dimension of each state observation (typically 8)
            action_size (int): Number of available discrete actions (typically 4)
        """
        super(QNetwork, self).__init__()
        
        # Network Architecture: Input(8) -> Hidden 1 (64) -> Hidden 2 (64) -> Output (4)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """
        Execute the forward pass to map states to action-values (Q-values).
        
        Inputs:
            state (Tensor): Current environmental observation matrix
        Returns:
            Tensor: Predicted score for each potential action
        """
        # Apply Rectified Linear Unit (ReLU) activation functions to hidden layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Output layer returns raw scores (Q-values) without activation
        return self.fc3(x)
    
    
# --- ARCHITECTURE VERIFICATION TEST ---
if __name__ == "__main__":
    # Instantiate the network with standard LunarLander dimensions
    model = QNetwork(state_size=8, action_size=4)
    print("DQN Architecture successfully initialized:")
    print(model)