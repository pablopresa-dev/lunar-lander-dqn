import collections
import random
from collections import namedtuple

# Define a structured blueprint for individual experience records
# This allows us to access elements cleanly via properties (e.g., experience.state)
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
class ReplayBuffer:
    """
    Fixed-size memory buffer to store experience tuples so the DQN agent
    can sample and learn from past interactions independently.
    """
    def __init__(self, memory_size):
        """
        Initialize the experience replay memory storage.
        """
        self.memory_size = memory_size
        self.buffer = collections.deque(maxlen = memory_size)
    
    def add(self, state, action, reward, next_state, done):
        """
        Wrap raw parameters into an Experience namedtuple and append it to the buffer.
        """
        e = Experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        
    def sample(self, batch_size):
        """
        Randomly sample a collection of experiences for training.
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """
        Return the current number of experiences stored in memory.
        """
        return len(self.buffer)
    
# --- INTEGRITY AND UNIT TESTING ---
if __name__ == "__main__":
    # 1. Initialize a test buffer with a small maximum capacity of 10 experiences
    test_memory = ReplayBuffer(memory_size=10)
    
    # 2. Add dummy experiences to verify the 'add' method functionality
    test_memory.add([0.1, -0.2], 1, 100, [0.15, -0.1], False)
    test_memory.add([0.3, -0.5], 0, -50, [0.4, -0.3], True)
    
    # 3. Verify that __len__ correctly tracks the number of stored experiences
    print(f"Current buffer size: {len(test_memory)} experiences.")
    
    # 4. Test the 'sample' method by retrieving a random mini-batch
    sample_batch = test_memory.sample(batch_size=1)
    print(f"Randomly retrieved sample: {sample_batch}")