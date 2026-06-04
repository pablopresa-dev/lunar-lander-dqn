import collections
import random

class ReplayBuffer:
    def __init__(self, memorySize):
        self.memorySize = memorySize
        self.buffer = collections.deque(maxlen = memorySize)
    
    def add(self, state, action, reward, nextState, hasDied):
        self.buffer.append((state, action, reward, nextState, hasDied))
        
    def sample(self, batchSize):
        return random.sample(self.buffer, batchSize)
    
    def __len__(self):
        return len(self.buffer)
    
# --- INTEGRITY AND UNIT TESTING ---
if __name__ == "__main__":
    # 1. Initialize a test buffer with a small maximum capacity of 10 experiences
    test_memory = ReplayBuffer(memorySize=10)
    
    # 2. Add dummy experiences to verify the 'add' method functionality
    # Format: (state, action, reward, next_state, has_died)
    test_memory.add([0.1, -0.2], 1, 100, [0.15, -0.1], False)
    test_memory.add([0.3, -0.5], 0, -50, [0.4, -0.3], True)
    
    # 3. Verify that __len__ correctly tracks the number of stored experiences
    print(f"Current buffer size: {len(test_memory)} experiences.")
    
    # 4. Test the 'sample' method by retrieving a random mini-batch
    sample_batch = test_memory.sample(batchSize=1)
    print(f"Randomly retrieved sample: {sample_batch}")