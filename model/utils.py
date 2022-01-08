from torch.distributions.categorical import Categorical
import numpy as np

class PPOMemory:
  def __init__(self, batch_size):
    self.batch_size = batch_size

    self.states = []
    self.rewards = []
    self.probs = []
    self.vals = []
    self.actions = []
    self.dones = []
  
  def generate_batches(self):
    n_states = len(self.states)
    batch_start = np.arange(0, n_states, self.batch_size)
    indices = np.arange(n_states, dtype=np.int64)

    np.random.shuffle(indices)
    batches = [indices[i:i+self.batch_size] for i in batch_start]

    return np.array(self.states),\
           np.array(self.actions),\
           np.array(self.probs),\
           np.array(self.vals),\
           np.array(self.rewards),\
           np.array(self.dones),\
           batches
  
  def store_memory(self, state, action, prob, val, reward, done):
    self.dones.append(done)
    self.probs.append(prob)
    self.states.append(state)
    self.actions.append(action)
    self.vals.append(val)
    self.rewards.append(reward)
  
  def clear_memory(self):
    self.states = []
    self.rewards = []
    self.probs = []
    self.vals = []
    self.actions = []
    self.dones = []
