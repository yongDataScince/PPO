import os
import numpy as np
import torch as T
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim as optim
from tqdm import tqdm
from model.utils import PPOMemory

class FullconnectedLayer(nn.Module):
  def __init__(self, in_dims, out_dims, is_end=False) -> None:
      super().__init__()

      self.l = nn.Linear(in_dims, out_dims)
      self.act = nn.Softmax(dim=-1) if is_end else nn.ReLU()
  
  def forward(self, x):
    x = self.l(x)
    x = self.act(x)

    return x 
      

class ActorNetwork(nn.Module):
  def __init__(self, n_actions, input_dims, alpha, layer_dims=[256, 512], checkpoint_dir='tmp/ppo', name='actor') -> None:
      super().__init__()

      self.learning_rate = alpha
      self.n_actions = n_actions
      self.input_dims = input_dims
      self.layers_dims = [*input_dims, *layer_dims]
      self.checkpoint = os.path.join(checkpoint_dir, f'{name}_pp0')
      self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

      self._init_model()
      self._init_optimizer()
      
      self.to(self.device)
  
  def _init_model(self):
    layers = []
    in_dim = self.layers_dims[0]

    # Create hidden layers
    for dim in self.layers_dims[1:]:
      layers.append(
        FullconnectedLayer(in_dim, dim)
      )
      in_dim = dim
    
    # Add out layer
    layers.append(
      FullconnectedLayer(self.layers_dims[-1], self.n_actions, is_end=True)
    )

    self.model = nn.Sequential(*layers)

  def _init_optimizer(self):
    self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
  
  def forward(self, state):
    dist = self.model(state)
    dist = Categorical(dist)

    return dist

  def save_model(self):
    T.save(self.state_dict(), self.checkpoint)
  
  def load_model(self):
    self.load_state_dict(T.load(self.checkpoint))

model = ActorNetwork(10, input_dims=[16], alpha=1e-3)
print(model(T.randn((16))))

class CriticNetwork(nn.Module):
  def __init__(self, input_dims, beta=1e-3, layer_dims=[256, 128], checkpoint_dir='tmp/ppo', name='actor') -> None:
      super().__init__()
      self.learning_rate = beta
      self.input_dims = input_dims
      self.layers_dims = [*input_dims, *layer_dims]
      self.checkpoint = os.path.join(checkpoint_dir, f'{name}_pp0')
      self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

      self._init_model()
      self._init_optimizer()
      
      self.to(self.device)

  def _init_model(self):
    layers = []
    in_dim = self.layers_dims[0]

    # Create hidden layers
    for dim in self.layers_dims[1:]:
      layers.append(
        FullconnectedLayer(in_dim, dim)
      )
      in_dim = dim
    
    # Add out layer
    layers.append(
      nn.Linear(self.layers_dims[-1], 1)
    )

    self.model = nn.Sequential(*layers)
  
  def _init_optimizer(self):
    self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
  
  def forward(self, state):
    value = self.model(state)
    return value
  
  def save_model(self):
    T.save(self.state_dict(), self.checkpoint)
  
  def load_model(self):
    self.load_state_dict(T.load(self.checkpoint))
  
class PPO:
  def __init__(self, n_actions, input_dims, gamma=0.99, alpha=1e-4, beta=1e-4, policy_clip=0.2,\
                gae_lambda=0.95, batch_size=64, N=2048, n_epochs=10) -> None:
    self.gamma = gamma
    self.policy_clip = policy_clip
    self.gae_lambda = gae_lambda
    self.batch_size = batch_size
    self.alpha = alpha
    self.beta= beta
    self.n_epochs = n_epochs
    self.n_actions = n_actions
    self.N = N

    self.actor = ActorNetwork(n_actions, input_dims, alpha=alpha)
    self.critic = CriticNetwork(input_dims, beta=beta)
    self.memory = PPOMemory(batch_size)

  def remember(self, state, action, props, vals, rewards, done):
    self.memory.store_memory(state, action, props, vals, rewards, done)
  
  def save(self):
    print('__[SAVE_MODELS]__')
    self.actor.save_model()
    self.critic.save_model()
  
  def save(self):
    print('__[LOAD_MODELS]__')
    self.actor.load_model()
    self.critic.load_model()
  
  def choise_action(self, obs):
    state = T.tensor([obs], dtype=T.float).to(self.actor.device)

    dist = self.actor(state)
    value = self.critic(state)
    action = dist.sample()

    props = T.squeeze(dist.log)
    action = T.squeeze(action).item()
    value = T.squeeze(value).item()
  
    return action, props, value
  
  def learn(self):
    for epoch in range(self.n_epochs):
      state_arr, action_arr, old_probs_arr, vals_arr,\
          reward_arr, done_arr, batches = self.memory.generate_batches()
      values = vals_arr
      advantage = np.zeros(len(reward_arr), dtype=np.float32)

      for t in range(len(reward_arr)-1):
        discount = 1
        a_t = 0
        for k in range(t, len(reward_arr)-1):
          a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
              (1 - int(done_arr[k])) - values[k])
          discount *= self.gamma*self.gae_lambda
        
        advantage[t] = a_t
      advantage = T.tensor(advantage).to(self.actor.device)

      values = T.tensor(values).to(self.actor.device)

      # start learning
      loop = tqdm(batches)
      
      for batch in loop:
        states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
        old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
        actions = T.tensor(action_arr[batch]).to(self.actor.device)

        dist = self.actor(states)
        critic_value = self.critic(states)

        critic_value = T.squeeze(critic_value)

        new_probs = dist.log_prob(actions)
        prob_ratio = new_probs.exp() / old_probs.exp()

        weighted_probs = advantage[batch] * prob_ratio
        weighted_cliped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]

        actor_loss = -T.min(weighted_probs, weighted_cliped_probs).mean()

        returns = advantage[batch] + values[batch]
        critic_loss = (returns-critic_value) ** 2
        critic_loss = critic_loss.mean()

        total_loss = actor_loss + 0.5*critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        total_loss.backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

        loop.set_postfix(epoch=epoch, actor_loss=actor_loss, critic_loss=critic_loss, total_loss=total_loss)
        
    self.memory.clear_memory()
