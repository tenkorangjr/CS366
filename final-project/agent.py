import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQNetwork(nn.Module):
    """Deep Q-Network for processing Atari game frames"""
    
    def __init__(self, input_shape, n_actions):
        super(DQNetwork, self).__init__()
        
        # Convolutional layers for processing game frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size after convolutions
        conv_out_size = self._get_conv_out(input_shape)
        
        # Fully connected layers to output q-values for each possible action
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        """Calculate the output size of convolutional layers"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """Forward pass through the network"""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store an experience in the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class Mario:
    """DDQN Agent for playing DonkeyKong"""
    
    def __init__(self, state_shape, n_actions, learning_rate=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=32, target_update=1000):
        """
        Initialize the DDQN agent
        
        Args:
            state_shape: Shape of the input state (channels, height, width)
            n_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update: Steps between target network updates
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0
        
        # Initialize online and target networks
        self.online_net = DQNetwork(state_shape, n_actions).to(self.device)
        self.target_net = DQNetwork(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_losses = []
    
    def preprocess_state(self, state):
        """
        Preprocess the state for the network
        
        Args:
            state: Raw state from environment
            
        Returns:
            Preprocessed state tensor
        """
        # Convert to grayscale if RGB
        if len(state.shape) == 3 and state.shape[2] == 3:
            state = np.dot(state[..., :3], [0.299, 0.587, 0.114])
        
        # Normalize to [0, 1]
        state = state.astype(np.float32) / 255.0
        
        # Add channel dimension if needed
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def act(self, state, training=True):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action
        """
        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        # Exploitation: best action from Q-network
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            q_values = self.online_net(state_tensor)
            return q_values.argmax(1).item()
    
    def cache(self, state, action, reward, next_state, done):
        """
        Store an experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def recall(self):
        """
        Sample a batch of experiences from replay buffer
        
        Returns:
            Batch of experiences or None if buffer is too small
        """
        if len(self.memory) < self.batch_size:
            return None
        
        return self.memory.sample(self.batch_size)
    
    def learn(self):
        """
        Train the network using DDQN algorithm
        
        Returns:
            Loss value or None if not enough samples
        """
        # Check if we have enough samples
        batch = self.recall()
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        # Preprocess states
        state_batch = torch.cat([self.preprocess_state(s) for s in states])
        next_state_batch = torch.cat([self.preprocess_state(s) for s in next_states])
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.online_net(state_batch).gather(1, action_batch)
        
        # DDQN: Use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using online network
            next_actions = self.online_net(next_state_batch).argmax(1, keepdim=True)
            # Evaluate actions using target network
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()
            # Calculate target Q values
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store loss for monitoring
        self.training_losses.append(loss.item())
        
        return loss.item()
    
    def save(self, path):
        """Save the model checkpoint"""
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Model loaded from {path}")