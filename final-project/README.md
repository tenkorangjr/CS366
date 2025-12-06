# Mario Bros DDQN Agent

This project implements a Double Deep Q-Network (DDQN) agent to automate gameplay for the Atari game Mario Bros using reinforcement learning.

## Overview

The implementation includes:
- **DDQN Algorithm**: Uses two networks (online and target) to reduce overestimation bias
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Convolutional Neural Network**: Processes raw game frames
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation

## Project Structure

```
.
├── agent.py          # DDQN agent implementation
├── main.py           # Training and evaluation scripts
├── config.py         # Hyperparameters and configuration
├── models/           # Saved model checkpoints (created during training)
└── README.md         # This file
```

## Key Components

### 1. DQNetwork (`agent.py`)
- Convolutional neural network for processing Atari frames
- 3 convolutional layers followed by 2 fully connected layers
- Outputs Q-values for each possible action

### 2. ReplayBuffer (`agent.py`)
- Stores experiences (state, action, reward, next_state, done)
- Samples random batches for training
- Configurable capacity (default: 100,000)

### 3. Mario Agent (`agent.py`)
- Main DDQN agent class
- Methods:
  - `act()`: Select actions using epsilon-greedy policy
  - `cache()`: Store experiences in replay buffer
  - `recall()`: Sample batch from replay buffer
  - `learn()`: Train using DDQN algorithm
  - `save()`/`load()`: Model persistence

## Installation

1. Install dependencies:
```bash
pip install torch numpy gymnasium ale-py
```

2. Install ROMs for Atari games:
```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```

## Usage

### Training the Agent

Train for 1000 episodes (default):
```bash
python main.py --mode train
```

Train with custom settings:
```bash
python main.py --mode train --episodes 2000 --save_interval 50 --render
```

### Playing with Trained Agent

Play using a specific checkpoint:
```bash
python main.py --mode play --model models/mario_episode_500.pth --episodes 10 --render
```

### Command Line Arguments

- `--mode`: Choose 'train' or 'play' (default: train)
- `--episodes`: Number of episodes (default: 1000)
- `--model`: Path to saved model for play mode (default: models/mario_final.pth)
- `--render`: Render the game during execution
- `--save_interval`: Save model every N episodes (default: 100)

## Hyperparameters

Key hyperparameters can be adjusted in `config.py`:

- **Learning Rate**: 1e-4
- **Discount Factor (γ)**: 0.99
- **Batch Size**: 32
- **Replay Buffer Size**: 100,000
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.01
- **Epsilon Decay**: 0.995
- **Target Network Update Frequency**: 1000 steps

## Algorithm Details

### DDQN (Double Deep Q-Network)

The DDQN algorithm addresses the overestimation bias in standard DQN:

1. **Action Selection**: Online network selects the best action for the next state
2. **Action Evaluation**: Target network evaluates the selected action
3. **Update Rule**: 
   ```
   Q_target = reward + γ * Q_target(next_state, argmax_a Q_online(next_state, a))
   ```

### Training Process

1. **Initialize**: Create online and target networks with same weights
2. **Collect Experience**: 
   - Select action using epsilon-greedy policy
   - Execute action in environment
   - Store (state, action, reward, next_state, done) in replay buffer
3. **Learn**:
   - Sample random batch from replay buffer
   - Compute loss using DDQN target
   - Update online network via backpropagation
   - Periodically copy weights to target network
4. **Decay Epsilon**: Gradually reduce exploration rate

### State Preprocessing

- Convert RGB frames to grayscale
- Normalize pixel values to [0, 1]
- Reshape to (1, height, width) for CNN input

## Expected Results

- **Initial Episodes**: Random exploration, low rewards
- **After ~100-200 Episodes**: Agent starts learning basic strategies
- **After ~500-1000 Episodes**: Noticeable improvement in gameplay

## References

- [Human-level control through deep reinforcement learning (DQN)](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (DDQN)](https://arxiv.org/abs/1509.06461)
