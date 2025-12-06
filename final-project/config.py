"""
Configuration file for DDQN Mario Bros Agent
"""

# Network Architecture
CONV_LAYERS = [
    {'out_channels': 32, 'kernel_size': 8, 'stride': 4},
    {'out_channels': 64, 'kernel_size': 4, 'stride': 2},
    {'out_channels': 64, 'kernel_size': 3, 'stride': 1}
]

FC_HIDDEN_SIZE = 512

# Training Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 32
BUFFER_SIZE = 100000
TARGET_UPDATE = 1000  # Update target network every N steps

# Exploration Parameters
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Training Configuration
NUM_EPISODES = 1000
SAVE_INTERVAL = 100
WARMUP_STEPS = 1000  # Steps before training starts

# Environment Configuration
ENV_NAME = "ALE/MarioBros-v5"
RENDER_TRAINING = False
RENDER_PLAYING = True

# File Paths
MODEL_DIR = "models"
CHECKPOINT_PREFIX = "mario_episode"
FINAL_MODEL_NAME = "mario_final.pth"

# Logging
LOG_INTERVAL = 10  # Print stats every N episodes
MOVING_AVERAGE_WINDOW = 100  # Window for computing average rewards