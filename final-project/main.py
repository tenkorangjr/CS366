import ale_py
import gymnasium as gym
import numpy as np
from agent import Mario
import torch
import time
import sys
import random
import glob
from collections import deque


gym.register_envs(ale_py)


def train_agent(episodes=1000, save_interval=100, render=False):
    """
    Train the DDQN agent on Mario Bros
    
    Args:
        episodes: Number of episodes to train
        save_interval: Save model every N episodes
        render: Whether to render the game
    """
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make("ALE/MarioBros-v5", render_mode=render_mode)
    
    # Get environment information
    n_actions = env.action_space.n
    print(f"Number of actions: {n_actions}")
    
    # Get initial observation to determine state shape
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Initialize agent
    state_shape = (1, obs.shape[0], obs.shape[1])  # Grayscale
    
    agent = Mario(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=32,
        target_update=1000
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    best_avg_reward = -float('inf')
    episodes_since_improvement = 0
    
    start_time = time.time()
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = []
        done = False
        step_count = 0
        
        while not done:
            # Select action
            action = agent.act(obs, training=True)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.cache(obs, action, reward, next_obs, done)
            
            # Learn from experience
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
                recent_losses.append(loss)
            
            # Update state and metrics
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            step_count += 1
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)
        
        # Calculate statistics
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_loss = np.mean(recent_losses) if recent_losses else 0
        avg_length = np.mean([episode_lengths[-min(100, len(episode_lengths)):]]) if episode_lengths else 0
        max_reward = max(episode_rewards) if episode_rewards else 0
        min_reward = min(episode_rewards) if episode_rewards else 0
        
        # Track best performance
        if avg_reward > best_avg_reward and len(recent_rewards) >= 10:
            best_avg_reward = avg_reward
            episodes_since_improvement = 0
        else:
            episodes_since_improvement += 1
        
        # Print progress with enhanced visualization
        elapsed_time = time.time() - start_time
        
        # Detailed progress every 10 episodes
        if episode % 10 == 0:
            print()  # New line after progress bar
            print("="*70)
            print(f"Episode {episode}/{episodes} Summary")
            print("-"*70)
            print(f"Current Episode:")
            print(f"   Reward: {episode_reward:.2f} | Length: {episode_length} steps")
            print(f"Statistics (last 100 episodes):")
            print(f"   Avg Reward: {avg_reward:.2f} | Max: {max_reward:.2f} | Min: {min_reward:.2f}")
            print(f"   Avg Length: {avg_length:.1f} steps")
            print(f"Learning:")
            print(f"   Epsilon: {agent.epsilon:.4f} | Loss: {avg_loss:.6f}")
            print(f"   Buffer Size: {len(agent.memory):,}/{agent.memory.buffer.maxlen:,}")
            print(f"   Training Steps: {agent.steps:,}")
            print(f"Performance:")
            print(f"   Elapsed: {elapsed_time/60:.2f}m | ETA: {((elapsed_time/episode)*(episodes-episode))/60:.2f}m")
            if episodes_since_improvement > 0:
                print(f"Best Avg Reward: {best_avg_reward:.2f} ({episodes_since_improvement} episodes ago)")
            print("="*70 + "\n")
        
        # Save model periodically
        if episode % save_interval == 0:
            print()  # New line before save message
            agent.save(f"models/mario_episode_{episode}.pth")
            print(f"Model checkpoint saved at episode {episode}")
            print(f"   Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}\n")
    
    # Final save
    print()  # New line before final summary
    agent.save("models/mario_final.pth")
    
    env.close()
    
    total_time = time.time() - start_time
    final_avg = np.mean(list(recent_rewards)[-100:]) if len(recent_rewards) >= 100 else np.mean(recent_rewards)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Final Statistics:")
    print(f"   Total Episodes: {episodes}")
    print(f"   Total Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Average Time per Episode: {total_time/episodes:.2f} seconds")
    print(f"\nPerformance:")
    print(f"   Final Avg Reward (last 100): {final_avg:.2f}")
    print(f"   Best Avg Reward: {best_avg_reward:.2f}")
    print(f"   Max Episode Reward: {max(episode_rewards):.2f}")
    print(f"   Final Epsilon: {agent.epsilon:.4f}")
    print(f"\nModel Saved: models/mario_final.pth")
    print("="*70 + "\n")
    
    return agent, episode_rewards, episode_lengths


def play_random(episodes=5, render=True):
    """
    Play using random actions (no trained model)
    
    Args:
        episodes: Number of episodes to play
        render: Whether to render the game
    """
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make("ALE/MarioBros-v5", render_mode=render_mode)
    
    # Get environment information
    n_actions = env.action_space.n
    
    print("\n" + "="*50)
    print(f"🎲 Playing with RANDOM actions")
    print(f"Action space size: {n_actions}")
    print("="*50 + "\n")
    
    total_rewards = []
    total_lengths = []
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select random action
            action = random.randint(0, n_actions - 1)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Random Agent Summary:")
    print(f"   Average Reward: {np.mean(total_rewards):.2f}")
    print(f"   Max Reward: {max(total_rewards):.2f}")
    print(f"   Min Reward: {min(total_rewards):.2f}")
    print(f"   Average Length: {np.mean(total_lengths):.1f}")
    print("="*50 + "\n")
    
    env.close()


def play_agent(model_path, episodes=5, render=True):
    """
    Play using a trained agent
    
    Args:
        model_path: Path to saved model
        episodes: Number of episodes to play
        render: Whether to render the game
    """
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make("ALE/MarioBros-v5", render_mode=render_mode)
    
    # Get environment information
    n_actions = env.action_space.n
    obs, info = env.reset()
    state_shape = (1, obs.shape[0], obs.shape[1])
    
    # Initialize and load agent
    agent = Mario(
        state_shape=state_shape,
        n_actions=n_actions,
        epsilon_start=0.05,  # Low epsilon for evaluation
        epsilon_end=0.05
    )
    agent.load(model_path)
    
    print("\n" + "="*50)
    print(f"Playing with trained agent: {model_path}")
    print("="*50 + "\n")
    
    total_rewards = []
    total_lengths = []
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action (no exploration)
            action = agent.act(obs, training=False)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    env.close()


if __name__ == "__main__":
    import os
    import argparse
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train or play Mario Bros with DDQN")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "play"],
                        help="Mode: 'train' or 'play'")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (for play mode). If not provided, randomly samples from available models")
    parser.add_argument("--render", action="store_true",
                        help="Render the game")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save model every N episodes")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Training mode selected")
        train_agent(
            episodes=args.episodes,
            save_interval=args.save_interval,
            render=args.render
        )
    elif args.mode == "play":
        print("Play mode selected")
        
        # If no model specified, play with random actions
        if args.model is None:
            print("No model specified. Playing with RANDOM actions...")
            play_random(
                episodes=args.episodes,
                render=args.render
            )
        else:
            # Check if the model exists
            if not os.path.exists(args.model):
                print(f"Error: Model file '{args.model}' not found!")
                print("Available models:")
                for model in glob.glob("models/*.pth"):
                    print(f"  - {model}")
                print("\nTrain a model first using: python main.py --mode train")
                print("Or run without --model to use random actions")
            else:
                play_agent(
                    model_path=args.model,
                    episodes=args.episodes,
                    render=args.render
                )
