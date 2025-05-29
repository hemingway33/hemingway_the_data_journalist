"""
Demo: Training a Reinforcement Learning Agent on the Digital Twin Environment

This script demonstrates how to train a simple RL agent to optimize loan portfolio
management decisions using the digital twin environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from twin_env import LoanPortfolioTwinEnv
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePortfolioAgent:
    """
    A simple Q-learning-based agent for portfolio management decisions.
    
    This is a basic demonstration agent. In production, you would use more
    sophisticated algorithms like PPO, SAC, or the existing DQN/MuZero
    implementations from the RL_learner project.
    """
    
    def __init__(
        self, 
        observation_dim: int,
        action_dim: int,
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Simple linear policy (in practice, use neural networks)
        self.weights = np.random.normal(0, 0.1, (observation_dim, action_dim))
        self.bias = np.zeros(action_dim)
        
        # Experience storage
        self.experience = []
        self.max_experience = 10000
        
    def get_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """Get action from the agent's policy"""
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # Exploitation: policy action
            action = np.tanh(observation @ self.weights + self.bias)
        
        # Scale actions to environment bounds
        action_bounds = np.array([0.1, 0.02, 0.1])  # Max changes per dimension
        scaled_action = action * action_bounds
        
        return scaled_action.astype(np.float32)
    
    def store_experience(
        self, 
        observation: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_observation: np.ndarray, 
        done: bool
    ):
        """Store experience for learning"""
        self.experience.append((observation, action, reward, next_observation, done))
        
        # Keep experience buffer size manageable
        if len(self.experience) > self.max_experience:
            self.experience.pop(0)
    
    def update_policy(self, batch_size: int = 32):
        """Update the agent's policy using stored experience"""
        if len(self.experience) < batch_size:
            return
        
        # Sample random batch from experience
        batch_indices = np.random.choice(len(self.experience), batch_size, replace=False)
        batch = [self.experience[i] for i in batch_indices]
        
        # Extract batch components
        observations = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_observations = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Compute target values (simplified Q-learning)
        current_q_values = observations @ self.weights + self.bias
        next_q_values = next_observations @ self.weights + self.bias
        target_q_values = current_q_values.copy()
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i] = rewards[i]
            else:
                target_q_values[i] = rewards[i] + 0.99 * np.max(next_q_values[i])
        
        # Compute gradients and update weights (simplified)
        error = target_q_values - current_q_values
        weight_gradient = observations.T @ error
        bias_gradient = np.mean(error, axis=0)
        
        self.weights += self.learning_rate * weight_gradient / batch_size
        self.bias += self.learning_rate * bias_gradient
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(
    episodes: int = 100,
    episode_length: int = 50,
    render_frequency: int = 10
) -> Tuple[SimplePortfolioAgent, List[float], pd.DataFrame]:
    """
    Train the portfolio management agent
    
    Args:
        episodes: Number of training episodes
        episode_length: Number of steps per episode
        render_frequency: How often to render environment state
    
    Returns:
        Trained agent, episode rewards, and performance data
    """
    
    # Create environment
    env = LoanPortfolioTwinEnv(
        initial_portfolio_size=200,
        max_portfolio_size=2000,
        simulation_days=episode_length,
        render_mode="human" if render_frequency > 0 else None
    )
    
    # Create agent
    agent = SimplePortfolioAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=0.001,
        epsilon=0.3,
        epsilon_decay=0.995
    )
    
    episode_rewards = []
    all_performance_data = []
    
    logger.info(f"Starting training for {episodes} episodes...")
    
    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_performance = []
        
        for step in range(episode_length):
            # Get action from agent
            action = agent.get_action(observation, training=True)
            
            # Take action in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # Store experience
            agent.store_experience(observation, action, reward, next_observation, terminated)
            
            # Update cumulative reward
            episode_reward += reward
            
            # Store performance data
            episode_performance.append({
                'episode': episode,
                'step': step,
                'portfolio_size': info['portfolio_size'],
                'total_value': info['total_value'],
                'expected_loss': info['expected_loss'],
                'reward': reward,
                'action_credit_policy': action[0],
                'action_pricing': action[1],
                'action_rebalancing': action[2]
            })
            
            # Render occasionally
            if render_frequency > 0 and episode % render_frequency == 0 and step % 10 == 0:
                env.render()
            
            observation = next_observation
            
            if terminated or truncated:
                break
        
        # Update agent policy
        agent.update_policy(batch_size=32)
        
        # Store episode data
        episode_rewards.append(episode_reward)
        all_performance_data.extend(episode_performance)
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode:3d}: Avg Reward = {avg_reward:8.2f}, Epsilon = {agent.epsilon:.3f}")
    
    logger.info("Training completed!")
    
    # Convert performance data to DataFrame
    performance_df = pd.DataFrame(all_performance_data)
    
    return agent, episode_rewards, performance_df


def evaluate_agent(
    agent: SimplePortfolioAgent, 
    episodes: int = 10,
    episode_length: int = 100
) -> pd.DataFrame:
    """
    Evaluate trained agent performance
    
    Args:
        agent: Trained agent to evaluate
        episodes: Number of evaluation episodes
        episode_length: Steps per episode
    
    Returns:
        Evaluation performance data
    """
    
    env = LoanPortfolioTwinEnv(
        initial_portfolio_size=500,
        max_portfolio_size=5000,
        simulation_days=episode_length,
        render_mode=None
    )
    
    evaluation_data = []
    
    logger.info(f"Evaluating agent for {episodes} episodes...")
    
    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0
        
        for step in range(episode_length):
            # Get action from agent (no exploration)
            action = agent.get_action(observation, training=False)
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Store evaluation data
            evaluation_data.append({
                'episode': episode,
                'step': step,
                'portfolio_size': info['portfolio_size'],
                'total_value': info['total_value'],
                'expected_loss': info['expected_loss'],
                'reward': reward,
                'cumulative_reward': episode_reward
            })
            
            observation = next_observation
            
            if terminated or truncated:
                break
        
        logger.info(f"Evaluation Episode {episode}: Total Reward = {episode_reward:.2f}")
    
    return pd.DataFrame(evaluation_data)


def plot_training_results(episode_rewards: List[float], performance_df: pd.DataFrame):
    """Plot training results and performance metrics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Digital Twin Agent Training Results', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Portfolio value over time
    grouped = performance_df.groupby('episode')['total_value'].mean()
    axes[0, 1].plot(grouped.index, grouped.values)
    axes[0, 1].set_title('Average Portfolio Value per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].grid(True)
    
    # Expected loss over time
    grouped = performance_df.groupby('episode')['expected_loss'].mean()
    axes[0, 2].plot(grouped.index, grouped.values)
    axes[0, 2].set_title('Average Expected Loss per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Expected Loss ($)')
    axes[0, 2].grid(True)
    
    # Action distributions
    axes[1, 0].hist(performance_df['action_credit_policy'], bins=30, alpha=0.7)
    axes[1, 0].set_title('Credit Policy Actions Distribution')
    axes[1, 0].set_xlabel('Action Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(performance_df['action_pricing'], bins=30, alpha=0.7)
    axes[1, 1].set_title('Pricing Actions Distribution')
    axes[1, 1].set_xlabel('Action Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    # Reward distribution
    axes[1, 2].hist(performance_df['reward'], bins=30, alpha=0.7)
    axes[1, 2].set_title('Reward Distribution')
    axes[1, 2].set_xlabel('Reward Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main demo function"""
    
    print("🎯 Digital Twin Portfolio Agent Training Demo")
    print("=" * 50)
    
    # Train agent
    print("\n📚 Training Phase...")
    agent, episode_rewards, performance_df = train_agent(
        episodes=50,
        episode_length=30,
        render_frequency=20
    )
    
    # Evaluate agent
    print("\n🧪 Evaluation Phase...")
    evaluation_df = evaluate_agent(agent, episodes=5, episode_length=50)
    
    # Display results
    print("\n📊 Training Results Summary:")
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Final Epsilon (Exploration): {agent.epsilon:.3f}")
    
    print("\n📈 Portfolio Performance Metrics:")
    final_episode_data = performance_df[performance_df['episode'] == performance_df['episode'].max()]
    avg_portfolio_value = final_episode_data['total_value'].mean()
    avg_expected_loss = final_episode_data['expected_loss'].mean()
    
    print(f"Average Portfolio Value: ${avg_portfolio_value:,.2f}")
    print(f"Average Expected Loss: ${avg_expected_loss:,.2f}")
    print(f"Expected Loss Rate: {(avg_expected_loss/avg_portfolio_value)*100:.2f}%")
    
    print("\n📉 Evaluation Results:")
    eval_rewards = evaluation_df.groupby('episode')['cumulative_reward'].last()
    print(f"Evaluation Episodes: {len(eval_rewards)}")
    print(f"Average Evaluation Reward: {eval_rewards.mean():.2f} ± {eval_rewards.std():.2f}")
    
    # Plot results
    print("\n📊 Generating performance plots...")
    plot_training_results(episode_rewards, performance_df)
    
    print("\n✅ Demo completed successfully!")
    print("\n🎯 Next Steps:")
    print("1. Integrate with existing RL_learner/ algorithms (DQN, MuZero)")
    print("2. Add more sophisticated reward functions")
    print("3. Implement multi-agent scenarios")
    print("4. Connect to real portfolio data")
    print("5. Add stress testing scenarios")


if __name__ == "__main__":
    main() 