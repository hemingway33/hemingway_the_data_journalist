import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from collections import deque, namedtuple

# Register the Go environment from gym_go
# Although gym-go uses the older 'gym' registration, gymnasium can find it.
import gym_go

# Define the experience tuple
Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): device to store tensors on (CPU or GPU)
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # Convert done from boolean to int (0 or 1) before converting to tensor
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# --- Placeholder for QNetwork ---
class QNetwork(nn.Module):
    """Neural Network model for approximating Q-values."""

    def __init__(self, board_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            board_size (int): Dimension of the Go board (e.g., 7 for 7x7)
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.board_size = board_size
        self.action_size = board_size * board_size + 1 # +1 for the pass move
        input_channels = 6 # Based on GymGo state representation

        # Define convolutional layers
        # Adjust kernel sizes, strides, padding, and channel counts as needed
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Potentially add more conv layers for larger boards

        # Calculate the flattened size after convolutions
        # This depends on the board size and conv layers
        # For a 7x7 board and the layers above (no pooling/stride > 1):
        # Size remains 7x7. Adjust if using different conv params.
        conv_output_size = 64 * board_size * board_size

        # Define fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- Placeholder for DQNAgent ---
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # For soft update of target parameters
LR = 5e-4               # Learning rate
UPDATE_EVERY = 4        # How often to update the network
TARGET_UPDATE_EVERY = 1000 # How often to update the target network

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, board_size, seed):
        """Initialize an Agent object.

        Params
        ======
            board_size (int): Dimension of the Go board
            seed (int): Random seed
        """
        self.board_size = board_size
        self.action_size = board_size * board_size + 1
        self.state_shape = (6, board_size, board_size) # C, H, W
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.qnetwork_local = QNetwork(board_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(board_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Initialize target network weights same as local network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0) # Hard copy

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed, self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self._t_step = 0
        # Initialize time step for target network update
        self._target_update_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self._t_step = (self._t_step + 1) % UPDATE_EVERY
        if self._t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

        # Update target network every TARGET_UPDATE_EVERY steps
        self._target_update_step = (self._target_update_step + 1) % TARGET_UPDATE_EVERY
        if self._target_update_step == 0:
             self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state (C, H, W)
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Add batch dimension if missing
        if state.ndim == len(self.state_shape):
             state = np.expand_dims(state, 0)

        state = torch.from_numpy(state).float().to(self.device)
        self.qnetwork_local.eval()  # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # Set network back to train mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            # Get the action with the highest Q-value
            return np.argmax(action_values.cpu().data.numpy()).item()
        else:
            # Choose a random action
            # Important: Need to consider *valid* moves from the state
            # The 4th channel of the state (index 3) indicates invalid moves (1 = invalid)
            # Let's assume state is already a numpy array here before torch conversion
            invalid_moves_flat = state[0, 3].flatten() # Get invalid moves channel for the single state
            valid_action_indices = [i for i, invalid in enumerate(invalid_moves_flat) if not invalid]
            # Also need to add the pass move if it's considered valid (always? GymGo might include it in invalid channel if ko)
            # GymGo action space includes pass as the last action (index board_size*board_size)
            pass_action_index = self.board_size * self.board_size
            if pass_action_index not in valid_action_indices:
                 # Check if the pass action itself is marked invalid in the state representation
                 # This part needs clarification from GymGo's state details or careful handling.
                 # For now, let's assume pass is always possible unless explicitly marked invalid.
                 # We might need to refine this based on GymGo specifics.
                 # Let's just add pass for now if it's not present.
                 valid_action_indices.append(pass_action_index)

            if not valid_action_indices: # Should not happen in Go unless board is full AND pass is invalid? Defensive check.
                return pass_action_index # Default to pass if no other valid moves somehow

            return random.choice(valid_action_indices)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # Note: For Go, we need to mask invalid actions in the next_state before max()
        # The 4th channel (index 3) of next_states holds invalid move masks
        next_state_invalid_masks = next_states[:, 3, :, :].bool() # Shape: (batch_size, H, W)
        # Flatten the masks to match action indices (H*W)
        next_state_invalid_masks_flat = next_state_invalid_masks.view(BATCH_SIZE, -1) # Shape: (batch_size, H*W)

        # Need to add the 'pass' action mask. Assuming 'pass' (last action index) is always valid unless explicitly masked (needs check)
        # Let's assume pass is valid for now if not masked.
        pass_mask = torch.zeros((BATCH_SIZE, 1), dtype=torch.bool, device=self.device)
        full_invalid_mask = torch.cat((next_state_invalid_masks_flat, pass_mask), dim=1) # Shape: (batch_size, H*W + 1)

        Q_targets_next = self.qnetwork_target(next_states).detach() # Shape: (batch_size, action_size)

        # Apply the mask: set Q-values of invalid actions to a very small number (-inf)
        # so they won't be chosen by max(). Add small epsilon to avoid -inf * 0 issues if needed.
        Q_targets_next = Q_targets_next.masked_fill(full_invalid_mask, -1e9)
        Q_targets_next = Q_targets_next.max(1)[0].unsqueeze(1) # Shape: (batch_size, 1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# --- Placeholder for training loop ---
def train_dqn(board_size=7, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Train the DQN agent.

    Params
    ======
        board_size (int): Size of the Go board (e.g., 7 for 7x7)
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode (can be game length for Go)
        eps_start (float): Starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): Minimum value of epsilon
        eps_decay (float): Multiplicative factor (per episode) for decreasing epsilon
    """
    # Use komi=0 for simplicity, reward_method='real' gives +1/-1 for win/loss
    # Use Gymnasium compatibility wrapper if needed, though make usually handles it
    env = gym.make('gym_go:go-v0', size=board_size, komi=0, reward_method='real')
    agent = DQNAgent(board_size=board_size, seed=0)

    scores = []                        # List containing scores from each episode
    scores_window = deque(maxlen=100)  # Last 100 scores
    eps = eps_start                    # Initialize epsilon

    print(f"Training on {agent.device}...")

    for i_episode in range(1, n_episodes+1):
        # Reset requires seed in newer Gymnasium versions, but gym_go might use older API
        # Try without first, add if it errors.
        try:
            state, info = env.reset()
        except TypeError:
            # Handle potential older gym API if reset requires seed
            state = env.reset()
            info = {} # Dummy info

        score = 0
        # GymGo state is (6, size, size). Ensure it matches agent expectation.

        # Go games can vary in length. Use max_t as a safeguard, but rely on 'done'.
        for t in range(max_t):
            action = agent.act(state, eps)

            # Gym-go returns state, reward, done, info
            # Gymnasium returns state, reward, terminated, truncated, info
            # Need to handle potential differences. Assume GymGo might return older format.
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # Likely older gym format: state, reward, done, info
                next_state, reward, done, info = env.step(action)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward # Accumulate reward (usually 0 until game end for 'real' reward)
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps) # Decrease epsilon

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        # Add condition to finish early if solved (e.g., avg score > threshold)
        # For Go win rate against a fixed opponent might be better metric.
        # if np.mean(scores_window)>=WIN_THRESHOLD:
        #     print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
        #     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_go_solved.pth')
        #     break

    # Save final model
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_go_final.pth')
    env.close()
    print("\nTraining finished.")
    return scores

if __name__ == '__main__':
    BOARD_DIMENSION = 7 # Example: 7x7 board
    print(f"Starting DQN training for {BOARD_DIMENSION}x{BOARD_DIMENSION} Go...")
    scores = train_dqn(board_size=BOARD_DIMENSION, n_episodes=5000, max_t=BOARD_DIMENSION*BOARD_DIMENSION*2) # max_t heuristic
    # Optional: Plot scores
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(len(scores)), scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()
