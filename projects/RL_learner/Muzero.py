import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import collections
import math
import time
import torch.nn.functional as F
import random
import cv2 # For preprocessing

# Configuration for MuZero (can be moved to a separate config file/class later)
class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.environment = "ALE/Breakout-v5" # Example Atari environment
        self.num_actors = 1
        self.num_training_steps = 1000000
        self.num_simulations = 50 # MCTS simulations per move
        self.td_steps = 10 # Number of future steps for TD learning
        self.num_unroll_steps = 5 # Number of steps to unroll in the dynamics model
        self.batch_size = 1024
        self.replay_buffer_size = 100000

        # Network architecture details
        self.observation_shape = (4, 96, 96) # Example for preprocessed Atari frames (stack 4, 96x96)
        self.action_space_size = 4 # Example for Breakout (NoOp, Fire, Right, Left)
        self.encoding_size = 256 # Dimension of the hidden state
        self.fc_reward_layers = [64]
        self.fc_value_layers = [64]
        self.fc_policy_layers = [64]

        # Learning rates
        self.lr_init = 0.05
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 350e3

        # Regularization
        self.weight_decay = 1e-4

        # MCTS parameters
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Loss scaling
        self.value_loss_weight = 0.25
        self.reward_loss_weight = 1.0
        self.policy_loss_weight = 1.0

        # Discount factor
        self.discount = 0.997


# Helper function for MLP
def mlp(input_size, layer_sizes, output_size, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    sizes = [input_size] + layer_sizes + [output_size]
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        layers.append(act())
    return nn.Sequential(*layers)


# Basic Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Basic Residual Block (Identity)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels) # No ReLU on final add

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity # Add identity
        return F.relu(out) # Apply ReLU after addition


# Network components implementation
class MuZeroNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        self.action_space_size = config.action_space_size
        self.encoding_size = config.encoding_size

        # Representation Network (h) - Based loosely on Atari ResNet architecture
        # Input: (B, C=4, H=96, W=96)
        self.repr_conv1 = ConvBlock(config.observation_shape[0], 128, stride=2) # -> (B, 128, 48, 48)
        self.repr_res1 = nn.Sequential(*[ResBlock(128) for _ in range(2)])
        self.repr_conv2 = ConvBlock(128, 256, stride=2) # -> (B, 256, 24, 24)
        self.repr_res2 = nn.Sequential(*[ResBlock(256) for _ in range(3)])
        self.repr_pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # -> (B, 256, 12, 12)
        self.repr_res3 = nn.Sequential(*[ResBlock(256) for _ in range(3)])
        self.repr_pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # -> (B, 256, 6, 6)

        # Calculate the flattened size after convolutions/pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, *config.observation_shape)
            conv_output = self.repr_pool2(self.repr_res3(self.repr_pool1(self.repr_res2(self.repr_conv2(self.repr_res1(self.repr_conv1(dummy_input)))))))
            self._repr_output_flat_size = conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3]

        self.repr_fc = nn.Linear(self._repr_output_flat_size, self.encoding_size)


        # Dynamics Network (g)
        # Input: hidden_state (B, encoding_size) + encoded_action (B, encoding_size)
        # We encode the action and add it to the hidden state.
        # A simple approach: Action is one-hot encoded, then passed through a small MLP or embedding.
        # Here, let's embed the action and add it to the state.
        # An alternative is to have separate layers processing state and action then combining.
        self.action_embedding_size = self.encoding_size # Make action embedding same size as state for simple addition
        self.action_encoder = nn.Embedding(self.action_space_size, self.action_embedding_size)

        # Simple Dynamics: MLP processing the combined state-action
        self.dynamics_state_mlp = mlp(self.encoding_size, [self.encoding_size], self.encoding_size) # Hidden state -> Hidden state
        # Reward prediction head
        self.dynamics_reward_mlp = mlp(self.encoding_size, config.fc_reward_layers, 1) # Hidden state -> Reward


        # Prediction Network (f)
        # Input: hidden_state (B, encoding_size)
        self.pred_policy_mlp = mlp(self.encoding_size, config.fc_policy_layers, self.action_space_size)
        self.pred_value_mlp = mlp(self.encoding_size, config.fc_value_layers, 1)


    def representation(self, observation_batch):
        # Normalize observation (assuming input is 0-255)
        x = observation_batch / 255.0
        x = self.repr_conv1(x)
        x = self.repr_res1(x)
        x = self.repr_conv2(x)
        x = self.repr_res2(x)
        x = self.repr_pool1(x)
        x = self.repr_res3(x)
        x = self.repr_pool2(x)
        x = x.view(x.size(0), -1) # Flatten
        hidden_state = self.repr_fc(x)
        # Normalize hidden state (optional but can help)
        hidden_state = F.normalize(hidden_state, p=2, dim=1)
        return hidden_state

    def dynamics(self, hidden_state_batch, action_batch):
        # Embed actions
        # Ensure action_batch is LongTensor for embedding lookup
        action_batch = action_batch.long().squeeze(-1) # Remove last dim if it exists (e.g., [B, 1] -> [B])
        action_embedding = self.action_encoder(action_batch) # (B, action_embedding_size)

        # Combine state and action embedding
        # Simple addition - more complex interactions possible (e.g., concatenation + MLP)
        state_action_combined = hidden_state_batch + action_embedding

        # Process combined vector to get next state
        next_hidden_state = self.dynamics_state_mlp(state_action_combined)
        # Normalize next hidden state
        next_hidden_state = F.normalize(next_hidden_state, p=2, dim=1)

        # Predict reward from the *next* hidden state
        reward = self.dynamics_reward_mlp(next_hidden_state)
        # Squeeze reward prediction: (B, 1) -> (B,)
        reward = reward.squeeze(-1)

        return next_hidden_state, reward

    def prediction(self, hidden_state_batch):
        policy_logits = self.pred_policy_mlp(hidden_state_batch) # (B, action_space_size)
        value = self.pred_value_mlp(hidden_state_batch) # (B, 1)
        # Squeeze value prediction: (B, 1) -> (B,)
        value = value.squeeze(-1)
        return policy_logits, value

    def initial_inference(self, observation_batch):
        hidden_state = self.representation(observation_batch)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state_batch, action_batch):
        next_hidden_state, reward = self.dynamics(hidden_state_batch, action_batch)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value

# MCTS Node Implementation
class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    # Calculate the PUCT score for a given child action
    def ucb_score(self, child_node: 'Node', parent_visit_count: int, config: MuZeroConfig) -> float:
        pb_c = math.log((parent_visit_count + config.pb_c_base + 1) /
                        config.pb_c_base) + config.pb_c_init
        pb_c *= math.sqrt(parent_visit_count) / (child_node.visit_count + 1)

        prior_score = pb_c * child_node.prior
        # Use the child node's value if visited, otherwise 0
        value_score = child_node.value()

        return prior_score + value_score

    # Select the child with the highest UCB score
    def select_child(self, config: MuZeroConfig):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = self.ucb_score(child, self.visit_count, config)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

# TODO: Implement Replay Buffer
class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.replay_buffer_size
        self.batch_size = config.batch_size
        self.buffer = collections.deque(maxlen=self.window_size)
        self.config = config # Store config

    def save_game(self, game):
        if len(self.buffer) == self.window_size:
            self.buffer.popleft()
        self.buffer.append(game)

    # Sample a batch of trajectories for training
    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        if len(self.buffer) < self.batch_size:
            # Not enough games in buffer to form a batch
            # Options: return None, raise error, or wait (depends on training loop)
            return None # Or raise ValueError("Not enough games in replay buffer to sample a batch.")

        obs_batch, action_batch, target_value_batch, target_reward_batch, target_policy_batch = [], [], [], [], []

        # Sample game indices
        game_indices = [random.randint(0, len(self.buffer) - 1) for _ in range(self.batch_size)]

        for game_idx in game_indices:
            game = self.buffer[game_idx]
            # Sample a starting point (state index) within the game
            # Ensure we have enough steps for unrolling
            max_start_index = len(game) - num_unroll_steps
            if max_start_index <= 0:
                 # Game is too short for the unroll steps, skip or handle differently
                 # For simplicity, we might resample or skip this game in a real implementation
                 # Here, let's just pick index 0 if possible, otherwise error
                 if len(game) > 0:
                     state_index = 0
                 else:
                     # This game is empty, highly unlikely but handle
                     print(f"Warning: Sampled an empty game (index {game_idx}). Skipping.")
                     continue # Or retry sampling
            else:
                state_index = random.randint(0, max_start_index -1)

            # Get the initial observation for this trajectory
            obs = game.get_observation(state_index)
            obs_batch.append(obs)

            # Get the sequence of actions taken during unrolling
            # Actions correspond to transitions *from* state k to k+1
            actions = [game.get_action(state_index + k) for k in range(1, num_unroll_steps + 1)]
            action_batch.append(actions)

            # Get the targets (value, reward, policy) for the initial state + unroll steps
            targets = game.make_target(state_index, num_unroll_steps, td_steps)
            # targets is a list of tuples: [(val_0, rew_0, pol_0), (val_1, rew_1, pol_1), ...]
            # Note: rew_0 is reward observed *after* action from state_index

            # Separate the targets
            target_values, target_rewards, target_policies = zip(*targets)
            target_value_batch.append(list(target_values))
            target_reward_batch.append(list(target_rewards)) # Rewards are for step k=0...num_unroll_steps
            target_policy_batch.append(list(target_policies))

        # Convert lists to tensors/numpy arrays
        # Observations: (batch_size, C, H, W)
        obs_batch = np.stack(obs_batch).astype(np.float32)
        # Actions: (batch_size, num_unroll_steps)
        action_batch = np.array(action_batch).astype(np.int64)
        # Targets (Value): (batch_size, num_unroll_steps + 1)
        target_value_batch = np.array(target_value_batch).astype(np.float32)
        # Targets (Reward): (batch_size, num_unroll_steps + 1)
        target_reward_batch = np.array(target_reward_batch).astype(np.float32)
        # Targets (Policy): (batch_size, num_unroll_steps + 1, action_space_size)
        target_policy_batch = np.stack(target_policy_batch).astype(np.float32)

        # Return data as a dictionary (easier to handle)
        batch = {
            "observation": obs_batch,
            "actions": action_batch, # Actions for steps k=1 to num_unroll_steps
            "target_value": target_value_batch, # Targets for state k=0 to num_unroll_steps
            "target_reward": target_reward_batch, # Targets for reward after action k=0 to num_unroll_steps
            "target_policy": target_policy_batch, # Targets for policy at state k=0 to num_unroll_steps
        }
        return batch

# Represents a single episode of self-play
class Game:
    def __init__(self, action_space_size: int, discount: float, config: MuZeroConfig):
        self.environment = None # This will be the gym environment instance
        self.config = config
        self.history = [] # Stores (observation, action, reward, done)
        self.child_visits = [] # Stores MCTS policy targets (visit counts) for each step
        self.root_values = [] # Stores MCTS root values for each step
        self.action_space_size = action_space_size
        self.discount = discount
        self.game_over = False

    def store_search_statistics(self, root_node: Node, action: int):
        # Store the policy target (visit counts) and root value estimate
        sum_visits = sum(child.visit_count for child in root_node.children.values())
        policy_target = np.zeros(self.action_space_size)
        if sum_visits > 0:
            for action_idx, child in root_node.children.items():
                policy_target[action_idx] = child.visit_count / sum_visits
        else: # If no visits (shouldn't happen often after MCTS), use uniform
            policy_target.fill(1.0 / self.action_space_size)

        self.child_visits.append(policy_target)
        self.root_values.append(root_node.value())

    def terminal(self) -> bool:
        # Checks if the game stored in history has ended
        return self.game_over

    # Generate targets for training the network at a given state index
    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        targets = []
        # The first target is for the board state at state_index
        # Value target is bootstrapped using n-step return
        # Reward target is the real reward observed
        # Policy target is the MCTS policy distribution

        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            # Calculate the value target (n-step return)
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                # Discount future rewards and add bootstrapped value
                value = self.root_values[bootstrap_index] * (self.discount**td_steps)
                for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                    value += reward * (self.discount**i)
            else:
                # If bootstrap index is beyond game end, just use discounted rewards to end
                value = 0
                for i, reward in enumerate(self.rewards[current_index:]):
                    value += reward * (self.discount**i)

            # Reward target: Use the real reward observed *after* the action at current_index
            # Note: MuZero predicts the reward *received after* taking the action from the current state.
            # So, target for state_index uses reward at state_index+1
            if current_index < len(self.rewards):
                 # Scale reward (optional but common)
                reward_target = self.rewards[current_index]
            else:
                # No more rewards if past game end
                reward_target = 0.0

            # Policy target: Use the MCTS search policy stored for current_index
            if current_index < len(self.child_visits):
                policy_target = self.child_visits[current_index]
            else:
                # No policy target if beyond game length (e.g., during unrolling past terminal state)
                # Use a uniform policy or zeros? Using zeros might be safer.
                policy_target = np.zeros(self.action_space_size, dtype=np.float32)

            targets.append((value, reward_target, policy_target))

        return targets

    def store_step(self, observation, action, reward, done):
        self.history.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        if done:
            self.game_over = True

    # Add helper methods for data access if needed
    def get_observation(self, index):
        return self.history[index]

    def get_action(self, index):
        return self.actions[index]

    def __len__(self):
        # Length of the game in terms of states/observations stored
        return len(self.history)

# TODO: Implement the main MuZero training loop and MCTS logic
class MuZero:
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Environment setup (with basic preprocessing wrapper)
        self.env = AtariPreprocessing(gym.make(config.environment, obs_type="grayscale"), config.observation_shape)
        # Update config based on actual env if needed (should match defaults generally)
        self.config.action_space_size = self.env.action_space.n

        self.network = MuZeroNetwork(config).to(self.device)
        self.replay_buffer = ReplayBuffer(config)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
        # TODO: Add learning rate scheduler based on config.lr_decay_steps, config.lr_decay_rate
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.lr_decay_rate**(1/config.lr_decay_steps)) # Example scheduler

    # Add exploration noise (Dirichlet) to the root priors
    def _add_exploration_noise(self, node: Node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    # Expand a leaf node using the network
    def _expand_node(self, node: Node, network: MuZeroNetwork):
        if node.hidden_state is None:
            raise ValueError("Cannot expand root node without hidden state")

        # Ensure hidden_state is on the correct device and add batch dim
        hidden_state_tensor = torch.tensor(node.hidden_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = network.prediction(hidden_state_tensor)

        # Squeeze batch dim and move to CPU
        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        value = value.item() # Get scalar value
        policy_probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits)) # Softmax

        # Populate children
        action_space_size = network.config.action_space_size
        for action in range(action_space_size):
            node.children[action] = Node(policy_probs[action])

        return value # Return the value for backpropagation

    # Backpropagate the value up the search path
    def _backpropagate(self, search_path: list, value: float, discount: float):
        # The value passed is from the perspective of the leaf node.
        # We need to discount it as we go back up the path.
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # Discount value for the parent node
            value *= discount

    # Run a single MCTS simulation from the root node
    def _run_simulation(self, root: Node, network: MuZeroNetwork, discount: float):
        node = root
        search_path = [node]
        action_history = []

        while node.expanded():
            action, node = node.select_child(self.config)
            search_path.append(node)
            action_history.append(action)

        # Now at a leaf node
        parent = search_path[-2]
        leaf_node = search_path[-1]

        # Get the hidden state from the parent and the action taken
        hidden_state = parent.hidden_state
        action = action_history[-1]

        if hidden_state is None: # Should only happen if root node is leaf
            raise Exception("Parent hidden state is None during expansion, this shouldn't happen unless root is leaf")

        # Ensure state and action are tensors on the correct device
        hidden_state_tensor = torch.tensor(hidden_state).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([[action]]).to(self.device) # Add batch dims

        # Use the recurrent inference function (dynamics + prediction)
        with torch.no_grad():
            next_hidden_state, reward, policy_logits, value = network.recurrent_inference(hidden_state_tensor, action_tensor)

        # Squeeze batch dim and move to CPU
        next_hidden_state = next_hidden_state.squeeze(0).cpu().numpy()
        reward = reward.item() # scalar
        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        value = value.item() # scalar
        policy_probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits)) # Softmax

        # Store the results in the leaf node
        leaf_node.hidden_state = next_hidden_state
        leaf_node.reward = reward

        # Expand the leaf node using the predicted policy
        action_space_size = network.config.action_space_size
        for a in range(action_space_size):
            leaf_node.children[a] = Node(policy_probs[a])
            leaf_node.children[a].reward = 0 # Rewards are associated with the *transition* into the state

        # Backpropagate the value found by the network
        self._backpropagate(search_path, value, discount)


    # Core MCTS function
    def run_mcts(self, root_observation: np.ndarray, network: MuZeroNetwork, discount: float):
        # 1. Initial Inference for the root node
        root_observation_tensor = torch.tensor(root_observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            root_hidden_state, root_policy_logits, root_value = network.initial_inference(root_observation_tensor)

        root_hidden_state = root_hidden_state.squeeze(0).cpu().numpy()
        root_policy_logits = root_policy_logits.squeeze(0).cpu().numpy()
        root_value = root_value.item()
        root_policy_probs = np.exp(root_policy_logits) / np.sum(np.exp(root_policy_logits))

        # Create root node
        root = Node(prior=0) # Prior doesn't matter for root if we expand immediately
        root.hidden_state = root_hidden_state
        # Expand root immediately
        action_space_size = network.config.action_space_size
        for action in range(action_space_size):
            root.children[action] = Node(root_policy_probs[action])
            # Children don't have hidden state until visited/expanded by simulation

        # Add exploration noise
        self._add_exploration_noise(root)

        # Backpropagate the initial network value estimate for the root node
        # This ensures the root value isn't zero if no simulations finish
        root.value_sum += root_value
        root.visit_count += 1 # Virtual visit for initial inference

        # 2. Run simulations
        num_simulations = self.config.num_simulations
        for _ in range(num_simulations):
            self._run_simulation(root, network, discount)

        return root

    # Select action based on visit counts (potentially with temperature)
    def select_action(self, node: Node, temperature: float = 1.0, deterministic: bool = False):
        if not node.expanded():
             raise ValueError("Cannot select action from a non-expanded node.")

        visit_counts = np.array([child.visit_count for child in node.children.values()], dtype=np.float32)
        actions = list(node.children.keys())

        if deterministic:
            # Choose the action with the highest visit count
            action_pos = np.argmax(visit_counts)
            action = actions[action_pos]
        else:
            if temperature == 0:
                # Should not happen if deterministic=False, but handle for safety
                action_pos = np.argmax(visit_counts)
                action = actions[action_pos]
            elif temperature == 1.0:
                # Sample directly proportionally to visit counts
                action_probs = visit_counts / np.sum(visit_counts)
                action = np.random.choice(actions, p=action_probs)
            else:
                # Sample using temperature scaling
                log_visits = np.log(visit_counts + 1e-10) # Add epsilon for numerical stability
                temp_scaled_visits = (log_visits / temperature)
                # Softmax calculation for temperature scaling
                exp_visits = np.exp(temp_scaled_visits - np.max(temp_scaled_visits)) # Subtract max for stability
                action_probs = exp_visits / np.sum(exp_visits)
                action = np.random.choice(actions, p=action_probs)

        # Return the chosen action and the visit count distribution (for training targets)
        visit_distribution = {a: node.children[a].visit_count for a in actions}
        return action, visit_distribution

    # Compute the MuZero loss
    def _compute_loss(self, batch) -> torch.Tensor:
        value_loss_accum, reward_loss_accum, policy_loss_accum = 0, 0, 0
        num_unroll_steps = batch['actions'].shape[1]

        # Initial inference
        initial_obs = torch.tensor(batch['observation']).to(self.device)
        hidden_state, policy_logits, value = self.network.initial_inference(initial_obs)

        # Calculate loss for the initial step (k=0)
        target_value_k = torch.tensor(batch['target_value'][:, 0]).to(self.device)
        target_policy_k = torch.tensor(batch['target_policy'][:, 0]).to(self.device)

        value_loss = F.mse_loss(value, target_value_k)
        # Use cross-entropy loss for policy (logits vs probabilities)
        policy_loss = F.cross_entropy(policy_logits, target_policy_k)

        value_loss_accum += value_loss
        policy_loss_accum += policy_loss
        # Reward loss starts from k=1 (prediction based on action taken at k=0)

        # Recurrent inference and loss calculation for unroll steps (k=1 to num_unroll_steps)
        for k in range(num_unroll_steps):
            action_k = torch.tensor(batch['actions'][:, k]).to(self.device)
            # Get predictions from dynamics and prediction functions
            hidden_state, reward, policy_logits, value = self.network.recurrent_inference(hidden_state, action_k)

            # Targets for step k+1 (value, reward, policy)
            target_value_k_plus_1 = torch.tensor(batch['target_value'][:, k + 1]).to(self.device)
            target_reward_k = torch.tensor(batch['target_reward'][:, k]).to(self.device) # Reward received *after* action k
            target_policy_k_plus_1 = torch.tensor(batch['target_policy'][:, k + 1]).to(self.device)

            # Calculate losses for step k+1
            value_loss = F.mse_loss(value, target_value_k_plus_1)
            reward_loss = F.mse_loss(reward, target_reward_k)
            policy_loss = F.cross_entropy(policy_logits, target_policy_k_plus_1)

            value_loss_accum += value_loss
            reward_loss_accum += reward_loss
            policy_loss_accum += policy_loss

            # Scale hidden state gradient (as per MuZero paper Appendix G)
            hidden_state.register_hook(lambda grad: grad * 0.5)

        # Combine losses with weights
        total_loss = (self.config.value_loss_weight * value_loss_accum +
                      self.config.reward_loss_weight * reward_loss_accum +
                      self.config.policy_loss_weight * policy_loss_accum)

        # Average loss over unroll steps (+ initial step)
        # Note: reward loss has only num_unroll_steps terms
        # Policy/Value loss have num_unroll_steps + 1 terms
        # A simple average across all terms might be fine, or weigh appropriately
        # Let's average across total number of predictions made
        total_loss /= (num_unroll_steps + 1) # Average over sequence length

        return total_loss

    # Main training loop
    def train(self):
        total_steps = 0
        total_episodes = 0
        start_time = time.time()

        while total_steps < self.config.num_training_steps:
            # --- Self-Play Phase ---            print(f"Starting episode {total_episodes + 1}")
            current_game = Game(self.config.action_space_size, self.config.discount, self.config)
            observation, _ = self.env.reset()
            current_game.history.append(observation) # Store initial observation
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done and episode_steps < 2000: # Max episode length safeguard
                # 1. Run MCTS
                root_node = self.run_mcts(observation, self.network, self.config.discount)

                # 2. Select action (using temperature for exploration)
                # Temperature scheduling (example: high early, low later)
                temperature = 1.0 # Simple constant temperature for now
                action, visit_dist = self.select_action(root_node, temperature=temperature, deterministic=False)

                # 3. Store search stats
                current_game.store_search_statistics(root_node, action)

                # 4. Take action in environment
                new_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # 5. Store step results in game history
                # Store the *next* observation, the action taken, the reward received, and done flag
                current_game.store_step(new_observation, action, reward, done)

                observation = new_observation
                episode_steps += 1
                total_steps += 1

                if done:
                    print(f"Episode {total_episodes + 1} finished after {episode_steps} steps. Reward: {episode_reward}")

            # --- End Self-Play Episode ---            current_game.game_over = True # Mark game as over
            self.replay_buffer.save_game(current_game)
            total_episodes += 1

            # --- Training Phase ---            if len(self.replay_buffer.buffer) >= self.config.batch_size: # Start training once buffer has enough games
            if total_steps % 50 == 0: # Train every N steps (adjust frequency)
                print(f"\n--- Training Step {total_steps} --- ")
                self.network.train() # Set network to training mode

                # Sample batch
                batch = self.replay_buffer.sample_batch(self.config.num_unroll_steps, self.config.td_steps)
                if batch is None: continue # Skip if buffer not ready

                # Compute loss
                loss = self._compute_loss(batch)

                # Backpropagate and optimize
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (optional but often helpful)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.lr_scheduler.step() # Step the LR scheduler

                self.network.eval() # Set back to evaluation mode for MCTS

                # Logging
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Learning Rate: {current_lr:.6f}")
                elapsed_time = time.time() - start_time
                print(f"  Elapsed Time: {elapsed_time:.2f}s")
                print(f"  Total Steps: {total_steps}")
                print(f"  Replay Buffer Size: {len(self.replay_buffer.buffer)}")
                print(f"---------------------------")

        # TODO: Add checkpointing periodically

        print("\nTraining finished.")
        self.env.close()


# --- Atari Preprocessing Wrapper (Placeholder) ---
# A proper implementation would handle grayscaling, resizing, frame stacking.
class AtariPreprocessing(gym.ObservationWrapper):
    def __init__(self, env, output_shape):
        super().__init__(env)
        self.output_shape = output_shape # (C, H, W)
        self._buffer = collections.deque(maxlen=output_shape[0])
        # Assuming input env is already grayscaled
        # Set observation space
        # Input shape from env is likely (H, W)
        input_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                  shape=output_shape, dtype=np.uint8)

    def observation(self, obs):
        # Resize
        h, w = self.output_shape[1], self.output_shape[2]
        resized_obs = cv2.resize(obs, (w, h), interpolation=cv2.INTER_AREA)
        # Add channel dimension if it's missing (was H,W -> H,W,1)
        if len(resized_obs.shape) == 2:
            resized_obs = resized_obs[:, :, None]

        # Add to buffer
        self._buffer.append(resized_obs)

        # Fill buffer initially if needed
        while len(self._buffer) < self.output_shape[0]:
            self._buffer.append(resized_obs) # Duplicate frame

        # Stack frames along the channel axis
        stacked_frames = np.concatenate(list(self._buffer), axis=2)
        # Transpose to (C, H, W)
        return np.transpose(stacked_frames, (2, 0, 1)).astype(np.uint8)

    def reset(self, **kwargs):
        self._buffer.clear()
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info

if __name__ == '__main__':
    config = MuZeroConfig()
    # Set specific config overrides if needed
    # config.environment = "ALE/Pong-v5"

    # Determine action space size from environment
    try:
        # Temporarily create env to get action space size if not overridden
        # This assumes the default config env exists
        temp_env = gym.make(config.environment)
        config.action_space_size = temp_env.action_space.n
        temp_env.close()
        print(f"Determined action space size for {config.environment}: {config.action_space_size}")
    except Exception as e:
        print(f"Warning: Could not automatically determine action space size for {config.environment}. Using default {config.action_space_size}. Error: {e}")
        # Ensure a default value is set if env creation fails
        if not hasattr(config, 'action_space_size') or config.action_space_size is None:
             config.action_space_size = 4 # Fallback default
             print(f"Falling back to default action space size: {config.action_space_size}")

    # Initialize MuZero
    muzero = MuZero(config)

    # Start training
    try:
        print("\nStarting MuZero training...")
        muzero.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        # Optional: Save state before exiting on error
        # muzero.save_checkpoint("error_checkpoint.pth")
    finally:
        # Ensure environment is closed properly
        if hasattr(muzero, 'env') and muzero.env is not None:
            muzero.env.close()
            print("Environment closed.")

    print("MuZero training process finished or exited.")
