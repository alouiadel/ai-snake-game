"""AI training module for the snake game using Deep Q-Learning."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from constants import GRID_WIDTH, GRID_HEIGHT
from collections import deque
import random
from datetime import datetime

# Set device for PyTorch (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SnakeAI(nn.Module):
    """Neural network model for the snake AI."""

    def __init__(self):
        """Initialize the neural network architecture."""
        super().__init__()
        self.linear1 = nn.Linear(11, 256)  # Input layer: 11 state features
        self.linear2 = nn.Linear(256, 128)  # Hidden layer
        self.linear3 = nn.Linear(128, 4)  # Output layer: 4 possible actions
        self.to(device)  # Move model to GPU if available

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.linear1(x))  # First layer with ReLU
        x = torch.relu(self.linear2(x))  # Second layer with ReLU
        return self.linear3(x)  # Output layer (no activation)


class AITrainer:
    """Handles the training and decision making for the snake AI."""

    def __init__(self):
        """Initialize the AI trainer with DQL parameters and models."""
        # Initialize neural networks
        self.model = SnakeAI()  # Main network
        self.target_model = SnakeAI()  # Target network for stable training
        self.target_model.load_state_dict(self.model.state_dict())

        # Training parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=100000)  # Experience replay buffer
        self.batch_size = 64  # Training batch size
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.target_update = 10  # Update target network every N episodes

        # Performance metrics
        self.metrics = {
            "Longest Snake": 1,
            "Average Length": 1,
            "Games Played": 0,
            "Current Length": 1,
        }

        # Model saving/loading
        self.models_dir = "models"
        self.latest_model = None
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:  # Exploration
            return random.randint(0, 3)

        # Exploitation: use model to predict best action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.cpu().argmax().item()

    def train_step(self):
        """Perform one step of training on a batch from replay memory."""
        if len(self.memory) < self.batch_size:
            return

        # Sample and prepare batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Compute Q values and loss
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Update model
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self, episode):
        """Update target network periodically."""
        if episode % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self, snake, food, game_over):
        """Calculate reward for the current state."""
        if game_over:
            return -10  # Penalty for dying
        elif snake.get_head_position() == food.position:
            return 10  # Reward for eating food
        else:
            # Small penalty based on distance to food
            head = snake.get_head_position()
            distance = abs(head[0] - food.position[0]) + abs(head[1] - food.position[1])
            return -0.1 * (distance / (GRID_WIDTH + GRID_HEIGHT))

    def update_metrics(self, length):
        """Update performance metrics after each game."""
        self.metrics["Games Played"] += 1
        self.metrics["Current Length"] = length
        self.metrics["Average Length"] = round(
            (
                self.metrics["Average Length"] * (self.metrics["Games Played"] - 1)
                + length
            )
            / self.metrics["Games Played"],
            2,
        )
        if length > self.metrics["Longest Snake"]:
            self.metrics["Longest Snake"] = length

    def get_metrics(self):
        """Return current performance metrics."""
        return self.metrics

    def get_state(self, snake, food):
        """Convert game state to neural network input format."""
        head = snake.get_head_position()
        # Check for danger in each direction
        danger_straight = self._is_direction_dangerous(snake, snake.direction)
        danger_right = self._is_direction_dangerous(
            snake, self._get_right_direction(snake.direction)
        )
        danger_left = self._is_direction_dangerous(
            snake, self._get_left_direction(snake.direction)
        )

        # Current direction
        dir_l = snake.direction[0] == -1
        dir_r = snake.direction[0] == 1
        dir_u = snake.direction[1] == -1
        dir_d = snake.direction[1] == 1

        # Food location relative to snake
        food_l = food.position[0] < head[0]
        food_r = food.position[0] > head[0]
        food_u = food.position[1] < head[1]
        food_d = food.position[1] > head[1]

        return np.array(
            [
                danger_straight,
                danger_right,
                danger_left,
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                food_l,
                food_r,
                food_u,
                food_d,
            ],
            dtype=np.float32,
        )

    def _is_direction_dangerous(self, snake, direction):
        """Check if moving in a direction would result in collision."""
        head = snake.get_head_position()
        next_pos = (
            (head[0] + direction[0]) % GRID_WIDTH,
            (head[1] + direction[1]) % GRID_HEIGHT,
        )
        return next_pos in snake.positions[1:]

    def _get_right_direction(self, direction):
        """Get the direction vector for a right turn."""
        return (direction[1], -direction[0])

    def _get_left_direction(self, direction):
        """Get the direction vector for a left turn."""
        return (-direction[1], direction[0])

    def save_model(self):
        """Save model and training state to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snake_ai_model_{timestamp}.pth"
        filepath = os.path.join(self.models_dir, filename)

        # Save model and optimizer state
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        metadata = {
            "epsilon": self.epsilon,
            "metrics": self.metrics,
            "timestamp": timestamp,
        }

        # Save to timestamped files
        torch.save(model_data, filepath)
        metadata_path = os.path.join(
            self.models_dir, f"snake_ai_metadata_{timestamp}.pt"
        )
        torch.save(metadata, metadata_path)

        self.latest_model = filepath

        # Update latest model links
        latest_model_link = os.path.join(self.models_dir, "latest_model.pth")
        latest_metadata_link = os.path.join(self.models_dir, "latest_metadata.pt")

        if os.path.exists(latest_model_link):
            os.remove(latest_model_link)
        if os.path.exists(latest_metadata_link):
            os.remove(latest_metadata_link)

        torch.save(model_data, latest_model_link)
        torch.save(metadata, latest_metadata_link)

    def load_model(self):
        """Load latest model and training state from disk."""
        latest_model = os.path.join(self.models_dir, "latest_model.pth")
        latest_metadata = os.path.join(self.models_dir, "latest_metadata.pt")

        if not os.path.exists(latest_model) or not os.path.exists(latest_metadata):
            print("Model files not found")
            return False

        try:
            # Load model state
            checkpoint = torch.load(
                latest_model, map_location=device, weights_only=True
            )
            if "model_state_dict" not in checkpoint:
                print("Invalid model file format")
                return False

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load metadata
            metadata = torch.load(
                latest_metadata, map_location=device, weights_only=True
            )
            if "epsilon" not in metadata:
                print("Invalid metadata file format")
                return False

            self.epsilon = metadata["epsilon"]
            if "metrics" in metadata:
                self.metrics = metadata["metrics"]

            # Update target network
            self.target_model.load_state_dict(self.model.state_dict())
            self.latest_model = latest_model
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False
