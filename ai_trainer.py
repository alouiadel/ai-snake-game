import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from constants import GRID_WIDTH, GRID_HEIGHT
from collections import deque
import random
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SnakeAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(11, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 4)
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


class AITrainer:
    def __init__(self):
        self.model = SnakeAI()
        self.target_model = SnakeAI()
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

        self.metrics = {
            "Longest Snake": 1,
            "Average Length": 1,
            "Games Played": 0,
            "Current Length": 1,
        }

        self.models_dir = "models"
        self.latest_model = None
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.cpu().argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self, episode):
        if episode % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self, snake, food, game_over):
        if game_over:
            return -10
        elif snake.get_head_position() == food.position:
            return 10
        else:
            head = snake.get_head_position()
            distance = abs(head[0] - food.position[0]) + abs(head[1] - food.position[1])
            return -0.1 * (distance / (GRID_WIDTH + GRID_HEIGHT))

    def update_metrics(self, length):
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
        return self.metrics

    def get_state(self, snake, food):
        head = snake.get_head_position()
        danger_straight = self._is_direction_dangerous(snake, snake.direction)
        danger_right = self._is_direction_dangerous(
            snake, self._get_right_direction(snake.direction)
        )
        danger_left = self._is_direction_dangerous(
            snake, self._get_left_direction(snake.direction)
        )

        dir_l = snake.direction[0] == -1
        dir_r = snake.direction[0] == 1
        dir_u = snake.direction[1] == -1
        dir_d = snake.direction[1] == 1

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
        head = snake.get_head_position()
        next_pos = (
            (head[0] + direction[0]) % GRID_WIDTH,
            (head[1] + direction[1]) % GRID_HEIGHT,
        )
        return next_pos in snake.positions[1:]

    def _get_right_direction(self, direction):
        return (direction[1], -direction[0])

    def _get_left_direction(self, direction):
        return (-direction[1], direction[0])

    def save_model(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snake_ai_model_{timestamp}.pth"
        filepath = os.path.join(self.models_dir, filename)

        model_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        metadata = {
            "epsilon": self.epsilon,
            "metrics": self.metrics,
            "timestamp": timestamp,
        }

        torch.save(model_data, filepath)
        metadata_path = os.path.join(
            self.models_dir, f"snake_ai_metadata_{timestamp}.pt"
        )
        torch.save(metadata, metadata_path)

        self.latest_model = filepath

        latest_model_link = os.path.join(self.models_dir, "latest_model.pth")
        latest_metadata_link = os.path.join(self.models_dir, "latest_metadata.pt")

        if os.path.exists(latest_model_link):
            os.remove(latest_model_link)
        if os.path.exists(latest_metadata_link):
            os.remove(latest_metadata_link)

        torch.save(model_data, latest_model_link)
        torch.save(metadata, latest_metadata_link)

    def load_model(self):
        latest_model = os.path.join(self.models_dir, "latest_model.pth")
        latest_metadata = os.path.join(self.models_dir, "latest_metadata.pt")

        if not os.path.exists(latest_model) or not os.path.exists(latest_metadata):
            print("Model files not found")
            return False

        try:
            checkpoint = torch.load(
                latest_model, map_location=device, weights_only=True
            )
            if "model_state_dict" not in checkpoint:
                print("Invalid model file format")
                return False

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            metadata = torch.load(
                latest_metadata, map_location=device, weights_only=True
            )
            if "epsilon" not in metadata:
                print("Invalid metadata file format")
                return False

            self.epsilon = metadata["epsilon"]
            if "metrics" in metadata:
                self.metrics = metadata["metrics"]

            self.target_model.load_state_dict(self.model.state_dict())
            self.latest_model = latest_model
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False
