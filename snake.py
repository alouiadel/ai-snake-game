"""Snake class that handles the snake's behavior and properties."""

import random
from constants import SNAKE_COLOR, GRID_WIDTH, GRID_HEIGHT


class Snake:
    """Represents the snake in the game with movement and growth mechanics."""

    def __init__(self):
        """Initialize snake with starting position and random direction."""
        self.length = 1  # Starting length
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]  # Start at center
        self.direction = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0)]
        )  # Random start direction
        self.color = SNAKE_COLOR  # Snake color from constants
        self.score = 1  # Initial score
        self.growth_pending = False  # Flag for growing after eating

    def get_head_position(self):
        """Return the position of snake's head (first element)."""
        return self.positions[0]

    def turn(self, point):
        """Change snake's direction unless it would cause immediate self collision."""
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return  # Prevent 180-degree turns when snake length > 1
        self.direction = point

    def move(self):
        """Update snake's position and handle growth. Returns False if collision occurs."""
        cur = self.get_head_position()
        x, y = self.direction
        new = (
            ((cur[0] + x) % GRID_WIDTH),
            (cur[1] + y) % GRID_HEIGHT,
        )  # Wrap around screen edges

        if new in self.positions[2:]:  # Check for self collision
            return False

        self.positions.insert(0, new)  # Add new head position
        if not self.growth_pending:
            self.positions.pop()  # Remove tail if not growing
        else:
            self.growth_pending = False  # Reset growth flag
        return True

    def grow(self):
        """Increase snake length and score after eating food."""
        self.growth_pending = True  # Set flag for next move
        self.length += 1  # Increase length
        self.score = self.length  # Update score
