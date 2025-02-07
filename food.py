"""Food class for the snake game, handles food position and appearance."""

import random
from constants import FOOD_COLOR, GRID_WIDTH, GRID_HEIGHT


class Food:
    """Represents the food item that the snake can eat to grow."""

    def __init__(self):
        """Initialize food with random position and color."""
        self.position = (0, 0)  # Initial position, will be randomized
        self.color = FOOD_COLOR  # Set food color from constants
        self.randomize_position()  # Place food at random position

    def randomize_position(self):
        """Generate a new random position for the food within grid bounds."""
        self.position = (
            random.randint(0, GRID_WIDTH - 1),  # Random x coordinate
            random.randint(0, GRID_HEIGHT - 1),  # Random y coordinate
        )
