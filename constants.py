"""Game constants and configuration settings."""

# Color scheme using One Dark theme colors
BACKGROUND = (40, 44, 52)  # Dark background color
SNAKE_COLOR = (97, 175, 239)  # Blue snake color
FOOD_COLOR = (224, 108, 117)  # Red food color
BORDER_COLOR = (86, 92, 100)  # Grid line color
TEXT_COLOR = (171, 178, 191)  # Light text color

# Game grid and display settings
CELL_SIZE = 20  # Size of each grid cell in pixels
GRID_WIDTH = 30  # Number of cells horizontally
GRID_HEIGHT = 25  # Number of cells vertically
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH  # Total screen width
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT + 60  # Total screen height with score area
