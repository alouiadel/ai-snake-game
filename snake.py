import random
from constants import SNAKE_COLOR, GRID_WIDTH, GRID_HEIGHT


class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.color = SNAKE_COLOR
        self.score = 1
        self.growth_pending = False

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        self.direction = point

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + x) % GRID_WIDTH), (cur[1] + y) % GRID_HEIGHT)
        if new in self.positions[2:]:
            return False

        self.positions.insert(0, new)
        if not self.growth_pending:
            self.positions.pop()
        else:
            self.growth_pending = False
        return True

    def grow(self):
        self.growth_pending = True
        self.length += 1
        self.score = self.length
