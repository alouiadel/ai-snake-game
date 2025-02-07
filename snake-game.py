import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import random
import sys
from pygame import mixer

pygame.init()
mixer.init()

BACKGROUND = (40, 44, 52)
SNAKE_COLOR = (97, 175, 239)
FOOD_COLOR = (224, 108, 117)
BORDER_COLOR = (86, 92, 100)
TEXT_COLOR = (171, 178, 191)

CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 25
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT + 60

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Snake Game")
clock = pygame.time.Clock()


class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.color = SNAKE_COLOR
        self.score = 0
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
        self.score += 10


class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = FOOD_COLOR
        self.randomize_position()

    def randomize_position(self):
        self.position = (
            random.randint(0, GRID_WIDTH - 1),
            random.randint(0, GRID_HEIGHT - 1),
        )


def draw_grid():
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            r = pygame.Rect((x * CELL_SIZE, y * CELL_SIZE), (CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BORDER_COLOR, r, 1)


def draw_play_again_prompt(screen, font, snake):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.fill((0, 0, 0))
    overlay.set_alpha(128)
    screen.blit(overlay, (0, 0))

    game_over_text = font.render("GAME OVER!", True, TEXT_COLOR)
    score_text = font.render(f"Final Score: {snake.score}", True, TEXT_COLOR)
    prompt_text = font.render("Play Again? (Y/N)", True, TEXT_COLOR)

    game_over_rect = game_over_text.get_rect(
        center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 40)
    )
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
    prompt_rect = prompt_text.get_rect(
        center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 40)
    )

    screen.blit(game_over_text, game_over_rect)
    screen.blit(score_text, score_rect)
    screen.blit(prompt_text, prompt_rect)


def main():
    snake = Snake()
    food = Food()
    game_over = False
    waiting_for_input = False

    font = pygame.font.Font(None, 36)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if not game_over:
                    if event.key == pygame.K_UP:
                        snake.turn((0, -1))
                    elif event.key == pygame.K_DOWN:
                        snake.turn((0, 1))
                    elif event.key == pygame.K_LEFT:
                        snake.turn((-1, 0))
                    elif event.key == pygame.K_RIGHT:
                        snake.turn((1, 0))
                elif waiting_for_input:
                    if event.key == pygame.K_y:
                        snake = Snake()
                        food = Food()
                        game_over = False
                        waiting_for_input = False
                    elif event.key == pygame.K_n:
                        pygame.quit()
                        sys.exit()

        if not game_over:
            if not snake.move():
                game_over = True
                waiting_for_input = True

            if snake.get_head_position() == food.position:
                snake.grow()
                food.randomize_position()

        screen.fill(BACKGROUND)
        draw_grid()

        food_rect = pygame.Rect(
            food.position[0] * CELL_SIZE,
            food.position[1] * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(screen, food.color, food_rect)

        for pos in snake.positions:
            snake_rect = pygame.Rect(
                pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, snake.color, snake_rect)

        score_text = font.render(f"Score: {snake.score}", True, TEXT_COLOR)
        screen.blit(score_text, (10, SCREEN_HEIGHT - 40))

        if game_over and waiting_for_input:
            draw_play_again_prompt(screen, font, snake)

        pygame.display.update()
        clock.tick(10)


if __name__ == "__main__":
    main()
