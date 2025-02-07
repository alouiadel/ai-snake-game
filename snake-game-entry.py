import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import sys
from pygame import mixer
from constants import BACKGROUND, CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, TEXT_COLOR
from snake import Snake
from food import Food
from game import draw_grid, draw_play_again_prompt, draw_main_menu

pygame.init()
mixer.init()


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Snake Game")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    in_menu = True
    snake = None
    food = None
    game_over = False
    waiting_for_input = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if in_menu:
                    if event.key == pygame.K_1:
                        in_menu = False
                        snake = Snake()
                        food = Food()
                    elif event.key == pygame.K_2:
                        pygame.quit()
                        sys.exit()
                elif not game_over:
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
                        in_menu = True
                        game_over = False
                        waiting_for_input = False

        if in_menu:
            draw_main_menu(screen, font)
        else:
            if not game_over:
                if not snake.move():
                    game_over = True
                    waiting_for_input = True

                if snake.get_head_position() == food.position:
                    snake.grow()
                    food.randomize_position()

            screen.fill(BACKGROUND)
            draw_grid(screen)

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
