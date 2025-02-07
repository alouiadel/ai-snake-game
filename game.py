import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
from constants import (
    CELL_SIZE,
    GRID_HEIGHT,
    GRID_WIDTH,
    BORDER_COLOR,
    TEXT_COLOR,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    BACKGROUND,
)


def draw_grid(screen):
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


def draw_main_menu(screen, font):
    screen.fill(BACKGROUND)

    play_text = font.render("1. Play Game", True, TEXT_COLOR)
    quit_text = font.render("2. Quit", True, TEXT_COLOR)

    play_rect = play_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
    quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 40))

    screen.blit(play_text, play_rect)
    screen.blit(quit_text, quit_rect)
