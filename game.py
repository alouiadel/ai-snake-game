"""Game rendering module for the Snake game."""

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
    SNAKE_COLOR,
)


def draw_grid(screen):
    """Draw the game grid with cell borders."""
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            r = pygame.Rect((x * CELL_SIZE, y * CELL_SIZE), (CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BORDER_COLOR, r, 1)  # Draw cell borders


def draw_play_again_prompt(screen, font, snake):
    """Draw game over screen with score and replay prompt."""
    # Create semi-transparent overlay
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.fill((0, 0, 0))
    overlay.set_alpha(128)
    screen.blit(overlay, (0, 0))

    # Render game over text and score
    game_over_text = font.render("GAME OVER!", True, TEXT_COLOR)
    score_text = font.render(f"Final Length: {snake.score}", True, TEXT_COLOR)
    prompt_text = font.render("Play Again? (Y/N)", True, TEXT_COLOR)

    # Position text elements
    game_over_rect = game_over_text.get_rect(
        center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 40)
    )
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
    prompt_rect = prompt_text.get_rect(
        center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 40)
    )

    # Draw all text elements
    screen.blit(game_over_text, game_over_rect)
    screen.blit(score_text, score_rect)
    screen.blit(prompt_text, prompt_rect)


def draw_main_menu(screen, font, model_exists=False):
    """Draw the main menu with all game options."""
    screen.fill(BACKGROUND)

    # Draw title with shadow effect
    title_font = pygame.font.Font(None, 72)
    shadow_text = title_font.render("AI Snake Game", True, (30, 30, 30))
    title_text = title_font.render("AI Snake Game", True, TEXT_COLOR)

    title_rect = title_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 4))
    shadow_rect = shadow_text.get_rect(
        center=(SCREEN_WIDTH / 2 + 2, SCREEN_HEIGHT / 4 + 2)
    )

    screen.blit(shadow_text, shadow_rect)
    screen.blit(title_text, title_rect)

    # Define menu items and their availability
    menu_items = [
        ("1. Play Game", True),
        ("2. Train AI", True),
        ("3. Watch AI Play", model_exists),
        ("4. Quit", True),
    ]

    spacing = 60
    start_y = SCREEN_HEIGHT / 2 - 40
    mouse_pos = pygame.mouse.get_pos()

    # Draw each menu item with hover effects
    for i, (text, enabled) in enumerate(menu_items):
        y_pos = start_y + (i * spacing)
        color = TEXT_COLOR if enabled else (128, 128, 128)

        menu_text = font.render(text, True, color)
        text_rect = menu_text.get_rect(center=(SCREEN_WIDTH / 2, y_pos))

        # Add hover highlight effect
        if enabled and text_rect.collidepoint(mouse_pos):
            indicator_rect = text_rect.inflate(20, 10)
            pygame.draw.rect(screen, (60, 60, 60), indicator_rect, border_radius=5)
            menu_text = font.render(text, True, (255, 255, 255))

        screen.blit(menu_text, text_rect)


def draw_training_prompt(screen, font):
    """Draw the training episodes input prompt."""
    screen.fill(BACKGROUND)
    text = font.render("Enter number of episodes (1-1000):", True, TEXT_COLOR)
    rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
    screen.blit(text, rect)


def draw_training_progress(screen, font, episode, max_episodes, metrics=None):
    """Draw training progress bar and metrics."""
    # Calculate and draw progress bar
    progress = episode / max_episodes
    bar_width = SCREEN_WIDTH * 0.8
    bar_height = 30
    bar_x = SCREEN_WIDTH * 0.1
    bar_y = SCREEN_HEIGHT / 2

    # Draw progress bar background and fill
    pygame.draw.rect(screen, BORDER_COLOR, (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(
        screen, SNAKE_COLOR, (bar_x, bar_y, bar_width * progress, bar_height)
    )

    # Draw progress percentage
    text = font.render(f"Training Progress: {int(progress * 100)}%", True, TEXT_COLOR)
    rect = text.get_rect(center=(SCREEN_WIDTH / 2, bar_y - 40))
    screen.blit(text, rect)

    # Draw training metrics if available
    if metrics:
        y_offset = bar_y + 60
        # Highlight current length if it's the best so far
        length_color = (
            SNAKE_COLOR
            if metrics["Current Length"] >= metrics["Longest Snake"]
            else TEXT_COLOR
        )
        headers = [
            ("Longest Snake Length", metrics["Longest Snake"], TEXT_COLOR),
            ("Current Snake Length", metrics["Current Length"], length_color),
            ("Average Length", metrics["Average Length"], TEXT_COLOR),
        ]

        # Draw each metric
        for label, value, color in headers:
            metric_text = font.render(f"{label}: {value}", True, color)
            metric_rect = metric_text.get_rect(center=(SCREEN_WIDTH / 2, y_offset))
            screen.blit(metric_text, metric_rect)
            y_offset += 30
