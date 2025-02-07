"""Main entry point for the AI Snake Game with training capabilities."""

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import sys
from pygame import mixer
from constants import BACKGROUND, CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, TEXT_COLOR
from snake import Snake
from food import Food
from game import (
    draw_grid,
    draw_play_again_prompt,
    draw_main_menu,
    draw_training_prompt,
    draw_training_progress,
)
from ai_trainer import AITrainer
import threading
import queue
import time

# Initialize Pygame and mixer
pygame.init()
mixer.init()

# Global training state
training_queue = queue.Queue()
training_status = {"running": False, "progress": 0, "metrics": None}


def training_worker(ai_trainer, episodes, training_status):
    """Background worker function for AI training.

    Args:
        ai_trainer: AITrainer instance
        episodes: Number of training episodes to run
        training_status: Shared dictionary for progress tracking
    """
    for episode in range(episodes):
        if not training_status["running"]:
            break

        # Initialize episode
        snake = Snake()
        food = Food()
        steps = 0
        max_steps = 1000
        game_over = False

        # Run episode
        while not game_over and steps < max_steps:
            state = ai_trainer.get_state(snake, food)
            action = ai_trainer.get_action(state)
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            snake.turn(directions[action])

            if not snake.move():
                game_over = True

            new_state = ai_trainer.get_state(snake, food)
            reward = ai_trainer.calculate_reward(snake, food, game_over)

            if snake.get_head_position() == food.position:
                snake.grow()
                food.randomize_position()

            # Store experience and train
            ai_trainer.remember(state, action, reward, new_state, game_over)
            ai_trainer.train_step()
            steps += 1

        # Update training progress
        ai_trainer.update_target_network(episode)
        ai_trainer.update_metrics(snake.length)
        training_status["progress"] = (episode + 1) / episodes
        training_status["metrics"] = ai_trainer.get_metrics()

        time.sleep(0.001)  # Prevent CPU overload

    # Save final model
    ai_trainer.save_model()
    ai_trainer.model.eval()
    ai_trainer.target_model.load_state_dict(ai_trainer.model.state_dict())
    training_status["running"] = False


def check_model_exists():
    """Check if a trained model exists on disk."""
    model_path = os.path.join("models", "latest_model.pth")
    metadata_path = os.path.join("models", "latest_metadata.pt")
    return os.path.exists(model_path) and os.path.exists(metadata_path)


def main():
    """Main game loop handling menu, gameplay, and AI interactions."""
    # Initialize display and game settings
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Snake Game")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    # Enable responsive controls
    pygame.key.set_repeat(150, 50)

    # Initialize AI components
    ai_trainer = AITrainer()
    model_exists = check_model_exists()
    if model_exists:
        try:
            model_exists = ai_trainer.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            model_exists = False

    # Game state variables
    in_menu = True
    in_training_prompt = False
    training = False
    watching_ai = False
    episodes = 0
    input_text = ""
    snake = None
    food = None
    game_over = False
    waiting_for_input = False
    last_move_time = 0
    base_move_delay = 100
    move_delay = base_move_delay
    training_thread = None

    # Main game loop
    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if training_status["running"]:
                    training_status["running"] = False
                    training_thread.join()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if in_menu:
                    if event.key == pygame.K_1:  # Play game
                        in_menu = False
                        snake = Snake()
                        food = Food()
                    elif event.key == pygame.K_2:  # Train AI
                        in_training_prompt = True
                        in_menu = False
                    elif event.key == pygame.K_3 and model_exists:  # Watch AI
                        in_menu = False
                        watching_ai = True
                        snake = Snake()
                        food = Food()
                        # Set AI to evaluation mode
                        ai_trainer.model.eval()
                        ai_trainer.epsilon = 0
                    elif event.key == pygame.K_4:  # Quit
                        pygame.quit()
                        sys.exit()
                elif in_training_prompt:
                    if event.key == pygame.K_RETURN and input_text:
                        episodes = max(1, min(1000, int(input_text)))
                        training = True
                        in_training_prompt = False
                        # Start training in background
                        training_status["running"] = True
                        training_status["progress"] = 0
                        training_thread = threading.Thread(
                            target=training_worker,
                            args=(ai_trainer, episodes, training_status),
                        )
                        training_thread.start()
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.unicode.isnumeric():
                        input_text += event.unicode
                elif not game_over:  # Handle game controls
                    if event.key == pygame.K_UP:
                        snake.turn((0, -1))
                    elif event.key == pygame.K_DOWN:
                        snake.turn((0, 1))
                    elif event.key == pygame.K_LEFT:
                        snake.turn((-1, 0))
                    elif event.key == pygame.K_RIGHT:
                        snake.turn((1, 0))
                elif waiting_for_input:  # Handle game over input
                    if event.key == pygame.K_y:  # Restart game
                        snake = Snake()
                        food = Food()
                        game_over = False
                        waiting_for_input = False
                    elif event.key == pygame.K_n:  # Return to menu
                        in_menu = True
                        game_over = False
                        waiting_for_input = False
            elif (
                event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
            ):  # Mouse input
                if in_menu:
                    # Handle menu item clicks
                    start_y = SCREEN_HEIGHT / 2 - 40
                    spacing = 60
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    if SCREEN_WIDTH / 2 - 100 <= mouse_x <= SCREEN_WIDTH / 2 + 100:
                        for i in range(4):
                            item_y = start_y + (i * spacing)
                            if item_y - 15 <= mouse_y <= item_y + 15:
                                if i == 0:  # Play game
                                    in_menu = False
                                    snake = Snake()
                                    food = Food()
                                elif i == 1:  # Train AI
                                    in_training_prompt = True
                                    in_menu = False
                                elif i == 2 and model_exists:  # Watch AI
                                    in_menu = False
                                    watching_ai = True
                                    snake = Snake()
                                    food = Food()
                                elif i == 3:  # Quit
                                    pygame.quit()
                                    sys.exit()

        # Clear screen
        screen.fill(BACKGROUND)

        # Render current game state
        if in_menu:
            draw_main_menu(screen, font, model_exists)
        elif in_training_prompt:
            draw_training_prompt(screen, font)
            text = font.render(input_text, True, TEXT_COLOR)
            rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 40))
            screen.blit(text, rect)
        elif training:
            if training_status["running"]:
                draw_training_progress(
                    screen,
                    font,
                    int(training_status["progress"] * episodes),
                    episodes,
                    training_status["metrics"],
                )
            else:
                model_exists = check_model_exists()
                training = False
                in_menu = True
        elif watching_ai:
            if not game_over:
                current_time = pygame.time.get_ticks()
                if current_time - last_move_time >= move_delay:
                    # Get AI decision and move
                    state = ai_trainer.get_state(snake, food)
                    action = ai_trainer.get_action(state)
                    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                    snake.turn(directions[action])

                    if not snake.move():
                        game_over = True
                        waiting_for_input = True

                    last_move_time = current_time

                if snake.get_head_position() == food.position:
                    snake.grow()
                    food.randomize_position()
                    move_delay = max(50, base_move_delay - (snake.score * 1.5))

            # Draw game elements
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
        else:  # Player game mode
            if not game_over:
                current_time = pygame.time.get_ticks()
                if current_time - last_move_time >= move_delay:
                    if not snake.move():
                        game_over = True
                        waiting_for_input = True
                    last_move_time = current_time

                if snake.get_head_position() == food.position:
                    snake.grow()
                    food.randomize_position()
                    move_delay = max(50, base_move_delay - (snake.score * 1.5))

            # Draw game elements
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

        # Update display
        pygame.display.update()
        clock.tick(60)  # Maintain 60 FPS


if __name__ == "__main__":
    main()
