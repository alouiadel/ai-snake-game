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

pygame.init()
mixer.init()


def check_model_exists():
    model_path = os.path.join("models", "latest_model.pth")
    metadata_path = os.path.join("models", "latest_metadata.pt")
    return os.path.exists(model_path) and os.path.exists(metadata_path)


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Snake Game")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    ai_trainer = AITrainer()
    model_exists = check_model_exists()
    if model_exists:
        try:
            model_exists = ai_trainer.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            model_exists = False

    in_menu = True
    in_training_prompt = False
    training = False
    watching_ai = False
    episodes = 0
    current_episode = 0
    input_text = ""
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
                        in_training_prompt = True
                        in_menu = False
                    elif event.key == pygame.K_3 and model_exists:
                        in_menu = False
                        watching_ai = True
                        snake = Snake()
                        food = Food()
                    elif event.key == pygame.K_4:
                        pygame.quit()
                        sys.exit()
                elif in_training_prompt:
                    if event.key == pygame.K_RETURN and input_text:
                        episodes = max(1, min(1000, int(input_text)))
                        training = True
                        in_training_prompt = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.unicode.isnumeric():
                        input_text += event.unicode
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

        screen.fill(BACKGROUND)

        if in_menu:
            draw_main_menu(screen, font, model_exists)
        elif in_training_prompt:
            draw_training_prompt(screen, font)
            text = font.render(input_text, True, TEXT_COLOR)
            rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 40))
            screen.blit(text, rect)
        elif training:
            if current_episode < episodes:
                snake = Snake()
                food = Food()
                steps = 0
                max_steps = 1000
                game_over = False

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

                    ai_trainer.remember(state, action, reward, new_state, game_over)
                    ai_trainer.train_step()

                    steps += 1

                ai_trainer.update_target_network(current_episode)

                ai_trainer.update_metrics(snake.length)
                current_episode += 1

            draw_training_progress(
                screen, font, current_episode, episodes, ai_trainer.get_metrics()
            )
            pygame.display.update()

            if current_episode >= episodes:
                ai_trainer.save_model()
                ai_trainer.model.eval()
                ai_trainer.target_model.load_state_dict(ai_trainer.model.state_dict())
                pygame.time.wait(1000)
                model_exists = check_model_exists()
                training = False
                in_menu = True
                current_episode = 0
                episodes = 0
        elif watching_ai:
            if not game_over:
                state = ai_trainer.get_state(snake, food)
                action = ai_trainer.get_action(state)
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                snake.turn(directions[action])

                if not snake.move():
                    game_over = True
                    waiting_for_input = True

                if snake.get_head_position() == food.position:
                    snake.grow()
                    food.randomize_position()

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
