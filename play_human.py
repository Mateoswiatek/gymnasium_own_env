# play_human.py - Human vs AI game
from MicroCivEnv import MicroCivEnv
import pygame

from enums import ActionType


def main():
    env = MicroCivEnv(render_mode="human")
    observation, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    action = ActionType.END_TURN.value
                    observation, reward, terminated, truncated, info = env.step(action)
                    if terminated:
                        print(f"Game over! Reason: {info.get('reason', 'Unknown')}")
                        running = False
                elif event.key == pygame.K_2:
                    action = ActionType.DO_NOTHING.value
                    env.step(action)
            elif env.process_event(event):
                pass  # Event was handled by the environment

        # Render the environment
        env.render()

    env.close()

if __name__ == "__main__":
    main()