from MicroCivEnv import MicroCivEnv
from enums import ActionType
import pygame

def main():
    env = MicroCivEnv(render_mode="human")
    observation, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Keyboard controls for movement/actions
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Movement (WASD)
                elif event.key == pygame.K_w:
                    action = ActionType.MOVE_UP.value
                elif event.key == pygame.K_s:
                    action = ActionType.MOVE_DOWN.value
                elif event.key == pygame.K_a:
                    action = ActionType.MOVE_LEFT.value
                elif event.key == pygame.K_d:
                    action = ActionType.MOVE_RIGHT.value

                # Building actions
                elif event.key == pygame.K_b:
                    action = ActionType.BUILD_CITY.value
                elif event.key == pygame.K_f:
                    action = ActionType.BUILD_FARM.value
                elif event.key == pygame.K_m:
                    action = ActionType.BUILD_MINE.value
                elif event.key == pygame.K_l:
                    action = ActionType.BUILD_LUMBERMILL.value

                # End turn / Do nothing
                elif event.key == pygame.K_e or event.key == pygame.K_1:
                    action = ActionType.END_TURN.value
                elif event.key == pygame.K_2:
                    action = ActionType.DO_NOTHING.value
                else:
                    continue  # Skip unmapped keys

                # Execute action
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    print(f"Game over! Reason: {info.get('reason', 'Unknown')}")
                    running = False

            # Mouse controls (unit/city selection)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left-click
                    x, y = event.pos[0] // env.cell_size, event.pos[1] // env.cell_size
                    if 0 <= x < env.map_size[1] and 0 <= y < env.map_size[0]:
                        env.select_unit(x, y) or env.select_city(x, y)
                elif event.button == 3:  # Right-click (deselect)
                    env.selected_unit_idx = None
                    env.selected_city_idx = None

        # Render the game
        env.render()

    env.close()

if __name__ == "__main__":
    main()