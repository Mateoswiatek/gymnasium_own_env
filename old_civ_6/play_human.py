from MicroCivEnv import MicroCivEnv
from enums import ActionType, UnitType
import pygame

def main():
    env = MicroCivEnv(render_mode="human")
    observation, _ = env.reset()

    running = True
    human_player_id = 0  # Player 1 is always human

    # Key mappings for all actions
    key_mappings = {
        # Movement
        pygame.K_w: ActionType.MOVE_UP.value,
        pygame.K_s: ActionType.MOVE_DOWN.value,
        pygame.K_a: ActionType.MOVE_LEFT.value,
        pygame.K_d: ActionType.MOVE_RIGHT.value,

        # Tile improvements
        pygame.K_b: ActionType.BUILD_CITY.value,
        pygame.K_f: ActionType.BUILD_FARM.value,
        pygame.K_m: ActionType.BUILD_MINE.value,
        pygame.K_l: ActionType.BUILD_LUMBERMILL.value,

        # City buildings
        pygame.K_h: ActionType.BUILD_HOUSE.value,
        pygame.K_g: ActionType.BUILD_GRANARY.value,
        pygame.K_r: ActionType.BUILD_BARRACKS.value,
        pygame.K_k: ActionType.BUILD_MARKET.value,

        # Unit production
        pygame.K_u: ActionType.CREATE_SETTLER.value,  # 'U' for settler
        pygame.K_v: ActionType.CREATE_WARRIOR.value,  # 'V' for warrior
        pygame.K_c: ActionType.CREATE_WORKER.value,   # 'C' for worker

        # Special actions
        pygame.K_e: ActionType.END_TURN.value,
        pygame.K_1: ActionType.END_TURN.value,
        pygame.K_2: ActionType.DO_NOTHING.value,
    }

    while running:
        current_player = env.players[env.current_player_idx]

        # Handle AI turn automatically
        if current_player.player_id != human_player_id:
            env._execute_ai_turn(current_player)
            env._end_turn()
            continue

        # Human player's turn - process inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Check if pressed key is mapped to an action
                elif event.key in key_mappings:
                    action = key_mappings[event.key]

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

        # Display current controls
        font = pygame.font.SysFont(None, 24)
        controls_text = [
            "Controls:",
            "WASD - Move | B - Build City | F/M/L - Build Farm/Mine/Lumbermill",
            "H/G/R/K - Build House/Granary/Barracks/Market",
            "U/V/C - Create Settler/Warrior/Worker",
            "E/1 - End Turn | 2 - Do Nothing | ESC - Quit"
        ]

        # Render the game
        env.render()

        # Draw controls help
        y_offset = env.map_size[0] * env.cell_size + 10
        for i, text in enumerate(controls_text):
            text_surface = font.render(text, True, (255, 255, 255))
            env.window.blit(text_surface, (10, y_offset + i * 25))

        pygame.display.flip()

    env.close()

if __name__ == "__main__":
    main()