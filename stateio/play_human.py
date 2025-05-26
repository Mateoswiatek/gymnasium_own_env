from env import GridGame


def main():
    """Funkcja główna programu."""
    # Parametry gry
    grid_size = 20       # Rozmiar siatki NxN
    screen_size = 800    # Rozmiar okna w pikselach
    num_cities = 16       # Liczba miast
    num_players = 2      # Liczba graczy

    # Utworzenie i uruchomienie gry
    game = GridGame(
        seed=50,
        grid_size=grid_size,
        screen_size=screen_size,
        num_cities=num_cities,
        num_players=num_players,
        render_mode="human",
    )

    running = True
    while running:
        current_player = game.players[game.current_player_idx]
        action = current_player.choose_action(game._get_obs())
        if action is None:
            continue
        obs, reward, done, _, info = game.step(action)
        if done:
            print(f"Gra zakończona! Wynik: {reward}")
            running = False


if __name__ == "__main__":
    main()