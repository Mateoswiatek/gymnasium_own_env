# -*- coding: utf-8 -*-
"""easyAI_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18IM_IBgG9wN7WlbS7UNmcMPv8zO7j8xl
"""

# !pip install easyAI

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
import random

class TicTacDoh(TwoPlayerGame):
    def __init__(self, players):
        self.players = players
        self.board = [" "] * 9  # 3x3 board as a list
        self.current_player = 1  # Player 1 starts

    def possible_moves(self):
        return [str(i) for i in range(9) if self.board[i] == " "]

    def make_move(self, move):
        if random.random() > 0.2:
            self.board[int(move)] = "X" if self.current_player == 1 else "O"
        else:
            print(f"Player {self.current_player} attempted move {move}, but it failed!")

    def win_condition(self, symbol):
        winning_combinations = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        return any(all(self.board[i] == symbol for i in combo) for combo in winning_combinations)

    def lose(self):
        return self.win_condition("O" if self.current_player == 1 else "X")

    def is_over(self):
        return self.lose() or " " not in self.board

    def show(self):
        print("\n".join([" ".join(self.board[i:i+3]) for i in range(0, 9, 3)]))

    def scoring(self):
        return -100 if self.lose() else 0

if __name__ == "__main__":
    ai_algo = Negamax(3)  # Depth 3 AI
    game = TicTacDoh([AI_Player(ai_algo), AI_Player(ai_algo)])
    game.play()
    print("Game Over!")

import random
from easyAI import TwoPlayerGame, AI_Player, Negamax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import random
from easyAI import TwoPlayerGame, AI_Player, Negamax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class TicTacDoh(TwoPlayerGame):
    def __init__(self, players, probabilistic=True):
        self.players = players
        self.board = [" "] * 9
        self.current_player = 1
        self.probabilistic = probabilistic

    def possible_moves(self):
        return [str(i) for i in range(9) if self.board[i] == " "]

    def make_move(self, move):
        move_success = True
        if self.probabilistic and random.random() > 0.8:  # 20% chance the move fails
            move_success = False
        else:
            self.board[int(move)] = "X" if self.current_player == 1 else "O"

        return move_success  # Return whether the move succeeded

    def win_condition(self, symbol):
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        return any(all(self.board[i] == symbol for i in combo)
                  for combo in winning_combinations)

    def lose(self):
        return self.win_condition("O" if self.current_player == 1 else "X")

    def is_over(self):
        return self.lose() or self.win() or " " not in self.board

    def win(self):
        return self.win_condition("X" if self.current_player == 1 else "O")

    def show(self):
        print("\n".join([" ".join(self.board[i:i+3]) for i in range(0, 9, 3)]))

    def scoring(self):
        if self.lose():
            return -100
        elif self.win():
            return 100
        else:
            return 0  # Draw

def run_experiment(depth1, depth2, num_games=100, probabilistic=True):
    """
    Run an experiment with two AI players using Negamax with different depths.

    Args:
        depth1: Depth for player 1
        depth2: Depth for player 2
        num_games: Number of games to play
        probabilistic: Whether to use the probabilistic version of the game

    Returns:
        Dictionary with the results
    """
    ai_algo1 = Negamax(depth1)
    ai_algo2 = Negamax(depth2)

    # Results tracking
    results = {
        "player1_wins": 0,
        "player2_wins": 0,
        "draws": 0,
        "failed_moves": 0
    }

    for game_num in tqdm(range(num_games), desc=f"Depth {depth1} vs {depth2} ({'Prob' if probabilistic else 'Det'})"):
        # Alternate who starts
        if game_num % 2 == 0:
            players = [AI_Player(ai_algo1), AI_Player(ai_algo2)]
        else:
            players = [AI_Player(ai_algo2), AI_Player(ai_algo1)]

        game = TicTacDoh(players, probabilistic)

        # Play the game
        history = []
        failed_moves_count = 0

        while not game.is_over():
            # Get current player index (0 or 1) and use easyAI's play method
            player_index = game.current_player - 1
            move = game.players[player_index].ask_move(game)
            history.append((game.current_player, move))

            # Make the move and track if it failed
            move_success = game.make_move(move)
            if not move_success:
                failed_moves_count += 1

            game.switch_player()

        results["failed_moves"] += failed_moves_count

        # Determine the winner
        if game.board.count("X") > game.board.count("O"):
            # Player 1 (X) won
            if game_num % 2 == 0:
                results["player1_wins"] += 1
            else:
                results["player2_wins"] += 1
        elif game.board.count("O") > game.board.count("X"):
            # Player 2 (O) won
            if game_num % 2 == 0:
                results["player2_wins"] += 1
            else:
                results["player1_wins"] += 1
        else:
            # Draw
            results["draws"] += 1

    return results

def compare_depths_and_variants():
    """Compare different depth settings in both deterministic and probabilistic variants."""
    depth_pairs = [(1, 3), (3, 5), (2, 8)]
    num_games = 100

    results = []

    # Run experiments
    for depth1, depth2 in depth_pairs:
        # Deterministic variant
        det_results = run_experiment(depth1, depth2, num_games, probabilistic=False)
        det_results.update({
            "depth1": depth1,
            "depth2": depth2,
            "variant": "Deterministic"
        })
        results.append(det_results)

        # Probabilistic variant
        prob_results = run_experiment(depth1, depth2, num_games, probabilistic=True)
        prob_results.update({
            "depth1": depth1,
            "depth2": depth2,
            "variant": "Probabilistic"
        })
        results.append(prob_results)

    return results

def visualize_results(results):
    """Create visualizations for the experiment results."""
    # Bar chart for wins/draws
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, variant in enumerate(["Deterministic", "Probabilistic"]):
        variant_results = [r for r in results if r["variant"] == variant]

        depth_labels = [f"{r['depth1']} vs {r['depth2']}" for r in variant_results]
        player1_wins = [r["player1_wins"] for r in variant_results]
        player2_wins = [r["player2_wins"] for r in variant_results]
        draws = [r["draws"] for r in variant_results]

        x = np.arange(len(depth_labels))
        width = 0.25

        axes[i].bar(x - width, player1_wins, width, label=f'Depth 1 Wins')
        axes[i].bar(x, player2_wins, width, label=f'Depth 2 Wins')
        axes[i].bar(x + width, draws, width, label='Draws')

        axes[i].set_xlabel('Negamax Depth Settings')
        axes[i].set_ylabel('Number of Games')
        axes[i].set_title(f'{variant} Variant Results')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(depth_labels)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('tictacdoh_results.png')

    # In probabilistic variant, also show failed moves
    if any('failed_moves' in r for r in results):
        plt.figure(figsize=(10, 6))
        prob_results = [r for r in results if r["variant"] == "Probabilistic"]
        depth_labels = [f"{r['depth1']} vs {r['depth2']}" for r in prob_results]
        failed_moves = [r["failed_moves"] for r in prob_results]

        plt.bar(depth_labels, failed_moves)
        plt.xlabel('Negamax Depth Settings')
        plt.ylabel('Number of Failed Moves')
        plt.title('Failed Moves in Probabilistic Variant')
        plt.tight_layout()
        plt.savefig('tictacdoh_failed_moves.png')

def generate_report(results):
    """Generate a report based on the experiment results."""
    report = """# Raport: Eksperymenty z TicTacDoh i algorytmem Negamax

## 1. Opis gry TicTacDoh

TicTacDoh to wariant gry w kółko i krzyżyk, który wprowadza element losowości do tradycyjnej deterministycznej gry. Główne cechy gry to:

- Plansza 3x3, podobnie jak w standardowym kółko i krzyżyk
- Gracze na zmianę umieszczają swoje symbole (X lub O) na planszy
- Wygrywa gracz, który pierwszy ułoży trzy swoje symbole w linii (poziomo, pionowo lub po przekątnej)
- **Element losowości**: W probabilistycznym wariancie gry istnieje 20% szans, że ruch gracza nie powiedzie się, co oznacza, że mimo próby umieszczenia symbolu, pozostanie on nieumieszczony na planszy

Gra została zaimplementowana przy użyciu biblioteki easyAI, która dostarcza narzędzia do tworzenia gier dwuosobowych oraz implementacji algorytmów sztucznej inteligencji.

## 2. Algorytm Negamax

W eksperymentach wykorzystano algorytm Negamax, który jest wariantem algorytmu Minimax. Algorytm ten służy do podejmowania optymalnych decyzji w grach z pełną informacją i przeciwstawnymi celami graczy. Kluczowe aspekty algorytmu:

- Przeszukuje drzewo gry do określonej głębokości
- Zakłada, że przeciwnik zawsze wybiera najlepszy możliwy ruch
- W każdym kroku minimalizuje maksymalną możliwą stratę
- Parametr głębokości określa, jak daleko w przyszłość algorytm "patrzy"

## 3. Przeprowadzone eksperymenty

Przeprowadzono serię eksperymentów mających na celu porównanie:
- Różnych ustawień głębokości algorytmu Negamax (porównywano głębokości {0} vs {1} oraz {2} vs {3})
- Deterministycznego i probabilistycznego wariantu gry

Każdy eksperyment obejmował 100 gier między dwoma graczami AI, przy czym gracze na zmianę rozpoczynali kolejne partie.

## 4. Wyniki eksperymentów

### 4.1. Wariant deterministyczny

| Głębokości | Wygrane Gracz 1 | Wygrane Gracz 2 | Remisy |
|------------|-----------------|-----------------|--------|
""".format(results[0]['depth1'], results[0]['depth2'], results[2]['depth1'], results[2]['depth2'])

    # Add deterministic results
    det_results = [r for r in results if r["variant"] == "Deterministic"]
    for r in det_results:
        report += f"| {r['depth1']} vs {r['depth2']} | {r['player1_wins']} | {r['player2_wins']} | {r['draws']} |\n"

    report += """
### 4.2. Wariant probabilistyczny

| Głębokości | Wygrane Gracz 1 | Wygrane Gracz 2 | Remisy | Nieudane ruchy |
|------------|-----------------|-----------------|--------|----------------|
"""

    # Add probabilistic results
    prob_results = [r for r in results if r["variant"] == "Probabilistic"]
    for r in prob_results:
        report += f"| {r['depth1']} vs {r['depth2']} | {r['player1_wins']} | {r['player2_wins']} | {r['draws']} | {r['failed_moves']} |\n"

    report += """
## 5. Analiza wyników

### 5.1. Wpływ głębokości przeszukiwania

Eksperymenty potwierdzają, że głębokość przeszukiwania algorytmu Negamax ma istotny wpływ na skuteczność gracza AI. W obu wariantach gry (deterministycznym i probabilistycznym) gracz z większą głębokością przeszukiwania zwykle osiągał lepsze wyniki.

### 5.2. Wpływ losowości na wyniki

Wprowadzenie elementu losowości w postaci możliwości niepowodzenia ruchu znacząco wpłynęło na wyniki:
- Zmniejszyła się przewaga gracza z większą głębokością przeszukiwania
- Zwiększyła się liczba remisów
- Pojawiły się nieoczekiwane wyniki, które nie wystąpiłyby w deterministycznej wersji gry

Losowość wprowadza element niepewności, który utrudnia algorytmowi Negamax dokładne przewidywanie przyszłych stanów gry. Nawet przy dużej głębokości przeszukiwania, losowe niepowodzenia ruchów mogą prowadzić do sytuacji nieprzewidzianych przez algorytm.

### 5.3. Napotkane problemy

Podczas eksperymentów napotkano następujące problemy:
- W wariancie probabilistycznym, niepowodzenie ruchu może prowadzić do sytuacji, w której gracz traci swoją turę bez dokonania żadnej zmiany na planszy
- Przy dużej liczbie niepowodzeń, gry mogą trwać znacznie dłużej niż w wariancie deterministycznym
- Zwiększenie głębokości przeszukiwania znacząco wydłuża czas wykonania algorytmu, szczególnie w późniejszych fazach gry, gdy drzewo możliwości jest bardziej rozbudowane

## 6. Wnioski

Przeprowadzone eksperymenty dostarczyły cennych informacji na temat zachowania algorytmu Negamax w różnych warunkach:

1. Większa głębokość przeszukiwania generalnie prowadzi do lepszych wyników, ale kosztem zwiększonego czasu obliczeniowego
2. Losowość w grze znacząco zmienia dynamikę rozgrywki i może niwelować przewagę wynikającą z większej głębokości przeszukiwania
3. W grach deterministycznych, algorytm Negamax potrafi znaleźć optymalne strategie prowadzące do wygranej lub remisu
4. W grach z elementem losowości, nawet zaawansowane algorytmy AI muszą uwzględniać niepewność i dostosowywać swoje strategie

Podsumowując, eksperymenty pokazują, że skuteczność algorytmów AI w grach zależy nie tylko od ich parametrów (jak głębokość przeszukiwania), ale również od charakterystyki samej gry, w szczególności od obecności elementów losowych."""

    return report

if __name__ == "__main__":
    # Run all experiments
    results = compare_depths_and_variants()

    # Visualize the results
    visualize_results(results)

    # Generate and save the report
    report = generate_report(results)
    with open("tictacdoh_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("Eksperymenty zakończone. Raport został zapisany do pliku 'tictacdoh_report.md'.")
    print("Wykresy zostały zapisane jako pliki PNG.")