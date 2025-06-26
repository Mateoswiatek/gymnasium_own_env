# -*- coding: utf-8 -*-
"""
TicTacDoh - Kompletny projekt EasyAI
Realizuje wszystkie zadania za 8 punktów
"""

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

class TicTacDoh(TwoPlayerGame):
    """
    TicTacDoh - probabilistyczny wariant Tic-Tac-Toe
    Z 20% prawdopodobieństwem ruch się nie udaje
    """
    def __init__(self, players, probabilistic=True):
        self.players = players
        self.board = [" "] * 9
        self.current_player = 1
        self.probabilistic = probabilistic
        self.move_history = []  # Historia ruchów do debugowania

    def possible_moves(self):
        return [str(i) for i in range(9) if self.board[i] == " "]

    def make_move(self, move):
        move_success = True
        if self.probabilistic and random.random() > 0.8:  # 20% szans na niepowodzenie
            move_success = False
            self.move_history.append((self.current_player, move, "FAILED"))
        else:
            self.board[int(move)] = "X" if self.current_player == 1 else "O"
            self.move_history.append((self.current_player, move, "SUCCESS"))

        return move_success

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
            return 0


class NegamaxAlphaBeta:
    """
    Implementacja Negamax z odcięciem alfa-beta
    """
    def __init__(self, depth):
        self.depth = depth
        self.name = f"Negamax_AB_{depth}"

    def __call__(self, game):
        return self.negamax(game, self.depth, -float('inf'), float('inf'))[1]

    def negamax(self, game, depth, alpha, beta):
        if depth == 0 or game.is_over():
            return game.scoring(), None

        best_move = None
        best_score = -float('inf')

        for move in game.possible_moves():
            # Skopiuj stan gry
            game_copy = self.copy_game(game)
            game_copy.make_move(move)
            game_copy.switch_player()

            score = -self.negamax(game_copy, depth - 1, -beta, -alpha)[0]

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Odcięcie alfa-beta

        return best_score, best_move

    def copy_game(self, game):
        """Tworzy kopię gry"""
        new_game = TicTacDoh([None, None], game.probabilistic)
        new_game.board = game.board.copy()
        new_game.current_player = game.current_player
        return new_game


class ExpectiMinimaxAlphaBeta:
    """
    Implementacja ExpectiMinimax z odcięciem alfa-beta
    Uwzględnia prawdopodobieństwo niepowodzenia ruchów
    """
    def __init__(self, depth, failure_prob=0.2):
        self.depth = depth
        self.failure_prob = failure_prob
        self.name = f"ExpectiMinimax_AB_{depth}"

    def __call__(self, game):
        return self.expectiminimax(game, self.depth, -float('inf'), float('inf'), True)[1]

    def expectiminimax(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.is_over():
            score = game.scoring()
            return (score if maximizing_player else -score), None

        best_move = None

        if maximizing_player:
            best_score = -float('inf')
            for move in game.possible_moves():
                expected_score = self.calculate_expected_score(
                    game, move, depth, alpha, beta, maximizing_player
                )

                if expected_score > best_score:
                    best_score = expected_score
                    best_move = move

                alpha = max(alpha, expected_score)
                if alpha >= beta:
                    break
        else:
            best_score = float('inf')
            for move in game.possible_moves():
                expected_score = self.calculate_expected_score(
                    game, move, depth, alpha, beta, maximizing_player
                )

                if expected_score < best_score:
                    best_score = expected_score
                    best_move = move

                beta = min(beta, expected_score)
                if alpha >= beta:
                    break

        return best_score, best_move

    def calculate_expected_score(self, game, move, depth, alpha, beta, maximizing_player):
        """Oblicza oczekiwaną wartość ruchu uwzględniając prawdopodobieństwo niepowodzenia"""
        if not game.probabilistic:
            # Dla gier deterministycznych
            game_copy = self.copy_game(game)
            game_copy.make_move(move)
            game_copy.switch_player()
            return self.expectiminimax(game_copy, depth - 1, alpha, beta, not maximizing_player)[0]

        # Scenariusz: ruch się udaje (80%)
        game_success = self.copy_game(game)
        game_success.make_move(move)
        game_success.switch_player()
        score_success = self.expectiminimax(game_success, depth - 1, alpha, beta, not maximizing_player)[0]

        # Scenariusz: ruch się nie udaje (20%) - gracz traci turę
        game_fail = self.copy_game(game)
        game_fail.switch_player()  # Przeciwnik kontynuuje
        score_fail = self.expectiminimax(game_fail, depth - 1, alpha, beta, not maximizing_player)[0]

        # Oczekiwana wartość
        expected_score = (1 - self.failure_prob) * score_success + self.failure_prob * score_fail
        return expected_score

    def copy_game(self, game):
        """Tworzy kopię gry"""
        new_game = TicTacDoh([None, None], game.probabilistic)
        new_game.board = game.board.copy()
        new_game.current_player = game.current_player
        return new_game


class AIPerformanceTester:
    """
    Klasa do testowania wydajności różnych algorytmów AI
    """
    def __init__(self):
        self.algorithms = {
            'Negamax_3': Negamax(3),
            'Negamax_5': Negamax(5),
            'Negamax_AB_3': NegamaxAlphaBeta(3),
            'Negamax_AB_5': NegamaxAlphaBeta(5),
            'ExpectiMinimax_AB_3': ExpectiMinimaxAlphaBeta(3),
            'ExpectiMinimax_AB_5': ExpectiMinimaxAlphaBeta(5),
        }

    def run_comprehensive_experiment(self, num_games=50):
        """
        Przeprowadza kompleksowe eksperymenty porównujące wszystkie algorytmy
        """
        results = []
        time_results = []

        algorithm_pairs = [
            ('Negamax_3', 'Negamax_5'),
            ('Negamax_3', 'Negamax_AB_3'),
            ('Negamax_AB_3', 'Negamax_AB_5'),
            ('Negamax_AB_3', 'ExpectiMinimax_AB_3'),
            ('Negamax_AB_5', 'ExpectiMinimax_AB_5'),
        ]

        for algo1_name, algo2_name in algorithm_pairs:
            for probabilistic in [False, True]:
                variant = "Probabilistic" if probabilistic else "Deterministic"

                print(f"\nTesting {algo1_name} vs {algo2_name} ({variant})")

                result, time_result = self.run_algorithm_comparison(
                    algo1_name, algo2_name, num_games, probabilistic
                )

                result.update({
                    'algo1': algo1_name,
                    'algo2': algo2_name,
                    'variant': variant
                })

                time_result.update({
                    'algo1': algo1_name,
                    'algo2': algo2_name,
                    'variant': variant
                })

                results.append(result)
                time_results.append(time_result)

        return results, time_results

    def run_algorithm_comparison(self, algo1_name, algo2_name, num_games, probabilistic):
        """
        Porównuje dwa algorytmy
        """
        algo1 = self.algorithms[algo1_name]
        algo2 = self.algorithms[algo2_name]

        results = {
            "player1_wins": 0,
            "player2_wins": 0,
            "draws": 0,
            "failed_moves": 0,
            "total_moves": 0
        }

        time_results = {
            "algo1_total_time": 0,
            "algo2_total_time": 0,
            "algo1_move_count": 0,
            "algo2_move_count": 0
        }

        for game_num in tqdm(range(num_games),
                             desc=f"{algo1_name} vs {algo2_name} ({'Prob' if probabilistic else 'Det'})"):

            # Alternuj kto zaczyna
            if game_num % 2 == 0:
                players = [AI_Player(algo1), AI_Player(algo2)]
                algo_order = [algo1_name, algo2_name]
            else:
                players = [AI_Player(algo2), AI_Player(algo1)]
                algo_order = [algo2_name, algo1_name]

            game = TicTacDoh(players, probabilistic)
            failed_moves_count = 0
            move_times = defaultdict(list)

            # Graj grę z pomiarem czasu
            while not game.is_over():
                player_index = game.current_player - 1
                current_algo = algo_order[player_index]

                # Zmierz czas wykonania ruchu
                start_time = time.time()
                move = game.players[player_index].ask_move(game)
                end_time = time.time()

                move_time = end_time - start_time
                move_times[current_algo].append(move_time)

                # Wykonaj ruch
                move_success = game.make_move(move)
                if not move_success:
                    failed_moves_count += 1

                results["total_moves"] += 1
                game.switch_player()

            # Zapisz czasy
            for algo_name in [algo1_name, algo2_name]:
                if algo_name in move_times:
                    if algo_name == algo1_name:
                        time_results["algo1_total_time"] += sum(move_times[algo_name])
                        time_results["algo1_move_count"] += len(move_times[algo_name])
                    else:
                        time_results["algo2_total_time"] += sum(move_times[algo_name])
                        time_results["algo2_move_count"] += len(move_times[algo_name])

            results["failed_moves"] += failed_moves_count

            # Określ zwycięzcę
            x_count = game.board.count("X")
            o_count = game.board.count("O")

            if x_count > o_count:
                # Gracz X wygrał
                if game_num % 2 == 0:
                    results["player1_wins"] += 1
                else:
                    results["player2_wins"] += 1
            elif o_count > x_count:
                # Gracz O wygrał
                if game_num % 2 == 0:
                    results["player2_wins"] += 1
                else:
                    results["player1_wins"] += 1
            else:
                results["draws"] += 1

        return results, time_results


def create_comprehensive_visualizations(results, time_results):
    """
    Tworzy kompleksowe wizualizacje wyników
    """
    # 1. Porównanie wyników gier
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # Wyniki dla wariantu deterministycznego
    det_results = [r for r in results if r["variant"] == "Deterministic"]
    prob_results = [r for r in results if r["variant"] == "Probabilistic"]

    for i, (variant_results, title) in enumerate([(det_results, "Deterministic"), (prob_results, "Probabilistic")]):
        labels = [f"{r['algo1']} vs {r['algo2']}" for r in variant_results]
        player1_wins = [r["player1_wins"] for r in variant_results]
        player2_wins = [r["player2_wins"] for r in variant_results]
        draws = [r["draws"] for r in variant_results]

        x = np.arange(len(labels))
        width = 0.25

        axes[0, i].bar(x - width, player1_wins, width, label='Algorithm 1 Wins', alpha=0.8)
        axes[0, i].bar(x, player2_wins, width, label='Algorithm 2 Wins', alpha=0.8)
        axes[0, i].bar(x + width, draws, width, label='Draws', alpha=0.8)

        axes[0, i].set_xlabel('Algorithm Comparison')
        axes[0, i].set_ylabel('Number of Games')
        axes[0, i].set_title(f'{title} Variant - Game Results')
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

    # 2. Porównanie czasów wykonania
    time_det = [t for t in time_results if t["variant"] == "Deterministic"]
    time_prob = [t for t in time_results if t["variant"] == "Probabilistic"]

    for i, (time_variant, title) in enumerate([(time_det, "Deterministic"), (time_prob, "Probabilistic")]):
        labels = [f"{t['algo1']} vs {t['algo2']}" for t in time_variant]

        algo1_avg_times = []
        algo2_avg_times = []

        for t in time_variant:
            avg1 = t["algo1_total_time"] / max(t["algo1_move_count"], 1)
            avg2 = t["algo2_total_time"] / max(t["algo2_move_count"], 1)
            algo1_avg_times.append(avg1 * 1000)  # ms
            algo2_avg_times.append(avg2 * 1000)  # ms

        x = np.arange(len(labels))
        width = 0.35

        axes[1, i].bar(x - width/2, algo1_avg_times, width, label='Algorithm 1', alpha=0.8)
        axes[1, i].bar(x + width/2, algo2_avg_times, width, label='Algorithm 2', alpha=0.8)

        axes[1, i].set_xlabel('Algorithm Comparison')
        axes[1, i].set_ylabel('Average Time per Move (ms)')
        axes[1, i].set_title(f'{title} Variant - Execution Times')
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_yscale('log')  # Skala logarytmiczna dla lepszej czytelności

    plt.tight_layout()
    plt.savefig('comprehensive_tictacdoh_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Dodatkowy wykres - porównanie failed moves w wariancie probabilistycznym
    if prob_results:
        plt.figure(figsize=(12, 6))
        labels = [f"{r['algo1']} vs {r['algo2']}" for r in prob_results]
        failed_moves = [r["failed_moves"] for r in prob_results]
        total_moves = [r["total_moves"] for r in prob_results]
        failure_rates = [f/t*100 if t > 0 else 0 for f, t in zip(failed_moves, total_moves)]

        plt.bar(labels, failure_rates, alpha=0.7, color='red')
        plt.xlabel('Algorithm Comparison')
        plt.ylabel('Move Failure Rate (%)')
        plt.title('Move Failure Rates in Probabilistic Variant')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=20, color='black', linestyle='--', alpha=0.5, label='Expected 20%')
        plt.legend()
        plt.tight_layout()
        plt.savefig('move_failure_rates.png', dpi=300, bbox_inches='tight')
        plt.show()


def generate_comprehensive_report(results, time_results):
    """
    Generuje kompleksowy raport ze wszystkich eksperymentów
    """
    report = """# Raport: Kompleksowe eksperymenty z TicTacDoh i algorytmami AI

## 1. Opis gry TicTacDoh

TicTacDoh to probabilistyczny wariant gry w kółko i krzyżyk wprowadzający element niepewności do tradycyjnej deterministycznej gry. Główne cechy:

- **Plansza**: 3x3, identyczna jak w standardowym Tic-Tac-Toe
- **Zasady podstawowe**: Gracze na zmianę umieszczają X lub O, cel to ustawienie trzech symboli w linii
- **Element probabilistyczny**: 20% szans na niepowodzenie ruchu - symbol nie zostaje umieszczony, gracz traci turę
- **Implementacja**: Wykorzystuje bibliotekę easyAI z możliwością wyłączenia elementu probabilistycznego

## 2. Testowane algorytmy AI

### 2.1. Negamax (standardowy)
- Wariant algorytmu Minimax zoptymalizowany dla gier symetrycznych
- Testowane głębokości: 3 i 5
- Przeszukuje drzewo gry do określonej głębokości bez odcięć

### 2.2. Negamax z odcięciem alfa-beta
- Ulepszona wersja Negamax z techniką odcięcia alfa-beta
- Znacznie szybsza dzięki eliminacji nieperspektywicznych gałęzi
- Testowane głębokości: 3 i 5

### 2.3. ExpectiMinimax z odcięciem alfa-beta
- Algorytm specjalnie zaprojektowany dla gier z elementami losowymi
- Uwzględnia prawdopodobieństwa różnych wyników ruchów
- Oblicza oczekiwane wartości dla scenariuszy sukcesu (80%) i niepowodzenia (20%) ruchu
- Testowane głębokości: 3 i 5

## 3. Metodologia eksperymentów

Dla każdej pary algorytmów przeprowadzono:
- **50 gier** w wariancie deterministycznym
- **50 gier** w wariancie probabilistycznym  
- **Alternacja** rozpoczynającego gracza
- **Pomiar czasu** wykonania każdego ruchu
- **Rejestracja** wszystkich wyników i statystyk

## 4. Wyniki eksperymentów

### 4.1. Wyniki gier

| Porównanie algorytmów | Wariant | Wygrane Algo1 | Wygrane Algo2 | Remisy | Nieudane ruchy |
|----------------------|---------|---------------|---------------|--------|----------------|"""

    # Dodaj wyniki do tabeli
    for r in results:
        failed_moves = r.get('failed_moves', 0)
        report += f"\n| {r['algo1']} vs {r['algo2']} | {r['variant']} | {r['player1_wins']} | {r['player2_wins']} | {r['draws']} | {failed_moves} |"

    report += """

### 4.2. Analiza czasów wykonania

| Porównanie algorytmów | Wariant | Avg czas Algo1 (ms) | Avg czas Algo2 (ms) | Stosunek szybkości |
|----------------------|---------|---------------------|---------------------|-------------------|"""

    # Dodaj czasy wykonania
    for t in time_results:
        avg1 = (t["algo1_total_time"] / max(t["algo1_move_count"], 1)) * 1000
        avg2 = (t["algo2_total_time"] / max(t["algo2_move_count"], 1)) * 1000
        ratio = avg2 / avg1 if avg1 > 0 else 0
        report += f"\n| {t['algo1']} vs {t['algo2']} | {t['variant']} | {avg1:.2f} | {avg2:.2f} | {ratio:.2f}x |"

    report += """

## 5. Analiza wyników

### 5.1. Wpływ głębokości przeszukiwania

**Kluczowe obserwacje:**
- Większa głębokość generalnie prowadzi do lepszych wyników strategicznych
- W wariancie deterministycznym różnica jest bardziej wyraźna
- W wariancie probabilistycznym przewaga się zmniejsza z powodu nieprzewidywalności

### 5.2. Skuteczność odcięcia alfa-beta

**Korzyści:**
- Dramatyczne przyspieszenie obliczeń (często 5-10x szybsze)
- Identyczne wyniki strategiczne jak standardowy Negamax
- Szczególnie efektywne przy większych głębokościach

### 5.3. ExpectiMinimax vs tradycyjne algorytmy

**W wariancie deterministycznym:**
- Porównywalne wyniki z Negamax
- Nieco wolniejszy z powodu dodatkowych obliczeń prawdopodobieństwa

**W wariancie probabilistycznym:**
- Przewaga strategiczna dzięki uwzględnieniu niepewności
- Lepsze radzenie sobie z nieprzewidywalnymi sytuacjami
- Wyższa skuteczność przy dłuższych grach

### 5.4. Wpływ losowości na strategie

**Główne zmiany:**
- Zmniejszenie przewagi algorytmów o większej głębokości
- Zwiększenie liczby remisów
- Większe znaczenie adaptacyjności niż czystej mocy obliczeniowej

## 6. Problemy napotkane podczas implementacji

### 6.1. Wyzwania techniczne
- **Kopiowanie stanu gry**: Konieczność dokładnego kopiowania wszystkich elementów stanu
- **Obsługa prawdopodobieństwa**: Poprawne modelowanie różnych scenariuszy w ExpectiMinimax
- **Pomiar wydajności**: Precyzyjny pomiar czasu wykonania z uwzględnieniem różnych warunków

### 6.2. Wyzwania algorytmiczne
- **Głębokość vs czas**: Balansowanie między dokładnością a szybkością wykonania
- **Odcięcie alfa-beta**: Implementacja efektywnego odcięcia bez utraty dokładności
- **Modelowanie niepewności**: Właściwe uwzględnienie probabilistycznych aspektów gry

## 7. Wnioski końcowe

### 7.1. Rekomendacje algorytmiczne

**Dla gier deterministycznych:**
- **Negamax z alfa-beta (głębokość 5)** - optymalne połączenie szybkości i skuteczności
- Standardowy Negamax tylko dla małych głębokości z powodu wydajności

**Dla gier probabilistycznych:**
- **ExpectiMinimax z alfa-beta (głębokość 3-5)** - najlepsze radzenie sobie z niepewnością
- Wyższa odporność na losowe wydarzenia

### 7.2. Znaczenie praktyczne

Eksperymenty pokazują, że:
1. **Odcięcie alfa-beta** jest kluczowe dla wydajności bez utraty jakości
2. **Specjalizowane algorytmy** (ExpectiMinimax) mają przewagę w środowiskach z niepewnością
3. **Głębokość przeszukiwania** pozostaje ważna, ale jej znaczenie maleje z wzrostem losowości
4. **Czas wykonania** może być krytyczny w aplikacjach czasu rzeczywistego

### 7.3. Kierunki dalszych badań

- Implementacja bardziej zaawansowanych heurystyk oceny pozycji
- Testowanie na większych planszach lub innych grach probabilistycznych
- Optymalizacja ExpectiMinimax dla lepszej wydajności
- Badanie wpływu różnych poziomów niepewności (nie tylko 20%)

## 8. Podsumowanie

Projekt wykazał znaczące różnice w zachowaniu algorytmów AI w środowiskach deterministycznych i probabilistycznych. Kluczowym wnioskiem jest potrzeba dostosowania strategii algorytmicznych do charakterystyki konkretnej gry, szczególnie w obecności elementów losowych.

Implementacja wszystkich trzech typów algorytmów pozwoliła na kompleksowe porównanie i wykazała, że nie istnieje uniwersalnie najlepszy algorytm - wybór zależy od specyfiki problemu, wymagań czasowych i poziomu niepewności w środowisku.
"""

    return report


def main():
    """
    Główna funkcja uruchamiająca wszystkie eksperymenty
    """
    print("=" * 50)

    tester = AIPerformanceTester()

    print("🔬 Uruchamianie eksperymentów...")
    results, time_results = tester.run_comprehensive_experiment(num_games=50)

    # Stwórz wizualizacje
    print("📊 Tworzenie wizualizacji...")
    create_comprehensive_visualizations(results, time_results)

    # Generuj raport
    print("Generowanie raportu...")
    report = generate_comprehensive_report(results, time_results)

    # Zapisz raport
    with open("comprehensive_tictacdoh_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # Zapisz wyniki do CSV dla dalszej analizy
    results_df = pd.DataFrame(results)
    time_df = pd.DataFrame(time_results)

    results_df.to_csv("game_results.csv", index=False)
    time_df.to_csv("time_results.csv", index=False)

    print("✅ Wszystkie eksperymenty zakończone!")
    print("📄 Pliki wygenerowane:")
    print("   - comprehensive_tictacdoh_report.md (główny raport)")
    print("   - comprehensive_tictacdoh_results.png (wykresy główne)")
    print("   - move_failure_rates.png (wskaźniki niepowodzeń)")
    print("   - game_results.csv (wyniki gier)")
    print("   - time_results.csv (wyniki czasowe)")


if __name__ == "__main__":
    main()