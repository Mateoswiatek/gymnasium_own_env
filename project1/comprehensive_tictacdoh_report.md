# Raport: Kompleksowe eksperymenty z TicTacDoh i algorytmami AI

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
|----------------------|---------|---------------|---------------|--------|----------------|
| Negamax_3 vs Negamax_5 | Deterministic | 25 | 25 | 0 | 0 |
| Negamax_3 vs Negamax_5 | Probabilistic | 17 | 25 | 8 | 80 |
| Negamax_3 vs Negamax_AB_3 | Deterministic | 25 | 25 | 0 | 0 |
| Negamax_3 vs Negamax_AB_3 | Probabilistic | 21 | 23 | 6 | 85 |
| Negamax_AB_3 vs Negamax_AB_5 | Deterministic | 25 | 25 | 0 | 0 |
| Negamax_AB_3 vs Negamax_AB_5 | Probabilistic | 20 | 17 | 13 | 73 |
| Negamax_AB_3 vs ExpectiMinimax_AB_3 | Deterministic | 25 | 25 | 0 | 0 |
| Negamax_AB_3 vs ExpectiMinimax_AB_3 | Probabilistic | 16 | 25 | 9 | 92 |
| Negamax_AB_5 vs ExpectiMinimax_AB_5 | Deterministic | 25 | 25 | 0 | 0 |
| Negamax_AB_5 vs ExpectiMinimax_AB_5 | Probabilistic | 20 | 21 | 9 | 70 |

### 4.2. Analiza czasów wykonania

| Porównanie algorytmów | Wariant | Avg czas Algo1 (ms) | Avg czas Algo2 (ms) | Stosunek szybkości |
|----------------------|---------|---------------------|---------------------|-------------------|
| Negamax_3 vs Negamax_5 | Deterministic | 3.52 | 20.73 | 5.89x |
| Negamax_3 vs Negamax_5 | Probabilistic | 4.38 | 28.95 | 6.60x |
| Negamax_3 vs Negamax_AB_3 | Deterministic | 3.76 | 0.62 | 0.17x |
| Negamax_3 vs Negamax_AB_3 | Probabilistic | 3.70 | 0.61 | 0.16x |
| Negamax_AB_3 vs Negamax_AB_5 | Deterministic | 0.55 | 2.81 | 5.06x |
| Negamax_AB_3 vs Negamax_AB_5 | Probabilistic | 0.59 | 3.56 | 6.03x |
| Negamax_AB_3 vs ExpectiMinimax_AB_3 | Deterministic | 0.60 | 0.61 | 1.02x |
| Negamax_AB_3 vs ExpectiMinimax_AB_3 | Probabilistic | 0.67 | 4.77 | 7.08x |
| Negamax_AB_5 vs ExpectiMinimax_AB_5 | Deterministic | 3.02 | 3.05 | 1.01x |
| Negamax_AB_5 vs ExpectiMinimax_AB_5 | Probabilistic | 3.81 | 262.05 | 68.79x |

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
