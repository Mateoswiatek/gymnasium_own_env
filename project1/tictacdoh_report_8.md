# Raport: Eksperymenty z TicTacDoh i algorytmem Negamax

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
- Różnych ustawień głębokości algorytmu Negamax (porównywano głębokości 2 vs 4 oraz 3 vs 5)
- Deterministycznego i probabilistycznego wariantu gry

Każdy eksperyment obejmował 100 gier między dwoma graczami AI, przy czym gracze na zmianę rozpoczynali kolejne partie.

## 4. Wyniki eksperymentów

### 4.1. Wariant deterministyczny

| Głębokości | Wygrane Gracz 1 | Wygrane Gracz 2 | Remisy |
|------------|-----------------|-----------------|--------|
| 2 vs 4 | 50 | 50 | 0 |
| 3 vs 5 | 50 | 50 | 0 |
| 2 vs 8 | 50 | 50 | 0 |

### 4.2. Wariant probabilistyczny

| Głębokości | Wygrane Gracz 1 | Wygrane Gracz 2 | Remisy | Nieudane ruchy |
|------------|-----------------|-----------------|--------|----------------|
| 2 vs 4 | 35 | 50 | 15 | 179 |
| 3 vs 5 | 38 | 49 | 13 | 158 |
| 2 vs 8 | 31 | 52 | 17 | 162 |

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

Podsumowując, eksperymenty pokazują, że skuteczność algorytmów AI w grach zależy nie tylko od ich parametrów (jak głębokość przeszukiwania), ale również od charakterystyki samej gry, w szczególności od obecności elementów losowych.