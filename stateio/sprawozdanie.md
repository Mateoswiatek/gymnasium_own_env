# Gra strategiczna na grafie z uczeniem ze wzmocnieniem

## Wprowadzenie

Projekt przedstawia zaawansowaną implementację gry strategicznej w czasie rzeczywistym, która łączy elementy klasycznych gier strategicznych z nowoczesnymi technikami uczenia ze wzmocnieniem. System został zaprojektowany jako środowisko badawcze do testowania algorytmów sztucznej inteligencji w kontekście podejmowania strategicznych decyzji w warunkach niepełnej informacji i ograniczeń czasowych.

Gra opiera się na modelu grafowym, w którym gracze kontrolują miasta rozmieszczone na dwuwymiarowej planszy i konkurują o dominację poprzez strategiczne rozmieszczenie jednostek militarnych oraz koordynację ataków na pozycje przeciwników. Środowisko charakteryzuje się wysokim poziomem złożoności strategicznej, wymagając od uczestników długoterminowego planowania, oceny ryzyka oraz adaptacji do zmieniających się warunków na polu bitwy.

## Mechanika rozgrywki

### Struktura planszy

Plansza gry składa się z miast rozmieszczonych na dwuwymiarowej powierzchni o skończonych wymiarach. Każde miasto posiada unikalne współrzędne kartezjańskie oraz charakteryzuje się trzema podstawowymi atrybutami: liczbą stacjonujących jednostek, przynależnością do określonego gracza lub statusem neutralnym, oraz maksymalną pojemnością jednostek. Miasta połączone są grafem pełnym, co oznacza że każde miasto może komunikować się bezpośrednio z każdym innym miastem na planszy.

### System produkcji jednostek

Miasta funkcjonują jako centra produkcji jednostek militarnych, generując automatycznie nowe siły z każdym krokiem temporalnym gry. Mechanizm produkcji zależy od statusu własnościowego miasta - miasta należące do aktywnych graczy mogą produkować maksymalnie pięćdziesiąt jednostek, podczas gdy miasta neutralne ograniczone są do dziesięciu jednostek. System ten wprowadza element gospodarczy do rozgrywki, wymagając od graczy strategicznego zarządzania zasobami oraz planowania ekspansji w oparciu o potencjał produkcyjny kontrolowanych terytoriów.

### System walki i podboju

Po dotarciu do miasta docelowego armie wykonują jedną z dwóch akcji: w przypadku miasta przyjaznego jednostki zostają włączone do lokalnej garnizonu, wzmacniając obronę pozycji; w przypadku miasta wrogiego następuje bezpośrednie starcie, w którym przewaga liczebna decyduje o sukcesie operacji. Miasto zostaje podbite gdy atakujące siły przewyższają liczebnie obronę, przejmując kontrolę nad pozycją oraz pozostawiając nadwyżkę jednostek jako nowy garnizon.

## Architektura techniczna

### Przestrzeń akcji

Przestrzeń akcji zostały zaprojektowana jako dyskretna, obejmująca wszystkie możliwe kombinacje miast źródłowych i docelowych dla przemieszczenia jednostek. Dodatkowo dostępna jest akcja passa, pozwalająca graczowi na świadome przekazanie tury bez wykonywania ruchu. Łączna liczba dostępnych akcji wynosi 2n(n-1)+1, gdzie n reprezentuje liczbę miast na planszy. Struktura ta zapewnia pełną swobodę strategiczną przy zachowaniu obliczalnej złożoności dla algorytmów uczenia.

### Przestrzeń obserwacji

Przestrzeń obserwacji została skonstruowana jako złożona struktura słownikowa zawierająca kompletne informacje o stanie gry. Komponenty obserwacji obejmują rozkład jednostek w poszczególnych miastach oraz informacje o właścicielach wszystkich pozycji na planszy.

### System wizualizacji

Komponent wizualizacji wykorzystuje bibliotekę PyGame do renderowania stanu gry w czasie rzeczywistym. Miasta przedstawiane są jako kolorowe okręgi, których barwa odpowiada przynależności do gracza, z wyraźnie oznaczoną liczbą stacjonujących jednostek.

## Komponenty systemowe

### Klasa GridGame

Główna klasa GridGame stanowi centralny punkt kontroli środowiska, dziedzicząc po gym.Env oraz implementując kompletną logikę gry. Klasa zarządza stanem globalnym, procesami temporalnymi związanymi z ruchem armii, kontrolą kolejności graczy oraz walidacją akcji. Dodatkowo odpowiada za generowanie obserwacji, kalkulację nagród oraz detekcję warunków zakończenia gry. Implementacja uwzględnia mechanizmy bezpieczeństwa zapobiegające niepoprawnym stanom gry oraz zapewnia deterministyczne zachowanie przy użyciu kontrolowanego generatora liczb pseudolosowych.

### Klasa City

Klasa City enkapsuluje logikę związaną z funkcjonowaniem poszczególnych miast na planszy. Implementuje mechanizm automatycznej produkcji jednostek, zarządzanie pojemnością magazynową oraz interakcję z systemem wizualizacji. Klasa udostępnia metody sprawdzania kolizji z kursorem myszy, co umożliwia intuicyjną interakcję z interfejsem użytkownika. Dodatkowo implementuje logikę renderowania, zapewniając spójną prezentację wizualną niezależnie od rozmiaru okna gry.

### Klasa Player

Klasa Player obsługuje zarówno graczy człowieka poprzez interfejs graficzny, jak i autonomicznych botów z wbudowaną sztuczną inteligencją. Dla graczy ludzkich implementuje system interakcji oparty na zdarzeniach myszy oraz klawiatury, umożliwiając intuicyjne wydawanie rozkazów poprzez klikanie miast oraz nawigację klawiaturą. Boty wykorzystują prostą heurystykę decyzyjną, analizującą stosunek sił oraz dystanse między miastami w celu podejmowania racjonalnych decyzji strategicznych.

### Klasa Army

Klasa Army reprezentuje armie przemieszczające się między miastami, śledząc wszystkie istotne parametry związane z podróżą. Implementuje logikę postupu czasowego, kalkulację pozycji dla celów wizualizacji oraz zarządzanie stanem jednostek w trakcie przemieszczania. Klasa współpracuje z systemem kolizji w celu detekcji i rozstrzygania konfliktów między przeciwstawnymi armiami na identycznych trasach.

## Algorytm uczenia ze wzmocnieniem

### Implementacja Q-learning

Projekt zawiera zaawansowaną implementację agenta Q-learning specjalnie dostosowaną do dyskretnej przestrzeni akcji środowiska gry. Algorytm wykorzystuje tabelaryczną reprezentację funkcji wartości, gdzie każdy stan gry serializowany jest do postaci hashowalnej struktury danych umożliwiającej efektywne przechowywanie oraz wyszukiwanie wartości Q. Implementacja uwzględnia standardową regułę aktualizacji Bellmana z konfigurowalnymi parametrami współczynnika uczenia oraz współczynnika dyskontowego.

### Strategia eksploracji

Agent implementuje strategię epsilon-greedy do równoważenia eksploracji przestrzeni stanów z eksploatacją wyuczonej wiedzy. Wartość epsilon podlega stopniowej redukcji w trakcie procesu treningu zgodnie z harmonogramem wykładniczym, zapewniając intensywną eksplorację w początkowych etapach uczenia oraz stopniowe przejście do eksploatacji w miarę stabilizowania się polityki. Mechanizm ten zostaje dostrojony do specyfiki środowiska, uwzględniając długość epizodów oraz złożoność przestrzeni stanów.

### Serializacja stanów

System implementuje zaawansowaną metodę serializacji stanów gry do postaci umożliwiającej efektywne indeksowanie w tablicy Q. Proces serializacji uwzględnia wszystkie istotne komponenty stanu, włączając rozmieszczenie jednostek, informacje o właścicielach miast oraz szczegółowe dane armii w ruchu. Implementacja zapewnia determinizm oraz odporność na błędy, gwarantując identyczną serializację dla identycznych stanów gry.

## Funkcjonalności interakcji z człowiekiem

### Interfejs graficzny

System udostępnia intuicyjny interfejs graficzny umożliwiający komfortową rozgrywkę przez graczy ludzkich. Interakcja opiera się na systemie wskaźnikowym, gdzie gracze wybierają miasta poprzez kliknięcie lewym przyciskiem myszy oraz wydają rozkazy przemieszczenia jednostek poprzez wskazanie miasta docelowego. Wybrane miasto zostaje wyróżnione charakterystyczną żółtą obwódką, zapewniając natychmiastową informację zwrotną o aktualnym stanie selekcji.

### System informacyjny

Interfejs udostępnia rozbudowany panel informacyjny prezentujący szczegółowe dane o wybranym mieście, włączając współrzędne pozycji, przynależność do gracza oraz aktualną liczbę stacjonujących jednostek. Panel aktualizowany jest dynamicznie w odpowiedzi na działania gracza, zapewniając dostęp do aktualnych informacji strategicznych niezbędnych do podejmowania świadomych decyzji.

### Kontrola klawiatury

System implementuje zestaw skrótów klawiaturowych ułatwiających kontrolę nad grą. Klawisz Enter umożliwia świadome przekazanie tury bez wykonywania akcji, co pozwala na implementację strategii defensywnych lub oczekiwanie na korzystniejsze warunki. Klawisz Backspace inicjuje natychmiastowe zresetowanie gry do stanu początkowego, umożliwiając szybkie rozpoczęcie nowej partii bez konieczności restartu aplikacji.

## Przykłady zastosowań

### Badania algorytmów uczenia ze wzmocnieniem

Środowisko stanowi platformę badawczą do testowania zaawansowanych algorytmów uczenia ze wzmocnieniem w kontekście gier strategicznych wieloagentowych. Badacze mogą wykorzystać system do porównania wydajności różnych architektur sieci neuronowych, analizy wpływu hiperparametrów na proces konwergencji oraz badania strategii eksploracji w złożonych przestrzeniach stanów. Środowisko umożliwia również badanie zjawisk emergentnych wynikających z interakcji między autonomicznymi agentami.

Przykładowe scenariusze badawcze obejmują analizę stabilności procesu uczenia w warunkach niestacjonarnego środowiska, gdzie przeciwnicy również podlegają procesowi uczenia, badanie transferu wiedzy między różnymi konfiguracjami planszy oraz ocenę robustności wyuczonych strategii wobec perturbacji środowiskowych.

### Edukacja w dziedzinie sztucznej inteligencji

System może służyć jako zaawansowane narzędzie edukacyjne do nauczania koncepcji uczenia ze wzmocnieniem w atrakcyjnym wizualnie środowisku interaktywnym. Studenci mogą obserwować proces uczenia agentów w czasie rzeczywistym, analizować ewolucję strategii oraz eksperymentować z różnymi konfiguracjami parametrów algorytmów. Wizualizacja procesu decyzyjnego oraz możliwość bezpośredniej rywalizacji z agentami sztucznej inteligencji zapewnia intuicyjne zrozumienie złożonych koncepcji teoretycznych.

### Analiza strategii w grach wieloagentowych

Środowisko umożliwia systematyczne badanie dynamiki strategicznej w kontekście gier o sumie zerowej z niepełną informacją. Badacze mogą analizować emergentne zachowania wynikające z interakcji między agentami, identyfikować równowagi strategiczne oraz badać wpływ różnych funkcji nagrody na wyuczone polityki. System pozwala również na badanie kooperacji oraz konkurencji w środowiskach mieszanych, gdzie część graczy może współpracować przeciwko pozostałym uczestnikom.

## Konfiguracja i parametryzacja

### Parametry środowiska

System udostępnia rozbudowany zestaw parametrów konfiguracyjnych umożliwiających dostosowanie środowiska do specyficznych wymagań badawczych. Kluczowe parametry obejmują rozmiar planszy, liczbę miast, maksymalną pojemność jednostek, współczynniki produkcji oraz nasienie generatora liczb pseudolosowych dla zapewnienia reprodukowalności eksperymentów. Dodatkowo możliwa jest konfiguracja liczby graczy, rozmiarów okna wizualizacji oraz częstotliwości renderowania.

### Parametry algorytmu uczenia

Implementacja Q-learning udostępnia pełną kontrolę nad hiperparametrami algorytmu, włączając współczynnik uczenia, współczynnik dyskontowy, strategię eksploracji oraz harmonogram redukcji parametru epsilon. System umożliwia również konfigurację kryteriów zakończenia treningu, częstotliwości ewaluacji oraz mechanizmów zapisu stanu agenta dla celów analizy post-hoc.

## Perspektywy rozwoju

Aktualna implementacja stanowi solidną podstawę dla przyszłych rozszerzeń funkcjonalności. Planowane kierunki rozwoju obejmują implementację zaawansowanych algorytmów głębokiego uczenia ze wzmocnieniem, rozszerzenie środowiska o elementy stochastyczne oraz wprowadzenie mechanizmów komunikacji między agentami. Dodatkowo przewidywane jest dodanie trybu sieciowego umożliwiającego rozgrywkę wieloosobową oraz implementacja systemu turniejowego dla systematycznej oceny wydajności różnych strategii.
