import sys
import random
import math
from typing import Tuple, Optional

import networkx as nx
import numpy as np
import pygame

SIZE_OF_CITY = 20

class City:
    def __init__(self, x: int, y: int):
        """
        Inicjalizacja miasta.

        Args:
            x: Współrzędna X na planszy
            y: Współrzędna Y na planszy
        """
        self.x = x
        self.y = y
        self.warrior = 10
        self.player = None

    def step(self):
        limit = 10 if self.player is None else 50
        if self.warrior < limit:
            self.warrior+=1

    # Interface
    def draw(self, screen, font, world_to_screen_fn):
        screen_x, screen_y = world_to_screen_fn(self.x, self.y)

        #TODO tu będzie kolor gracza.
        pygame.draw.circle(screen, [255, 0, 0], (screen_x, screen_y), SIZE_OF_CITY)
        pygame.draw.circle(screen, (0, 0, 0), (screen_x, screen_y), SIZE_OF_CITY, 2)
        if font:
            text = font.render("City", True, (0, 0, 0))
            rect = text.get_rect(center=(screen_x, screen_y - SIZE_OF_CITY - 10))
            screen.blit(text, rect)


    def is_clicked(self, mouse_pos: Tuple[int, int], world_to_screen_fn) -> bool:
        """
        Sprawdza, czy miasto zostało kliknięte.

        Args:
            mouse_pos: Pozycja kliknięcia myszy (x, y)

        Returns:
            bool: True, jeśli kliknięcie było wewnątrz miasta
        """
        screen_x, screen_y = world_to_screen_fn(self.x, self.y)
        distance = math.sqrt((mouse_pos[0] - screen_x) ** 2 + (mouse_pos[1] - screen_y) ** 2)
        return distance <= SIZE_OF_CITY


class GridGame:
    """Główna klasa gry z planszą NxN i miastami."""

    def __init__(self,
                 grid_size: int = 10,
                 world_size: int = 10,
                 screen_size: int = 1000,
                 num_cities: int = 5,
                 max_neutral_warrior: int = 10,
                 seed: Optional[int] = None
                 ):

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.py_rng = random.Random(seed)
        self.world_size = world_size

        self._init_for_pygame(grid_size, screen_size, num_cities, max_neutral_warrior)

        # Initialize graph
        self.graph = nx.Graph()
        # Generowanie miast
        self._generate_cities()


        # Wybrane miasto
        self.selected_city = None

    def _init_for_pygame(self,
                         grid_size: int = 10,
                         screen_size: int = 200,
                         num_cities: int = 5,
                         max_neutral_warrior: int = 10):
        # Inicjalizacja PyGame
        pygame.init()

        # Ustawienia gry
        self.grid_size = grid_size
        self.screen_size = screen_size
        self.cell_size = screen_size // grid_size
        self.num_cities = min(num_cities, grid_size * grid_size)
        self.max_neutral_warrior = max_neutral_warrior

        # Tworzenie okna gry
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption(f"Plansza {grid_size}x{grid_size} z miastami")

        # Czcionki
        self.font = pygame.font.SysFont('Arial', 14)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

        # Kolory
        self.BACKGROUND_COLOR = (240, 240, 240)
        self.GRID_COLOR = (200, 200, 200)


        self.PLAYERS_COLORS = [
            (255, 0, 0),    # Czerwony
            (0, 0, 255),    # Niebieski
            (0, 128, 0),    # Zielony
            (255, 165, 0),  # Pomarańczowy
            (128, 0, 128),  # Fioletowy
            (165, 42, 42),  # Brązowy
            (0, 128, 128),  # Morski
            (255, 105, 180) # Różowy
        ]


    def _generate_cities(self):
        """Generuje losowe miasta na planszy."""

        positions = set()
        for _ in range(self.num_cities):
            for _ in range(100):
                x = self.py_rng.randint(0, self.world_size)
                y = self.py_rng.randint(0, self.world_size)
                if (x, y) in positions:
                    continue
                positions.add((x, y))
                city = City(x, y)
                self.graph.add_node(city)
                break

        # Tworzymy pełny graf – krawędzie między wszystkimi miastami
        cities = list(self.graph.nodes)
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                city_a = cities[i]
                city_b = cities[j]
                dx = city_a.x - city_b.x
                dy = city_a.y - city_b.y
                distance = round((dx ** 2 + dy ** 2) ** 0.5, 1)  # Euklidesowy dystans
                self.graph.add_edge(city_a, city_b, weight=distance)



##################################################################
##################################################################
#              RYSOWANIE DLA CELÓW WIZUALIZACYJNYCH              #
##################################################################
##################################################################
    def world_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """Skaluje pozycję świata do rozmiaru ekranu."""
        sx = int((x / self.world_size) * self.screen_size)
        sy = int((y / self.world_size) * self.screen_size)
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> Tuple[int, int]:
        """Przekształca pozycję ekranu na współrzędne świata."""
        wx = int((sx / self.screen_size) * self.world_size)
        wy = int((sy / self.screen_size) * self.world_size)
        return wx, wy

    def _draw_edges(self):

        for city_a, city_b, data in self.graph.edges(data=True):
            ax, ay = self.world_to_screen(city_a.x, city_a.y)
            bx, by = self.world_to_screen(city_b.x, city_b.y)

            pygame.draw.line(self.screen, (100, 100, 100), (ax, ay), (bx, by), 2)

            mx, my = (ax + bx) // 2, (ay + by) // 2
            text = self.font.render(f"{data['weight']}", True, (0, 0, 0))
            rect = text.get_rect(center=(mx, my))
            self.screen.blit(text, rect)

    def _draw_game(self):
        """Rysuje siatkę planszy."""

        # Czyszczenie ekranu
        self.screen.fill(self.BACKGROUND_COLOR)

        for i in range(self.grid_size + 1):
            # Poziome linie
            pygame.draw.line(
                self.screen,
                self.GRID_COLOR,
                (0, i * self.cell_size),
                (self.screen_size, i * self.cell_size),
                1
            )

            # Pionowe linie
            pygame.draw.line(
                self.screen,
                self.GRID_COLOR,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.screen_size),
                1
            )

        # Rysowanie miast
        for city in self.graph.nodes:
            # Podświetl wybrane miasto
            if city == self.selected_city:
                screen_x, screen_y = self.world_to_screen(city.x, city.y)
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 0),
                    (screen_x, screen_y),
                    SIZE_OF_CITY + 5,
                    3
                )

            city.draw(self.screen, self.font, self.world_to_screen)

        self._draw_edges()

        # Wyświetlanie informacji o wybranym mieście
        self._draw_city_info()

        # Aktualizacja ekranu
        pygame.display.flip()

    def _draw_city_info(self):
        """Wyświetla informacje o wybranym mieście."""
        if self.selected_city:
            # Przygotuj informacje o mieście
            info_text = [
                f"Pozycja: ({self.selected_city.x}, {self.selected_city.y})",
                f"Player: {self.selected_city.player}",
            ]

            # TODO - narysować ilość wojowników

            # Rysuj panel informacyjny
            info_surface = pygame.Surface((250, 100))
            info_surface.fill((255, 255, 255))
            pygame.draw.rect(info_surface, (0, 0, 0), info_surface.get_rect(), 2)

            # Tytuł
            title = self.title_font.render("Informacje o mieście", True, (0, 0, 0))
            info_surface.blit(title, (10, 10))

            # Informacje
            for i, text in enumerate(info_text):
                info_line = self.font.render(text, True, (0, 0, 0))
                info_surface.blit(info_line, (20, 40 + i * 20))

            # Wyświetl panel w prawym górnym rogu
            self.screen.blit(info_surface, (self.screen_size - 260, 10))

    def _handle_events(self):
        """Obsługa zdarzeń PyGame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


            #TODO (20.05.2025): poprawić wybieranie miast na mapie. tak aby było git. + Debuggowanie, wyświetlaie lokalizacji w świecie.
            # Obsługa kliknięcia myszy
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Lewy przycisk myszy
                    mouse_pos = pygame.mouse.get_pos()
                    for city in self.graph.nodes:
                        if city.is_clicked(mouse_pos, self.world_to_screen):
                            self.selected_city = city
                            print("Wybrano Miasto")
                            return
                    self.selected_city = None

            # Obsługa klawiatury
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Klawisz R - wygeneruj nowe miasta
                    self._generate_cities()
                    self.selected_city = None
                elif event.key == pygame.K_ESCAPE:  # ESC - odznacz miasto
                    self.selected_city = None

    def run(self):
        """Główna pętla gry."""
        clock = pygame.time.Clock()

        while True:

            for city in self.graph.nodes:
                city.step()

            # Jednostki.

            # Obsługa zdarzeń / Tutaj będą działania graczy
            self._handle_events()

            # Rysowanie Gry
            self._draw_game()

            # Ograniczenie FPS
            clock.tick(120)

    # Interface



def main():
    """Funkcja główna programu."""
    # Parametry gry
    grid_size = 20       # Rozmiar siatki NxN
    screen_size = 800    # Rozmiar okna w pikselach
    num_cities = 7       # Liczba miast

    # Utworzenie i uruchomienie gry
    game = GridGame(
        seed=1000,
        grid_size=grid_size,
        screen_size=screen_size,
        num_cities=num_cities,
    )

    game.run()


if __name__ == "__main__":
    main()