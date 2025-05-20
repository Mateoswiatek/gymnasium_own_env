import sys
import random
import math
from typing import Tuple, Optional

import networkx as nx
import numpy as np
import pygame


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
    def draw(self, screen: pygame.Surface, font: Optional[pygame.font.Font] = None):
        """
        Rysuje miasto na ekranie.

        Args:
            screen: Powierzchnia PyGame do rysowania
            font: Czcionka do wyświetlania nazwy miasta
        """
        #TODO (20.05.2025): Tutaj będzie kolor gracza.

        # Rysowanie okręgu reprezentującego miasto
        pygame.draw.circle(screen, [255, 0, 0], (self.x, self.y), 20)

        # Rysowanie obramowania miasta
        pygame.draw.circle(screen, (0, 0, 0), (self.x, self.y), 20, 2)

        # Wyświetlanie nazwy miasta, jeśli podano czcionkę
        if font:
            text = font.render("City", True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.x, self.y - 20 - 10))
            screen.blit(text, text_rect)

    def is_clicked(self, mouse_pos: Tuple[int, int]) -> bool:
        """
        Sprawdza, czy miasto zostało kliknięte.

        Args:
            mouse_pos: Pozycja kliknięcia myszy (x, y)

        Returns:
            bool: True, jeśli kliknięcie było wewnątrz miasta
        """
        distance = math.sqrt((mouse_pos[0] - self.x) ** 2 + (mouse_pos[1] - self.y) ** 2)
        return distance <= 20


class GridGame:
    """Główna klasa gry z planszą NxN i miastami."""

    def __init__(self,
                 grid_size: int = 10,
                 screen_size: int = 800,
                 num_cities: int = 5,
                 max_neutral_warrior: int = 10,
                 seed: Optional[int] = None
                 ):
        """
        Inicjalizacja gry.

        Args:
            grid_size: Rozmiar siatki (NxN)
            screen_size: Rozmiar okna gry w pikselach
            num_cities: Liczba miast do wygenerowania
            min_city_size: Minimalny rozmiar miasta
            max_city_size: Maksymalny rozmiar miasta
        """

        if seed is not None:
            np.random.seed(seed)

        self._init_for_pygame(grid_size, screen_size,num_cities, max_neutral_warrior)


        # Initialize graph
        self.graph = nx.Graph()
        # Generowanie miast
        self._generate_cities()


        # Wybrane miasto
        self.selected_city = None

    def _init_for_pygame(self,
                         grid_size: int = 10,
                         screen_size: int = 800,
                         num_cities: int = 5,
                         max_neutral_warrior: int = 10,):
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
        city_positions = set()  # Do śledzenia zajętych pozycji


        for i in range(self.num_cities):
            for _ in range(100):
                gx = random.randint(0, self.grid_size - 1)
                gy = random.randint(0, self.grid_size - 1)

                if (gx, gy) in city_positions:
                    continue

                city_positions.add((gx, gy))
                px = gx * self.cell_size + self.cell_size // 2
                py = gy * self.cell_size + self.cell_size // 2
                self.graph.add_node(City(px, py))
                break

        #
        # for i in range(self.num_cities):
        #     # Próbuj znaleźć wolną pozycję dla miasta
        #     attempts = 0
        #     while attempts < 100:  # Limit prób, aby uniknąć nieskończonej pętli
        #         # Wybierz losową komórkę siatki
        #         grid_x = random.randint(0, self.grid_size - 1)
        #         grid_y = random.randint(0, self.grid_size - 1)
        #
        #         # Sprawdź, czy pozycja jest już zajęta
        #         if (grid_x, grid_y) not in city_positions:
        #             city_positions.add((grid_x, grid_y))
        #             break
        #
        #         attempts += 1
        #
        #     if attempts == 100:
        #         print(f"Nie udało się znaleźć miejsca dla miasta {i+1}.")
        #         continue
        #
        #     # Oblicz pozycję piksela (środek komórki)
        #     pixel_x = grid_x * self.cell_size + self.cell_size // 2
        #     pixel_y = grid_y * self.cell_size + self.cell_size // 2
        #
        #     self.graph.add_node(City(pixel_x, pixel_y))

    def _draw_grid(self):
        """Rysuje siatkę planszy."""
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

    def _draw_city_info(self):
        """Wyświetla informacje o wybranym mieście."""
        if self.selected_city:
            # Przygotuj informacje o mieście
            info_text = [
                f"Nazwa: {self.selected_city.name}",
                f"Pozycja: ({self.selected_city.x}, {self.selected_city.y})",
                f"Rozmiar: {self.selected_city.size}",

            ]

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

            # Obsługa kliknięcia myszy
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Lewy przycisk myszy
                    mouse_pos = pygame.mouse.get_pos()
                    self._handle_click(mouse_pos)

            # Obsługa klawiatury
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Klawisz R - wygeneruj nowe miasta
                    self._generate_cities()
                    self.selected_city = None
                elif event.key == pygame.K_ESCAPE:  # ESC - odznacz miasto
                    self.selected_city = None

    def _handle_click(self, mouse_pos: Tuple[int, int]):
        """
        Obsługuje kliknięcie myszy.

        Args:
            mouse_pos: Pozycja kliknięcia (x, y)
        """
        # Sprawdź, czy kliknięto miasto
        for city in self.cities:
            if city.is_clicked(mouse_pos):
                self.selected_city = city
                print(f"Wybrano {city.name}")
                return

        # Jeśli kliknięto poza miastami, odznacz wybrane miasto
        self.selected_city = None

    def run(self):
        """Główna pętla gry."""
        clock = pygame.time.Clock()

        while True:
            # Obsługa zdarzeń
            self._handle_events()

            # Czyszczenie ekranu
            self.screen.fill(self.BACKGROUND_COLOR)

            # Rysowanie siatki
            self._draw_grid()

            for city in self.graph.nodes:
                city.step()

            # Rysowanie miast
            for city in self.graph.nodes:
                # Podświetl wybrane miasto
                if city == self.selected_city:
                    # Rysuj podświetlenie
                    pygame.draw.circle(
                        self.screen,
                        (255, 255, 0),
                        (city.x, city.y),
                        city.size + 5,
                        3
                    )

                city.draw(self.screen, self.font)

            # Wyświetlanie informacji o wybranym mieście
            self._draw_city_info()

            # Aktualizacja ekranu
            pygame.display.flip()

            # Ograniczenie FPS
            clock.tick(120)


    # Interface



def main():
    """Funkcja główna programu."""
    # Parametry gry
    grid_size = 10       # Rozmiar siatki NxN
    screen_size = 800    # Rozmiar okna w pikselach
    num_cities = 7       # Liczba miast

    # Utworzenie i uruchomienie gry
    game = GridGame(
        grid_size=grid_size,
        screen_size=screen_size,
        num_cities=num_cities,
    )

    game.run()


if __name__ == "__main__":
    main()