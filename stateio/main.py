import sys
import random
import math
from typing import List, Tuple, Optional

import pygame


class City:
    def __init__(self, x: int, y: int, size: int, color: Tuple[int, int, int], name: str = ""):
        """
        Inicjalizacja miasta.

        Args:
            x: Współrzędna X na planszy
            y: Współrzędna Y na planszy
            size: Rozmiar miasta (promień okręgu)
            color: Kolor miasta w formacie RGB
            name: Nazwa miasta
        """
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.name = name if name else f"Miasto {x},{y}"
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
        # Rysowanie okręgu reprezentującego miasto
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

        # Rysowanie obramowania miasta
        pygame.draw.circle(screen, (0, 0, 0), (self.x, self.y), self.size, 2)

        # Wyświetlanie nazwy miasta, jeśli podano czcionkę
        if font:
            text = font.render(self.name, True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.x, self.y - self.size - 10))
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
        return distance <= self.size


class GridGame:
    """Główna klasa gry z planszą NxN i miastami."""

    def __init__(self,
                 grid_size: int = 10,
                 screen_size: int = 800,
                 num_cities: int = 5,
                 min_city_size: int = 10,
                 max_city_size: int = 30):
        """
        Inicjalizacja gry.

        Args:
            grid_size: Rozmiar siatki (NxN)
            screen_size: Rozmiar okna gry w pikselach
            num_cities: Liczba miast do wygenerowania
            min_city_size: Minimalny rozmiar miasta
            max_city_size: Maksymalny rozmiar miasta
        """
        # Inicjalizacja PyGame
        pygame.init()

        # Ustawienia gry
        self.grid_size = grid_size
        self.screen_size = screen_size
        self.cell_size = screen_size // grid_size
        self.num_cities = min(num_cities, grid_size * grid_size)  # Nie więcej miast niż komórek
        self.min_city_size = min_city_size
        self.max_city_size = max_city_size

        # Tworzenie okna gry
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption(f"Plansza {grid_size}x{grid_size} z miastami")

        # Czcionki
        self.font = pygame.font.SysFont('Arial', 14)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

        # Lista miast
        self.cities: List[City] = []

        # Kolory
        self.BACKGROUND_COLOR = (240, 240, 240)
        self.GRID_COLOR = (200, 200, 200)
        self.CITY_COLORS = [
            (255, 0, 0),    # Czerwony
            (0, 0, 255),    # Niebieski
            (0, 128, 0),    # Zielony
            (255, 165, 0),  # Pomarańczowy
            (128, 0, 128),  # Fioletowy
            (165, 42, 42),  # Brązowy
            (0, 128, 128),  # Morski
            (255, 105, 180) # Różowy
        ]

        # Wybrane miasto
        self.selected_city = None

        # Generowanie miast
        self._generate_cities()

    def _generate_cities(self):
        """Generuje losowe miasta na planszy."""
        self.cities = []
        city_positions = set()  # Do śledzenia zajętych pozycji

        for i in range(self.num_cities):
            # Próbuj znaleźć wolną pozycję dla miasta
            attempts = 0
            while attempts < 100:  # Limit prób, aby uniknąć nieskończonej pętli
                # Wybierz losową komórkę siatki
                grid_x = random.randint(0, self.grid_size - 1)
                grid_y = random.randint(0, self.grid_size - 1)

                # Sprawdź, czy pozycja jest już zajęta
                if (grid_x, grid_y) not in city_positions:
                    city_positions.add((grid_x, grid_y))
                    break

                attempts += 1

            if attempts == 100:
                print(f"Nie udało się znaleźć miejsca dla miasta {i+1}.")
                continue

            # Oblicz pozycję piksela (środek komórki)
            pixel_x = grid_x * self.cell_size + self.cell_size // 2
            pixel_y = grid_y * self.cell_size + self.cell_size // 2

            # Losowy rozmiar miasta (nie większy niż połowa rozmiaru komórki)
            max_possible_size = min(self.max_city_size, self.cell_size // 2 - 2)
            size = random.randint(self.min_city_size, max_possible_size)

            # Losowy kolor miasta
            color = random.choice(self.CITY_COLORS)

            # Dodaj miasto do listy
            self.cities.append(City(pixel_x, pixel_y, size, color, f"Miasto {i+1}"))

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

            for city in self.cities:
                city.step()

            # Rysowanie miast
            for city in self.cities:
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


def main():
    """Funkcja główna programu."""
    # Parametry gry
    grid_size = 10       # Rozmiar siatki NxN
    screen_size = 800    # Rozmiar okna w pikselach
    num_cities = 7       # Liczba miast
    min_city_size = 15   # Minimalny rozmiar miasta
    max_city_size = 25   # Maksymalny rozmiar miasta

    # Utworzenie i uruchomienie gry
    game = GridGame(
        grid_size=grid_size,
        screen_size=screen_size,
        num_cities=num_cities,
        min_city_size=min_city_size,
        max_city_size=max_city_size
    )

    game.run()


if __name__ == "__main__":
    main()