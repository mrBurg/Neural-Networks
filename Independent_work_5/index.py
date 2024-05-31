"""Module"""

import dataclasses
import math
from datetime import datetime as Date
from abc import abstractmethod


def calculate_deposit(contribution, rate, year):
    """
    contribution: Вклад
    rate: % ставка
    term: Кінцевий рік
    """

    now = Date.now()
    future_date = now.replace(year)
    terms = max(future_date.year - now.year, 0)

    return contribution * (1 + rate / 100) ** terms


CONTRIBUTION = 100
RATE = 5
YEAR = 2027

RES = calculate_deposit(CONTRIBUTION, RATE, YEAR)
print(
    f"Сума на депозиті в [\033[35m {YEAR} \033[0m] році при ставці в [\033[92m {RATE}% \033[0m] складатиме [\033[92m {RES:.2f} \033[0m] одиниць"
)

RATE = 7
RES = calculate_deposit(CONTRIBUTION, RATE, YEAR)
print(
    f"Сума на депозиті в [\033[35m {YEAR} \033[0m] році при ставці в [\033[92m {RATE}% \033[0m] складатиме [\033[92m {RES:.2f} \033[0m] одиниць"
)


@dataclasses.dataclass
class Circle:
    """
    :radius: Радіус кола
    :name: Назва об’єкту (не обов’язковоий)
    """

    def __init__(self, radius, name="Circle") -> None:
        self.__radius = radius
        self.name = name

    def get_area(self) -> float:
        """Area calculation"""

        return math.pi * self.__radius**2

    def get_perimeter(self) -> float:
        """Perimeter calculation"""

        return 2 * math.pi * self.__radius

    def info(self):
        """Printing Info"""

        print(self)

    @abstractmethod
    def calculate_expenses(self, expenses):
        """Calculate expenses"""

    def __str__(self):
        return f"Периметр кола \033[35m{self.name}\033[0m з радіусом [\033[92m {self.__radius:.2f}m \033[0m] становить [\033[92m {self.get_perimeter():.2f}m \033[0m] та площа [\033[92m {self.get_area():.2f}m² \033[0m]"


circle1 = Circle(math.pi, "Circle 1")
circle2 = Circle(math.pi**math.pi, "Circle 2")

circle1.info()
circle2.info()


class Lacquered(Circle):
    """Lacquered shape"""

    expenses = 0

    def __init__(self, radius, name) -> None:
        super().__init__(radius, name)

    def calculate_expenses(self, expenses=150):  # 150г/1m²
        """calculate_expenses"""

        self.expenses = super().get_area() * expenses

        print(self)

    def __str__(self):
        return f"Витрата лаку на фігуру \033[35m{self.name}\033[0m площею в [\033[92m {super().get_area():.2f}m² \033[0m] становить [\033[92m {self.expenses:.2f}\u0433 \033[0m] лаку."


class Aerosol(Circle):
    """Aerosol shape"""

    expenses = 0

    def __init__(self, radius, name) -> None:
        super().__init__(radius, name)

    def calculate_expenses(self, expenses=2.5):  # 1б/2.5m²
        """calculate_expenses"""

        self.expenses = super().get_area() * expenses

        print(self)

    def __str__(self):
        return f"Витрата аерозолі на фігуру \033[35m{self.name}\033[0m площею в [\033[92m {super().get_area():.2f}m² \033[0m] становить [\033[92m {self.expenses:.2f} \033[0m] банки."


figure_list = [
    Lacquered(math.pi, "Lacquered Circle 1"),
    Lacquered(math.pi**math.pi, "Lacquered Circle 1"),
    Aerosol(math.pi, "Aerosol Circle 1"),
    Aerosol(math.pi**math.pi, "Aerosol Circle 1"),
]

for i in figure_list:
    i.calculate_expenses()
