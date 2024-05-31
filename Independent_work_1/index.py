"""Module"""

print("·⁰²³⁴⁵⁶⁷⁸⁹\u00b7\u2070\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")

print(
    f"Result of (10 / 2.3 - 3⁴) * 0.7 + 9⁰·⁵ is : {(10 / 2.3 - (3**4)) * 0.7 + (9**0.5)}"
)


def quadratic_equation(a, b, c) -> int:
    """quadratic equation"""

    # A * X * X + B * X + C = 0
    # Ax²+Bx+C = 0

    discriminant = b**2 - 4 * a * c

    if discriminant == 0:
        return -b / 2 * a

    if discriminant > 0:
        x1 = -b + discriminant**2 / 2 * a
        x2 = -b - discriminant**2 / 2 * a

        return x1, x2

    return "Unsolvable equation"


for x, y, z in ((1, 4, -5), (3, -6, 3), (2, 1, 1), (0, 2, -1), (0, 0, 1)):
    print(quadratic_equation(x, y, z))

RES = 0

for i in range(1, 101):
    RES += i**3

print(RES)


def factorial(n):
    """Calculation factorial"""

    if n == 0:
        return 1

    return n * factorial(n - 1)


def calculate_order(n):
    """Calculates the sequence number"""

    return 2 * n - 1 / factorial(n)


for i in range(1, 21):
    RES = calculate_order(i)

print(RES)
