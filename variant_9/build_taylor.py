"""Символьное вычисление полинома Тейлора."""

import sympy as sp


def build_taylor():
    """Выводит значения в нуле и возвращает переменную, функцию и полином."""
    x = sp.Symbol("x")
    f = sp.sin(x) * sp.exp(-x / 2)
    f0 = f.subs(x, 0)
    f1 = sp.diff(f, x).subs(x, 0)
    f2 = sp.diff(f, x, 2).subs(x, 0)
    f3 = sp.diff(f, x, 3).subs(x, 0)

    # Делим производные в нуле на соответствующие факториалы.
    p3 = f0 + f1 * x + f2 * x**2 / sp.factorial(2) + f3 * x**3 / sp.factorial(3)

    print(f"f(0) = {f0}")
    print(f"f'(0) = {f1}")
    print(f"f''(0) = {f2}")
    print(f"f'''(0) = {f3}")
    print(f"P3(x) = {p3}")

    return x, f, p3
