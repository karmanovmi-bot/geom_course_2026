"""Численное вычисление значений для графика."""

import numpy as np
import sympy as sp


def evaluate_curves(x, f, p3):
    """Возвращает сетку и значения обеих функций в её точках."""
    x_values = np.linspace(-1.5, 2.5, 400)

    # Превращаем символьные формулы в функции, принимающие массивы NumPy.
    f_numeric = sp.lambdify(x, f, "numpy")
    p3_numeric = sp.lambdify(x, p3, "numpy")
    f_values = f_numeric(x_values)
    p3_values = p3_numeric(x_values)

    return x_values, f_values, p3_values
