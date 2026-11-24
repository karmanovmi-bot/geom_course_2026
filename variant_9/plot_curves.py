"""Построение и сохранение графика."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_curves(x_values, f_values, p3_values):
    """Показывает обе кривые в одном окне и сохраняет их в PNG."""
    fig = plt.figure(figsize=(9, 5))
    plt.plot(x_values, f_values, label=r"Исходная функция $f(x)$", linewidth=2)
    plt.plot(x_values, p3_values, "--", label=r"Полином Тейлора $P_3(x)$", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("Значение функции")
    plt.title("Приближение функции полиномом Тейлора")
    plt.ylim(-2.7, 0.8)
    plt.grid(True, alpha=0.35)
    plt.legend()
    fig.tight_layout()

    output_path = Path(__file__).with_name("taylor_comparison.png")
    fig.savefig(output_path, dpi=160)
    plt.show()
