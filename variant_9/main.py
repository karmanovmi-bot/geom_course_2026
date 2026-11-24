"""Запуск практической части варианта 9."""

from build_taylor import build_taylor
from evaluate_curves import evaluate_curves
from plot_curves import plot_curves


def main():
    x, f, p3 = build_taylor()
    x_values, f_values, p3_values = evaluate_curves(x, f, p3)
    plot_curves(x_values, f_values, p3_values)


if __name__ == "__main__":
    main()
