"""Bisection method for f(x) = 0 on a specified interval."""


def bisect(f, a: float, b: float, epsilon: float) -> tuple[float, int]:
    if a >= b:
        raise ValueError("a must be less than b")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")

    steps = 0
    while b - a >= epsilon:
        c = (a + b) / 2
        fc = f(c)
        steps += 1
        if fc == 0 or abs(fc) < epsilon:
            return c, steps
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2, steps


def f(x: float) -> float:
    """Example equation; replace this function to solve another equation."""
    return x**3 - x - 2


def main() -> None:
    a, b = 1.0, 2.0
    epsilon = 1e-6
    root, steps = bisect(f, a, b, epsilon)
    print(f"Корень: {root:.10f}")
    print(f"f(корень): {f(root):.3e}")
    print(f"Число шагов: {steps}")


if __name__ == "__main__":
    main()
