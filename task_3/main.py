"""Double pendulum: trajectory, animation and sensitivity to initial data."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import solve_ivp


M1 = M2 = 1.0
L1 = L2 = 1.0
G = 9.81
DELTA = 1e-6
T = np.linspace(0.0, 20.0, 1000)
Y0_A = (np.pi / 2, 0.0, np.pi / 2 + 0.01, 0.0)
Y0_B = (np.pi / 2, 0.0, np.pi / 2 + 0.01 + DELTA, 0.0)


def right_hand_side(t: float, y: np.ndarray) -> tuple[float, float, float, float]:
    """Equation (5.5) from the assignment; y = (theta1, omega1, theta2, omega2)."""
    theta1, omega1, theta2, omega2 = y
    difference = theta1 - theta2
    denominator = 2 * M1 + M2 - M2 * np.cos(2 * difference)
    acceleration1 = (
        -G * (2 * M1 + M2) * np.sin(theta1)
        - M2 * G * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(difference) * M2
        * (omega2**2 * L2 + omega1**2 * L1 * np.cos(difference))
    ) / (L1 * denominator)
    acceleration2 = (
        2 * np.sin(difference)
        * (omega1**2 * L1 * (M1 + M2)
           + G * (M1 + M2) * np.cos(theta1)
           + omega2**2 * L2 * M2 * np.cos(difference))
    ) / (L2 * denominator)
    return omega1, acceleration1, omega2, acceleration2


def solve(y0: tuple[float, float, float, float], tolerance: float):
    result = solve_ivp(
        right_hand_side, (T[0], T[-1]), y0,
        method="RK45", t_eval=T, rtol=tolerance, atol=tolerance,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def coordinates(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta1, _, theta2, _ = y
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


def energy(y: np.ndarray) -> np.ndarray:
    theta1, omega1, theta2, omega2 = y
    kinetic = (
        (M1 + M2) * L1**2 * omega1**2 / 2
        + M2 * L2**2 * omega2**2 / 2
        + M2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )
    potential = -(M1 + M2) * G * L1 * np.cos(theta1) - M2 * G * L2 * np.cos(theta2)
    return kinetic + potential


def make_animation(t: np.ndarray, xy: tuple[np.ndarray, ...]) -> FuncAnimation:
    x1, y1, x2, y2 = xy
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-(L1 + L2) * 1.1, (L1 + L2) * 1.1)
    ax.set_ylim(-(L1 + L2) * 1.1, (L1 + L2) * 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title("Двойной маятник")
    rods, = ax.plot([], [], "o-", lw=2)
    trace, = ax.plot([], [], "-", lw=1, alpha=0.7)

    frames = range(0, len(t), 5)

    def update(i: int):
        rods.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        trace.set_data(x2[:i + 1], y2[:i + 1])
        ax.set_title(f"Двойной маятник, t = {t[i]:.1f} с")
        return rods, trace

    return FuncAnimation(fig, update, frames=frames, interval=100, blit=False)


def main(show: bool = True, save_gif: bool = False) -> None:
    accurate_a = solve(Y0_A, 1e-8)
    accurate_b = solve(Y0_B, 1e-8)
    loose_a = solve(Y0_A, 1e-5)
    xy_a = coordinates(accurate_a.y)
    xy_b = coordinates(accurate_b.y)
    output = Path(__file__).resolve().parent

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy_a[2], xy_a[3])
    ax.set(xlabel="x, м", ylabel="y, м", title="Траектория нижнего груза, случай A")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, xy, theta2 in zip(axes, (xy_a, xy_b), (Y0_A[2], Y0_B[2])):
        ax.plot(xy[2], xy[3])
        ax.set_title(f"θ₂(0) = {theta2:.9f} рад")
        ax.set_xlabel("x, м")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y, м")
    fig.tight_layout()
    fig.savefig(output / "trajectories_comparison.png", dpi=150)

    e_accurate = energy(accurate_a.y)
    e_loose = energy(loose_a.y)
    drift_accurate = np.abs(e_accurate - e_accurate[0]) / abs(e_accurate[0])
    drift_loose = np.abs(e_loose - e_loose[0]) / abs(e_loose[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(T, drift_accurate, label="rtol = atol = 1e-8")
    ax.plot(T, drift_loose, label="rtol = atol = 1e-5")
    ax.set(xlabel="t, с", ylabel="|E(t) − E(0)| / |E(0)|", title="Относительное изменение энергии")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "energy_comparison.png", dpi=150)
    print(f"Макс. изменение энергии (1e-8): {drift_accurate.max():.3e}")
    print(f"Макс. изменение энергии (1e-5): {drift_loose.max():.3e}")

    if show or save_gif:
        animation = make_animation(T, xy_a)
        if save_gif:
            animation.save(output / "pendulum.gif", writer="pillow", fps=10)
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening windows")
    parser.add_argument("--save-gif", action="store_true", help="Also save the animation as a GIF")
    args = parser.parse_args()
    main(show=not args.no_show, save_gif=args.save_gif)
