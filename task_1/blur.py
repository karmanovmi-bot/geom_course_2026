import numpy as np


def pad_image(image: np.ndarray, pad: int, mode: str) -> np.ndarray:
    """Add a border of zeros or repeated edge pixels to a 2D image."""
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    if pad < 0:
        raise ValueError("pad must be non-negative")
    if mode not in ("zero", "edge"):
        raise ValueError("mode must be 'zero' or 'edge'")
    if pad == 0:
        return image.copy()

    rows, cols = image.shape
    result = np.zeros((rows + 2 * pad, cols + 2 * pad), dtype=image.dtype)
    result[pad:pad + rows, pad:pad + cols] = image
    if mode == "edge":
        result[:pad, pad:pad + cols] = image[0, :]
        result[pad + rows:, pad:pad + cols] = image[-1, :]
        result[:, :pad] = result[:, pad:pad + 1]
        result[:, pad + cols:] = result[:, pad + cols - 1:pad + cols]
    return result


def box_blur(image: np.ndarray, kernel_size: int, mode: str) -> np.ndarray:
    """Replace each pixel with the mean of its square neighbourhood."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    padded = pad_image(image, kernel_size // 2, mode)
    rows, cols = image.shape
    result = np.empty((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            window = padded[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = np.mean(window)
    return result


def make_test_image(size: int = 40) -> np.ndarray:
    import random

    random.seed(0)
    img = np.zeros((size, size))
    for i in range(size):
        row = np.zeros(size)
        for j in range(size):
            base = 220.0 if (i // 5 + j // 5) % 2 == 0 else 40.0
            noise = random.uniform(-15, 15)
            row[j] = max(0.0, min(255.0, base + noise))
        img[i] = row
    return img


def main() -> None:
    import matplotlib.pyplot as plt

    image = make_test_image(size=40)
    blurred_k3 = box_blur(image, kernel_size=3, mode="edge")
    blurred_k9 = box_blur(image, kernel_size=9, mode="zero")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Исходное", "Blur k=3 (edge)", "Blur k=9 (zero)"]
    for ax, title, data in zip(axes, titles, (image, blurred_k3, blurred_k9)):
        ax.imshow(data, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
