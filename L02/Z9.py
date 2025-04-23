import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def apply_average_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[i:i + kernel_size, j:j + kernel_size]
            filtered[i, j] = np.mean(neighborhood)

    return filtered.astype(np.uint8)


def apply_median_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[i:i + kernel_size, j:j + kernel_size].flatten()
            filtered[i, j] = np.median(neighborhood)

    return filtered.astype(np.uint8)


def apply_min_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[i:i + kernel_size, j:j + kernel_size]
            filtered[i, j] = np.min(neighborhood)

    return filtered.astype(np.uint8)


def apply_max_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[i:i + kernel_size, j:j + kernel_size]
            filtered[i, j] = np.max(neighborhood)

    return filtered.astype(np.uint8)


def process_image(path, kernel_sizes):
    img = np.array(Image.open(path).convert('L'))
    results = {'Original': img}

    for size in kernel_sizes:
        avg = apply_average_filter(img, size)
        med = apply_median_filter(img, size)
        mn = apply_min_filter(img, size)
        mx = apply_max_filter(img, size)

        results.update({
            f'Średnia {size}x{size}': avg,
            f'Mediana {size}x{size}': med,
            f'Min {size}x{size}': mn,
            f'Max {size}x{size}': mx
        })

    return results


def plot_results(results, title):
    plt.figure(figsize=(20, 10))
    keys = list(results.keys())

    for i, (key, img) in enumerate(results.items(), 1):
        plt.subplot(2, 4, i)
        plt.imshow(img, cmap='gray')
        plt.title(key)
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# Parametry eksperymentu
kernel_sizes = [3, 5, 7]
images = {
    'Tylko pieprz': 'cboard_pepper_only.tif',
    'Tylko sól': 'cboard_salt_only.tif',
    'Mieszany szum': 'cboard_salt_pepper.tif'
}

# Przeprowadzenie eksperymentów
for name, path in images.items():
    results = process_image(path, kernel_sizes)
    plot_results(results, f"Analiza redukcji szumu: {name}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(results['Original'].ravel(), 256, [0, 256], color='gray')
    plt.title('Histogram oryginału')

    plt.subplot(1, 2, 2)
    for size in kernel_sizes:
        plt.hist(results[f'Mediana {size}x{size}'].ravel(), 256, [0, 256], alpha=0.5, label=f'Mediana {size}x{size}')
    plt.legend()
    plt.title('Porównanie histogramów po filtracji medianowej')
    plt.show()
