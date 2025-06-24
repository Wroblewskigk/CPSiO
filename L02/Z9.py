import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

INPUT_DIR = "./Images"
OUTPUT_DIR = "./Images-converted-Z9"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    img = np.array(Image.open(os.path.join(INPUT_DIR, path)).convert('L'))
    results = {'Original': img}
    for size in kernel_sizes:
        results[f'Average {size}x{size}'] = apply_average_filter(img, size)
        results[f'Median {size}x{size}'] = apply_median_filter(img, size)
        results[f'Min {size}x{size}'] = apply_min_filter(img, size)
        results[f'Max {size}x{size}'] = apply_max_filter(img, size)
    return results


def plot_results(results, title, image_name):
    keys = list(results.keys())
    n = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 5, rows * 5))
    for i, key in enumerate(keys, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(results[key], cmap='gray')
        plt.title(key)
        plt.axis('off')

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Zapis całego porównania do pliku
    filename = f"{image_name}_comparison.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    print(f"Zapisano porównanie do: {filename}")

    plt.show()


def plot_histograms(results, kernel_sizes, image_name):
    plt.figure(figsize=(10, 6))

    # Histogram oryginału
    plt.hist(results['Original'].ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7, label='Oryginał')

    # Histogramy filtrów medianowych
    for size in kernel_sizes:
        plt.hist(results[f'Median {size}x{size}'].ravel(), bins=256, range=(0, 256), alpha=0.5,
                 label=f'Mediana {size}x{size}')

    plt.legend()
    plt.title('Histogramy obrazu oryginalnego i filtrów medianowych')
    plt.xlabel('Wartość piksela')
    plt.ylabel('Liczba pikseli')
    plt.tight_layout()

    # Zapis histogramu do pliku
    filename = f"{image_name}_histograms.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    print(f"Zapisano histogramy do: {filename}")

    plt.show()


kernel_sizes = [3, 5, 7]
images = {
    'pepper_only': 'cboard_pepper_only.tif',
    'salt_only': 'cboard_salt_only.tif',
    'salt_pepper': 'cboard_salt_pepper.tif'
}

for image_name, path in images.items():
    print(f"\nPrzetwarzanie obrazu: {path}")
    results = process_image(path, kernel_sizes)
    plot_results(results, f"Redukcja szumu: {image_name.replace('_', ' ')}", image_name)
    plot_histograms(results, kernel_sizes, image_name)
