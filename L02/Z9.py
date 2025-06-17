import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ----------------------------
# Ścieżki do katalogów wejściowego i wyjściowego
# ----------------------------
INPUT_DIR = "./Images"
OUTPUT_DIR = "./Images-converted-Z9"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Utworzenie folderu wyjściowego, jeśli nie istnieje


# ----------------------------
# Filtr uśredniający (średnia arytmetyczna w oknie)
# image - tablica obrazu w skali szarości
# kernel_size - rozmiar maski (np. 3, 5, 7)
# Zwraca obraz po filtracji jako uint8
# ----------------------------
def apply_average_filter(image, kernel_size):
    pad = kernel_size // 2
    # Dopełnienie obrazu odbiciem brzegów (tryb 'reflect')
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)  # Tablica wynikowa float, by uśrednianie było precyzyjne

    # Przechodzimy przez każdy piksel oryginalnego obrazu
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Wycinamy lokalne okno
            neighborhood = padded[i:i + kernel_size, j:j + kernel_size]
            # Obliczamy średnią wartość w oknie
            filtered[i, j] = np.mean(neighborhood)

    return filtered.astype(np.uint8)


# ----------------------------
# Filtr medianowy
# Działa podobnie jak średni, ale zwraca medianę wartości z okna
# ----------------------------
def apply_median_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[i:i + kernel_size, j:j + kernel_size].flatten()
            filtered[i, j] = np.median(neighborhood)

    return filtered.astype(np.uint8)


# ----------------------------
# Filtr min (minimum wartości w oknie)
# ----------------------------
def apply_min_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[i:i + kernel_size, j:j + kernel_size]
            filtered[i, j] = np.min(neighborhood)

    return filtered.astype(np.uint8)


# ----------------------------
# Filtr max (maksimum wartości w oknie)
# ----------------------------
def apply_max_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[i:i + kernel_size, j:j + kernel_size]
            filtered[i, j] = np.max(neighborhood)

    return filtered.astype(np.uint8)


# ----------------------------
# Funkcja przetwarzająca obraz:
# - wczytuje obraz z pliku
# - dla każdego rozmiaru maski aplikuje cztery filtry (średni, medianowy, min, max)
# - zwraca słownik wyników (obraz oryginalny i po filtrach)
# ----------------------------
def process_image(path, kernel_sizes):
    img = np.array(Image.open(os.path.join(INPUT_DIR, path)).convert('L'))
    results = {'Original': img}  # Zachowanie oryginału do porównania

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


# ----------------------------
# Funkcja wyświetlająca obrazy i zapisująca wyniki do plików
# results - słownik obrazów (klucz: nazwa, wartość: tablica obrazu)
# title - tytuł wykresu
# image_name - nazwa obrazu, używana w nazwach plików wynikowych
# ----------------------------
def plot_results(results, title, image_name):
    plt.figure(figsize=(20, 10))
    keys = list(results.keys())

    # Rysowanie obrazów w subplotach
    for i, (key, img) in enumerate(results.items(), 1):
        plt.subplot(4, 4, i)
        plt.imshow(img, cmap='gray')
        plt.title(key)
        plt.axis('off')

        # Zapis obrazu do pliku (oprócz oryginału)
        if key != 'Original':
            filename = f"{image_name}_{key.replace(' ', '_')}.tif"
            Image.fromarray(img).save(os.path.join(OUTPUT_DIR, filename))

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Funkcja rysująca histogramy:
# - histogram oryginału
# - histogramy obrazów po filtrze medianowym dla różnych rozmiarów masek
# ----------------------------
def plot_histograms(results, kernel_sizes):
    plt.figure(figsize=(15, 5))

    # Histogram oryginalnego obrazu
    plt.subplot(1, 2, 1)
    plt.hist(results['Original'].ravel(), 256, range=[0, 256], color='gray')
    plt.title('Histogram oryginału')

    # Histogramy dla filtrów medianowych w różnych rozmiarach
    plt.subplot(1, 2, 2)
    for size in kernel_sizes:
        plt.hist(results[f'Mediana {size}x{size}'].ravel(), 256, range=[0, 256], alpha=0.5, label=f'{size}x{size}')

    plt.legend()
    plt.title('Histogramy filtrów medianowych')
    plt.tight_layout()
    plt.show()


# ----------------------------
# Główna część programu:
# - definiuje rozmiary masek do filtrów
# - lista obrazów do przetworzenia
# - przetwarza kolejno każdy obraz i wyświetla wyniki
# ----------------------------
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
    plot_histograms(results, kernel_sizes)
