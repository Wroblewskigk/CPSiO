import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Ustawienie backendu Matplotlib na TkAgg dla GUI
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter  # Funkcje do konwolucji i filtru Gaussa

INPUT_DIR = "./Images"                # Katalog z obrazami wejściowymi
OUTPUT_DIR = "./Images-converted-Z11"  # Katalog na wyniki
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Utwórz katalog, jeśli nie istnieje


# --- Filtr Sobela w różnych kierunkach ---
def sobel_edges(image):
    # Definicja masek Sobela: poziomy, pionowy i dwa ukośne
    sobel_h = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    sobel_v = np.array([[-1,  0, 1],
                        [-2,  0, 2],
                        [-1,  0, 1]])
    sobel_d1 = np.array([[-2, -1, 0],
                         [-1,  0, 1],
                         [ 0,  1, 2]])
    sobel_d2 = np.array([[ 0,  1, 2],
                         [-1,  0, 1],
                         [-2, -1, 0]])

    # Nakładanie masek konwolucją oraz pobieranie wartości bezwzględnej
    results = {
        'Sobel poziomy': np.abs(convolve(image, sobel_h)),
        'Sobel pionowy': np.abs(convolve(image, sobel_v)),
        'Sobel ukośny 1': np.abs(convolve(image, sobel_d1)),
        'Sobel ukośny 2': np.abs(convolve(image, sobel_d2))
    }
    # Ograniczenie wyników do zakresu [0, 255] i konwersja na uint8
    return {k: np.clip(v, 0, 255).astype(np.uint8) for k, v in results.items()}


# --- Wyostrzanie obrazu przez filtr Laplace'a ---
def laplacian_sharpen(image):
    # Maska Laplace'a wykrywająca krawędzie
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    lap = convolve(image, kernel)  # Wykonanie konwolucji Laplacjana
    sharp = image - lap            # Odejmujemy Laplacjan od oryginału - wyostrzanie
    # Ograniczamy zakres i konwertujemy do uint8
    return np.clip(sharp, 0, 255).astype(np.uint8)


# --- Unsharp mask i High Boost Filtering ---
def unsharp_mask(image, sigma=1.0, k=1.0):
    # Najpierw rozmywamy obraz filtrem Gaussa
    blurred = gaussian_filter(image, sigma=sigma)
    # Maskę tworzymy jako różnicę oryginału i rozmycia
    mask = image - blurred
    # Dodajemy skalowaną maskę do oryginału
    result = image + k * mask
    # Ograniczamy wynik do [0, 255]
    return np.clip(result, 0, 255).astype(np.uint8)


# --- Funkcje przetwarzające obrazy dla konkretnych zadań ---
def process_sobel(path):
    # Wczytanie obrazu jako skala szarości
    img = np.array(Image.open(os.path.join(INPUT_DIR, path)).convert('L'))
    results = {'Oryginał': img}
    # Dodanie wyników filtracji Sobela do słownika
    results.update(sobel_edges(img))
    return results


def process_laplacian(path):
    img = np.array(Image.open(os.path.join(INPUT_DIR, path)).convert('L'))
    sharp = laplacian_sharpen(img)
    return {'Oryginał': img, 'Laplacjan wyostrzony': sharp}


def process_unsharp(path):
    img = np.array(Image.open(os.path.join(INPUT_DIR, path)).convert('L'))
    results = {'Oryginał': img}
    # Unsharp z k=1 (standardowe wyostrzanie)
    results['Unsharp (k=1)'] = unsharp_mask(img, sigma=1.0, k=1.0)
    # High Boost z k=2 (silniejsze wyostrzanie)
    results['High Boost (k=2)'] = unsharp_mask(img, sigma=1.0, k=2.0)
    return results


# --- Funkcja do wyświetlania i zapisywania wyników ---
def plot_results(results, title, image_name):
    plt.figure(figsize=(15, 5))
    # Tworzymy subploty dla każdego obrazu w słowniku results
    for i, (label, img) in enumerate(results.items(), 1):
        plt.subplot(1, len(results), i)
        plt.imshow(img, cmap='gray')
        plt.title(label)
        plt.axis('off')

        # Zapisujemy każdy obraz poza oryginałem
        if label != 'Oryginał':
            fname = f"{image_name}_{label.replace(' ', '_')}.tif"
            Image.fromarray(img).save(os.path.join(OUTPUT_DIR, fname))

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# --- Główna część programu ---
if __name__ == "__main__":
    # Słowniki nazw i ścieżek obrazów do poszczególnych podzadań
    zad11a = {
        'circuitmask': 'circuitmask.tif',
        'testpat1': 'testpat1.png'
    }
    zad11b = {
        'blurry_moon': 'blurry-moon.tif'
    }
    zad11c = {
        'text_blurred': 'text-dipxe-blurred.tif'
    }

    # Przetwarzanie obrazów zadania 11a - Sobel
    for name, path in zad11a.items():
        print(f"\nZadanie 11a – Sobel: {path}")
        res = process_sobel(path)
        plot_results(res, f"Krawędzie Sobel – {name}", name)

    # Przetwarzanie obrazów zadania 11b - Laplacjan
    for name, path in zad11b.items():
        print(f"\nZadanie 11b – Laplacjan: {path}")
        res = process_laplacian(path)
        plot_results(res, f"Laplacjan Wyostrzanie – {name}", name)

    # Przetwarzanie obrazów zadania 11c - Unsharp mask i High Boost
    for name, path in zad11c.items():
        print(f"\nZadanie 11c – Unsharp/High Boost: {path}")
        res = process_unsharp(path)
        plot_results(res, f"Wyostrzanie – {name}", name)
