import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve

# ----------------------------
# Ścieżki do pliku obrazu i folderu wynikowego
# ----------------------------
IMAGE_PATH = './Images/hidden-symbols.tif'
OUTPUT_DIR = './Images-converted-Z8'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Lokalna korekcja histogramu dla obrazu w skali szarości
# image - tablica NumPy z wartościami pikseli
# mask_size - rozmiar okna (maski) do lokalnej analizy (np. 3,5,7...)
# Zwraca lokalnie wyrównany obraz jako tablicę uint8
# ----------------------------
def local_histogram_equalization(image, mask_size):
    height, width = image.shape
    pad = mask_size // 2

    # Dopełnienie obrazu odbiciem brzegów, aby maska mieściła się na krawędziach
    padded = np.pad(image, pad, mode='reflect')

    # Tablica wynikowa
    equalized = np.zeros_like(image)

    # Iteracja po każdym pikselu obrazu
    for i in range(height):
        for j in range(width):
            # Wycięcie lokalnego okna wokół piksela
            window = padded[i:i + mask_size, j:j + mask_size]

            # Histogram wartości w oknie (256 poziomów od 0 do 255)
            hist, _ = np.histogram(window, bins=256, range=(0, 255))

            # Dystrybuanta histogramu (CDF)
            cdf = hist.cumsum()

            # Normalizacja CDF do zakresu [0,255] z małą poprawką by uniknąć dzielenia przez zero
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-7)

            # Przypisanie wyrównanej wartości do piksela
            equalized[i, j] = cdf[image[i, j]]

    return equalized.astype(np.uint8)


# ----------------------------
# Lokalna poprawa statystyczna obrazu - wyostrzenie i korekcja gamma
# image - tablica NumPy obrazu
# mask_size - rozmiar maski do wyliczenia lokalnej średniej
# a - współczynnik wzmocnienia wyostrzenia (domyślnie 1.5)
# gamma - współczynnik korekcji gamma (domyślnie 1.0 - brak korekcji)
# Zwraca poprawiony obraz jako uint8
# ----------------------------
def local_statistics_enhancement(image, mask_size, a=1.5, gamma=1.0):
    # Filtr uśredniający (jednostajna maska)
    kernel = np.ones((mask_size, mask_size)) / (mask_size ** 2)

    # Obliczenie lokalnej średniej konwolucją (tryb odbicia brzegów)
    local_mean = convolve(image.astype(float), kernel, mode='reflect')

    # Wyostrzenie: oryginał + wzmocniona różnica między oryginałem a lokalną średnią
    sharpened = image + a * (image - local_mean)

    # Przycięcie wartości do zakresu [0,255]
    sharpened = np.clip(sharpened, 0, 255)

    # Korekcja gamma (jeśli inna niż 1)
    if gamma != 1.0:
        sharpened = 255 * (sharpened / 255) ** gamma

    return sharpened.astype(np.uint8)


# ----------------------------
# Funkcja przetwarzająca obraz dla różnych rozmiarów masek:
# - lokalne wyrównanie histogramu
# - lokalna poprawa statystyczna
# Wyniki są zapisywane i wyświetlane
# ----------------------------
def process_image(image, mask_sizes):
    for size in mask_sizes:
        # Lokalna korekcja histogramu
        eq = local_histogram_equalization(image, size)

        # Lokalna poprawa statystyczna (wyostrzenie)
        enhanced = local_statistics_enhancement(image, size)

        # Konwersja wyników do obrazów PIL
        eq_img = Image.fromarray(eq)
        enhanced_img = Image.fromarray(enhanced)

        # Ścieżki do zapisu wyników
        eq_path = os.path.join(OUTPUT_DIR, f"hidden_eq_{size}x{size}.tif")
        enhanced_path = os.path.join(OUTPUT_DIR, f"hidden_stat_{size}x{size}.tif")

        # Zapis obrazów
        eq_img.save(eq_path)
        enhanced_img.save(enhanced_path)
        print(f"Zapisano: {eq_path}, {enhanced_path}")

        # Wyświetlanie oryginału i wyników obok siebie
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Oryginał'), plt.axis('off')
        plt.subplot(1, 3, 2), plt.imshow(eq, cmap='gray'), plt.title(f'Lokalne wyrównanie ({size}x{size})'), plt.axis('off')
        plt.subplot(1, 3, 3), plt.imshow(enhanced, cmap='gray'), plt.title(f'Poprawa statystyczna ({size}x{size})'), plt.axis('off')
        plt.tight_layout()
        plt.show()


# ----------------------------
# Funkcja główna programu
# - próba wczytania obrazu
# - jeśli brak pliku, generuje sztuczny obraz z kołem i szumem
# - wykonuje przetwarzanie dla podanych rozmiarów masek
# ----------------------------
def main():
    try:
        image = Image.open(IMAGE_PATH).convert('L')
        image_np = np.array(image)
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {IMAGE_PATH}, generuję przykładowy obraz...")

        # Tworzenie pustego obrazu 200x200 pikseli
        image_np = np.zeros((200, 200), dtype=np.uint8)

        # Rysowanie koła o promieniu 30 pikseli w środku obrazu
        for i in range(200):
            for j in range(200):
                if (i - 100) ** 2 + (j - 100) ** 2 < 900:
                    image_np[i, j] = 150

        # Dodanie szumu normalnego (średnia=0, std=30)
        noise = np.random.normal(0, 30, image_np.shape)

        # Dodanie szumu i ograniczenie wartości do [0,255]
        image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)

    # Lista rozmiarów masek do analizy
    mask_sizes = [3, 5, 7, 9, 11]

    # Przetworzenie obrazu
    process_image(image_np, mask_sizes)


# ----------------------------
# Uruchomienie programu
# ----------------------------
if __name__ == "__main__":
    main()
