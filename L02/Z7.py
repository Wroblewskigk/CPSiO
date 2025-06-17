import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ----------------------------
# Ścieżki do folderów z obrazami wejściowymi i wynikowymi
# ----------------------------
IMAGE_DIR = './Images'
OUTPUT_DIR = './Images-converted-Z7'

# Utworzenie folderu na wyniki, jeśli nie istnieje
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Funkcja wyrównująca histogram obrazu
# img_array - tablica NumPy z wartościami pikseli obrazu w skali szarości
# Zwraca wyrównany obraz jako tablicę NumPy (uint8)
# ----------------------------
def equalize_histogram(img_array):
    # Obliczenie histogramu - liczba pikseli dla każdej wartości od 0 do 255
    hist, _ = np.histogram(img_array.flatten(), 256, [0, 256])

    # Obliczenie dystrybuanty (CDF) histogramu
    cdf = hist.cumsum()

    # Maskowanie zer (żeby nie dzielić przez zero)
    cdf_masked = np.ma.masked_equal(cdf, 0)

    # Minimalna i maksymalna wartość CDF (poza zerami)
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()

    # Normalizacja CDF do zakresu [0,255]
    cdf_masked = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)

    # Wypełnienie zamaskowanych miejsc zerami i konwersja na uint8
    cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')

    # Zastosowanie przekształcenia CDF do oryginalnej tablicy pikseli
    img_eq = cdf_final[img_array]

    return img_eq


# ----------------------------
# Funkcja wyświetlająca obrazy i histogramy oryginalne i wyrównane
# original - oryginalny obraz jako tablica NumPy
# equalized - wyrównany obraz jako tablica NumPy
# title - tytuł wykresu (zazwyczaj nazwa pliku)
# ----------------------------
def plot_histograms(original, equalized, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Oryginalny obraz
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Oryginał')
    axes[0, 0].axis('off')

    # Histogram oryginału
    axes[0, 1].hist(original.flatten(), bins=256, color='blue')
    axes[0, 1].set_title('Histogram oryginału')

    # Obraz po wyrównaniu
    axes[1, 0].imshow(equalized, cmap='gray')
    axes[1, 0].set_title('Po wyrównaniu')
    axes[1, 0].axis('off')

    # Histogram po wyrównaniu
    axes[1, 1].hist(equalized.flatten(), bins=256, color='green')
    axes[1, 1].set_title('Histogram po wyrównaniu')

    # Tytuł całego wykresu
    fig.suptitle(f'Wyrównanie histogramu: {title}')
    plt.tight_layout()
    plt.show()


# ----------------------------
# Funkcja wykonująca pełne przetwarzanie obrazu:
# - wczytanie obrazu
# - wyrównanie histogramu
# - wyświetlenie obrazów i histogramów
# - zapis wyniku do pliku
# ----------------------------
def process_image(filename):
    # Ścieżka do pliku
    image_path = os.path.join(IMAGE_DIR, filename)

    # Wczytanie i konwersja do skali szarości
    img = Image.open(image_path).convert('L')

    # Konwersja do tablicy NumPy
    img_array = np.array(img)

    # Wyrównanie histogramu
    img_eq_array = equalize_histogram(img_array)

    # Wyświetlenie wykresów obrazów i histogramów
    plot_histograms(img_array, img_eq_array, filename)

    # Konwersja wyniku do obiektu Image i zapis do pliku
    img_eq = Image.fromarray(img_eq_array)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_DIR, f"{name}_equalized.tif")
    img_eq.save(output_path)

    print(f"Zapisano wyrównany obraz: {output_path}")


# ----------------------------
# Lista plików obrazów do przetworzenia
# ----------------------------
images = [
    'chest_xray.tif',
    'pollen-dark.tif',
    'pollen-ligt.tif',
    'pollen-lowcontrast.tif',
    'pout.tif',
    'spectrum.tif'
]

# ----------------------------
# Główna część programu
# Przetwarza każdy obraz z listy, obsługuje ewentualne błędy
# ----------------------------
if __name__ == "__main__":
    for img_name in images:
        try:
            print(f"Przetwarzanie: {img_name}")
            process_image(img_name)
        except Exception as e:
            print(f"Błąd przetwarzania {img_name}: {e}")
