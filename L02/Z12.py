import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Ustawienie backendu Matplotlib na TkAgg do wyświetlania GUI
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter  # Import filtrów: Gaussa i medianowego

# Ścieżki do katalogów wejściowego i wyjściowego
INPUT_DIR = "./Images"
OUTPUT_DIR = "./Images-converted-Z12"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Tworzymy katalog wyjściowy, jeśli nie istnieje

# Funkcja do wczytywania obrazu w skali szarości
def load_image(filename):
    path = os.path.join(INPUT_DIR, filename)
    return np.array(Image.open(path).convert('L'))

# Funkcja do zapisywania obrazu na dysk
def save_image(array, name):
    # Konwersja do uint8 i ograniczenie wartości do [0, 255]
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    img.save(os.path.join(OUTPUT_DIR, name))

# Etap 1: Rozciąganie histogramu obrazu, by zwiększyć kontrast
def histogram_stretching(image):
    imin, imax = np.min(image), np.max(image)  # minimalna i maksymalna wartość pikseli
    stretched = (image - imin) * 255.0 / (imax - imin)  # liniowe przeskalowanie na pełen zakres 0-255
    return stretched.astype(np.uint8)

# Etap 2: Usuwanie szumu za pomocą filtru medianowego
def denoise_median(image, size=3):
    return median_filter(image, size=size)  # medianowy filtr o zadanym rozmiarze okna (domyślnie 3x3)

# Etap 3: Wyostrzanie obrazu metodą unsharp masking
def unsharp_mask(image, sigma=1.0, k=1.5):
    blurred = gaussian_filter(image, sigma=sigma)  # rozmycie Gaussa o odchyleniu sigma
    mask = image - blurred                          # maska - różnica między oryginałem a rozmyciem
    sharpened = image + k * mask                     # dodanie wzmocnionej maski do oryginału
    return np.clip(sharpened, 0, 255).astype(np.uint8)  # ograniczenie wartości do 0-255

# Etap 4: Opcjonalne wygładzenie końcowe filtrem dolnoprzepustowym Gaussa
def final_smoothing(image, sigma=0.5):
    return gaussian_filter(image, sigma=sigma).astype(np.uint8)

# Główna funkcja przetwarzająca obraz wg powyższych etapów
def enhance_bonescan(image_path):
    original = load_image(image_path)  # Wczytanie oryginalnego obrazu

    # Kolejne etapy przetwarzania obrazu
    stretched = histogram_stretching(original)
    denoised = denoise_median(stretched, size=3)
    sharpened = unsharp_mask(denoised, sigma=1.0, k=1.5)
    smoothed = final_smoothing(sharpened, sigma=0.5)

    # Przechowujemy wyniki etapów w słowniku (etykieta: obraz)
    steps = {
        "01_Oryginał": original,
        "02_Rozciąganie histogramu": stretched,
        "03_Filtr medianowy": denoised,
        "04_Wyostrzanie (unsharp)": sharpened,
        "05_Wygładzanie końcowe": smoothed
    }

    # Wyświetlanie i zapisywanie wyników
    plt.figure(figsize=(15, 8))
    for i, (label, img) in enumerate(steps.items(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(label)
        plt.axis('off')
        save_image(img, f"Z12_{label.replace(' ', '_')}.tif")  # Zapis na dysk
    plt.tight_layout()
    plt.show()

# Uruchomienie przetwarzania jeśli plik jest głównym modułem
if __name__ == "__main__":
    enhance_bonescan("bonescan.tif")
