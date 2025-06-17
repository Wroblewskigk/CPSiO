import os
import math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ----------------------------
# Ścieżki do folderów na obrazy wejściowe i wynikowe
# ----------------------------
IMAGE_DIR = './Images'
OUTPUT_DIR = './Images-converted-Z6'

# Utworzenie folderu na wyniki, jeśli nie istnieje
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Funkcja wczytująca obraz i konwertująca go do skali szarości
# ----------------------------
def load_grayscale_image(filename):
    path = os.path.join(IMAGE_DIR, filename)
    img = Image.open(path).convert('L')  # Konwersja do skali szarości (L)
    return img

# ----------------------------
# Funkcja zapisu obrazu z dopiskiem suffix do nazwy pliku
# ----------------------------
def save_image(image, original_name, suffix):
    name, _ = os.path.splitext(original_name)  # Oddzielenie nazwy od rozszerzenia
    save_path = os.path.join(OUTPUT_DIR, f"{name}_{suffix}.tif")
    image.save(save_path)
    print(f"Zapisano: {save_path}")

# ----------------------------
# ZADANIE 6a: Mnożenie obrazu przez stałą c
# Tworzy LUT (Look-Up Table) i nakłada na obraz, ograniczając max do 255
# ----------------------------
def multiply_constant(image, c):
    lut = [min(int(i * c), 255) for i in range(256)]
    return image.point(lut)

# ----------------------------
# ZADANIE 6b: Transformacja logarytmiczna
# ----------------------------
def log_transform(image):
    c = 255 / math.log(1 + 255)  # Normalizacja stałej c
    lut = [int(c * math.log(1 + i)) for i in range(256)]  # LUT oparty na funkcji logarytmicznej
    return image.point(lut)

# ----------------------------
# ZADANIE 6c: Transformacja kontrastu
# Funkcja pomocnicza do obliczania pojedynczej wartości kontrastu
# r_norm to poziom szarości znormalizowany do [0,1]
# ----------------------------
def contrast_transform_value(r, m=0.45, e=8):
    r_norm = r / 255
    # Funkcja kontrastu oparta na wzorze logistycznym, zwraca wartość w zakresie 0-255
    return int((1 / (1 + (m / r_norm) ** e)) * 255) if r > 0 else 0

# Funkcja nakładająca transformację kontrastu na cały obraz przez LUT
def contrast_transform(image, m=0.45, e=8):
    lut = [contrast_transform_value(r, m, e) for r in range(256)]
    return image.point(lut)

# ----------------------------
# Korekcja gamma
# Przyjmuje gamma i tworzy LUT z korektą gamma
# ----------------------------
def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    lut = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
    return image.point(lut)

# ----------------------------
# Funkcja wyświetlająca wykres funkcji transformacji kontrastu
# ----------------------------
def plot_contrast_function(m=0.45, e=8):
    x = np.linspace(1, 255, 255)  # Zakres wejściowy poziomów szarości (bez 0)
    y = [contrast_transform_value(xi, m, e) for xi in x]
    plt.plot(x, y)
    plt.title('Funkcja transformacji kontrastu')
    plt.xlabel('Poziom wejściowy r')
    plt.ylabel('Poziom wyjściowy s')
    plt.grid(True)
    plt.show()

# ----------------------------
# Główna część programu
# Wykonuje wszystkie transformacje dla wybranych obrazów
# ----------------------------
if __name__ == "__main__":
    # Mnożenie obrazu przez stałą c=1.5
    for fname in ['chest-xray.tif', 'pollen-dark.tif', 'spectrum.tif']:
        img = load_grayscale_image(fname)
        result = multiply_constant(img, c=1.5)
        result.show(title=f"{fname} * 1.5")
        save_image(result, fname, "multiply1.5")

    # Transformacja logarytmiczna na obrazie 'spectrum.tif'
    spectrum = load_grayscale_image('spectrum.tif')
    log_img = log_transform(spectrum)
    log_img.show(title="Log transform")
    save_image(log_img, 'spectrum.tif', "log")

    # Transformacja kontrastu - wykres i zastosowanie na 'spectrum.tif'
    plot_contrast_function()
    contrast_img = contrast_transform(spectrum)
    contrast_img.show(title="Contrast transform")
    save_image(contrast_img, 'spectrum.tif', "contrast")

    # Korekcja gamma na obrazie 'aerial_view.tif' z gamma=0.5
    aerial = load_grayscale_image('aerial_view.tif')
    gamma_img = gamma_correction(aerial, gamma=0.5)
    gamma_img.show(title="Gamma correction")
    save_image(gamma_img, 'aerial_view.tif', "gamma0.5")
