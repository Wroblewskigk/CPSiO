import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


#ZADANIE1###############################################################################################################

def multiply_constant(image_path, c):
    img = Image.open(image_path)
    img = img.convert('L')  # Konwersja do skali szarości

    # Tworzenie tablicy LUT
    lut = [min(int(i * c), 255) for i in range(256)]

    return img.point(lut)


# Przykład użycia
result = multiply_constant('chest_xray.tif', 1.5)
result.show()

#ZADANIE2###############################################################################################################

def log_transform(image_path):
    img = Image.open(image_path)
    max_pixel = 255
    c = 255 / math.log(1 + max_pixel)

    lut = [int(c * math.log(1 + i)) for i in range(256)]

    return img.point(lut)


# Przykład użycia
log_image = log_transform('spectrum.tif')
log_image.show()

#ZADANIE3###############################################################################################################

def contrast_transform(r, m=0.45, e=8):
    return 1 / (1 + (m / (r / 255)) ** e) * 255

# Wykres transformacji
x = np.linspace(1, 255, 255)
y = np.array([contrast_transform(xi) for xi in x])
plt.plot(x, y)
plt.title('Funkcja zmiany kontrastu')
plt.show()

# Implementacja na obrazie
def apply_contrast(image_path, m=0.45, e=8):
    img = Image.open(image_path)
    pixels = img.load()

    for i in range(img.width):
        for j in range(img.height):
            r = pixels[i, j]
            pixels[i, j] = int(contrast_transform(r, m, e))

    return img

#ZADANIE4###############################################################################################################

def gamma_correction(image_path, gamma=1.0):
    # Wczytaj obraz i konwertuj do trybu "L" (skala szarości) lub "RGB"
    img = Image.open(image_path)
    if img.mode not in ('L', 'RGB'):
        img = img.convert('RGB')

    # Oblicz odwrotność gamma
    inv_gamma = 1.0 / gamma

    # Generuj tablicę LUT
    lut = np.array([((i / 255.0) ** inv_gamma) * 255
                    for i in range(256)], dtype=np.uint8)

    # Zastosuj transformację do obrazu
    return img.point(lut.tolist())

# Przykład użycia
result = gamma_correction('aerial_view.tif', gamma=0.5)
result.show()
