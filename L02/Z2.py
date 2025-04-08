import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def load_image(path):
    """Wczytuje obraz i konwertuje do float64 bez normalizacji"""
    img = io.imread(path)
    return img.astype(np.float64)

# Ścieżki do obrazów (zmień na prawidłowe!)
paths = {
    "chest_xray": "chest_xray.tif",
    "pollen": "pollen-dark.tif",
    "spectrum": "spectrum.tif"
}

# Wczytaj wszystkie obrazy
images = {name: load_image(path) for name, path in paths.items()}

# --------------------------------------------------------------
# a) Mnożenie przez stałą
# --------------------------------------------------------------
def multiply(img, c):
    result = img * c
    return np.clip(result, 0, 255).astype(np.uint8)

results_multiply = {
    "chest_xray": multiply(images["chest_xray"], 2),
    "pollen": multiply(images["pollen"], 3),
    "spectrum": multiply(images["spectrum"], 0.5)
}

# --------------------------------------------------------------
# b) Transformacja logarytmiczna
# --------------------------------------------------------------
def log_transform(img, c=46):
    transformed = c * np.log1p(img)  # log1p(x) = log(1+x)
    return np.clip(transformed, 0, 255).astype(np.uint8)

spectrum_log = log_transform(images["spectrum"])

# --------------------------------------------------------------
# c) Transformacja kontrastu
# --------------------------------------------------------------
def contrast_transform(img, m=0.5, e=8):
    transformed = 1 / (1 + (m / (img + 1e-10))**e)
    return (255 * transformed).astype(np.uint8)

spectrum_contrast = contrast_transform(images["spectrum"])

# --------------------------------------------------------------
# Wizualizacja
# --------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

# Kolumna 1: Oryginały
axes[0,0].imshow(images["chest_xray"], cmap='gray', vmin=0, vmax=255)
axes[0,0].set_title('Klatka piersiowa (oryginał)')

axes[1,0].imshow(images["pollen"], cmap='gray', vmin=0, vmax=255)
axes[1,0].set_title('Pyłek (oryginał)')

axes[2,0].imshow(images["spectrum"], cmap='gray', vmin=0, vmax=255)
axes[2,0].set_title('Widmo (oryginał)')

# Kolumna 2: Mnożenie
axes[0,1].imshow(results_multiply["chest_xray"], cmap='gray')
axes[0,1].set_title('c=2: Wzmocnienie kości')

axes[1,1].imshow(results_multiply["pollen"], cmap='gray')
axes[1,1].set_title('c=3: Rozjaśnienie pyłku')

axes[2,1].imshow(results_multiply["spectrum"], cmap='gray')
axes[2,1].set_title('c=0.5: Redukcja prześwietlenia')

# Kolumna 3: Transformacje dla spectrum
axes[2,2].imshow(spectrum_log, cmap='gray')
axes[2,2].set_title('Transformacja logarytmiczna')

axes[1,2].imshow(spectrum_contrast, cmap='gray')
axes[1,2].set_title('Regulacja kontrastu (m=0.5, e=8)')

# Ukryj puste wykresy
axes[0,2].axis('off')

plt.tight_layout()
plt.show()
