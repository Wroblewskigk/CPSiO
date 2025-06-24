import os
import math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

IMAGE_DIR = './Images'
OUTPUT_DIR = './Images-converted-Z6'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image_as_array(filename):
    path = os.path.join(IMAGE_DIR, filename)
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.uint8)

def save_comparison(original_array, processed_array, original_name, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_array, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Oryginał')
    axes[0].axis('off')
    axes[1].imshow(processed_array, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Przetworzony ({suffix})')
    axes[1].axis('off')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(original_name)[0]}_{suffix}_comparison.png")
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)
    print(f"Zapisano porównanie: {save_path}")

def multiply_constant_np(img_array, c):
    result = np.clip(img_array * c, 0, 255).astype(np.uint8)
    return result

def log_transform_np(img_array):
    c = 255 / math.log(1 + 255)
    # Konwersja do float, aby uniknąć problemów z int i log
    img_float = img_array.astype(np.float32)
    # Dodanie epsilon, by uniknąć log(0)
    epsilon = 1e-10
    result = c * np.log(1 + img_float + epsilon)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def contrast_transform_value_np(r, m=0.45, e=8):
    r_norm = r / 255
    return int((1 / (1 + (m / r_norm) ** e)) * 255) if r > 0 else 0

def contrast_transform_np(img_array, m=0.45, e=8):
    lut = np.array([contrast_transform_value_np(r, m, e) for r in range(256)], dtype=np.uint8)
    result = lut[img_array]
    return result

def plot_contrast_function(m=0.45, e=8, output_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(256)
    y = np.array([contrast_transform_value_np(r, m, e) for r in x])

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'T(r), m={m}, e={e}')
    plt.title('Funkcja transformacji kontrastu T(r)')
    plt.xlabel('Poziom wejściowy r')
    plt.ylabel('Poziom wyjściowy T(r)')
    plt.grid(True)
    plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Zapisano wykres funkcji kontrastu: {output_path}")
    else:
        plt.show()


def gamma_correction_np(img_array, gamma=1.0):
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    result = lut[img_array]
    return result

if __name__ == "__main__":
    # Mnożenie przez 1.5
    for fname in ['chest-xray.tif', 'pollen-dark.tif', 'spectrum.tif']:
        original = load_image_as_array(fname)
        processed = multiply_constant_np(original, 1.5)
        save_comparison(original, processed, fname, "multiply1.5")

    # Log transformacja
    spectrum = load_image_as_array('spectrum.tif')
    log_img = log_transform_np(spectrum)
    save_comparison(spectrum, log_img, 'spectrum.tif', 'log')

    # Transformacja kontrastu
    for fname in ['chest-xray.tif', 'einstein-low-contrast.tif', 'pollen-lowcontrast.tif']:
        original = load_image_as_array(fname)
        contrast_img = contrast_transform_np(original)
        save_comparison(original, contrast_img, fname, 'contrast')
    plot_contrast_function(output_path=os.path.join(OUTPUT_DIR, 'contrast_function.png'))

    # Korekcja gamma
    aerial = load_image_as_array('aerial_view.tif')
    gamma_img = gamma_correction_np(aerial, gamma=0.5)
    save_comparison(aerial, gamma_img, 'aerial_view.tif', 'gamma0.5')
