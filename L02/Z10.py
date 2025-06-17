import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# ----------------------------
# Filtr uśredniający (średnia arytmetyczna w oknie)
# image - tablica obrazu (2D)
# kernel_size - rozmiar okna filtru (np. 3, 5, 7)
# Zwraca obraz po filtracji jako uint8
# ----------------------------
def apply_average_filter(image, kernel_size):
    pad = kernel_size // 2
    # Dopełnienie obrazu odbiciem brzegów, aby filtry działały na krawędziach
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)  # float, by uśrednianie było precyzyjne

    # Przechodzimy po każdym pikselu oryginału
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Wycinamy lokalne okno (kernel_size x kernel_size)
            window = padded[i:i+kernel_size, j:j+kernel_size]
            # Obliczamy średnią wartość w oknie
            filtered[i, j] = np.mean(window)

    return filtered.astype(np.uint8)  # Konwersja do uint8


# ----------------------------
# Funkcja generująca jądro Gaussa
# kernel_size - rozmiar jądra (np. 3, 5, 7)
# sigma - odchylenie standardowe Gaussa
# Zwraca macierz jądra Gaussa (kernel_size x kernel_size) zsumowaną do 1
# ----------------------------
def gaussian_kernel(kernel_size, sigma=1.0):
    # Wektor współrzędnych od -k//2+1 do k//2
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    # Wzór jądra Gaussa 2D
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    # Normalizacja, by suma elementów wyniosła 1
    kernel /= np.sum(kernel)
    return kernel


# ----------------------------
# Filtr Gaussa
# image - obraz do przefiltrowania (2D)
# kernel_size - rozmiar jądra Gaussa
# sigma - odchylenie standardowe Gaussa
# Zwraca przefiltrowany obraz uint8
# ----------------------------
def apply_gaussian_filter(image, kernel_size, sigma=1.0):
    # Generujemy jądro Gaussa
    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)

    # Nakładamy jądro na każdy piksel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            # Mnożenie elementów i sumowanie
            filtered[i, j] = np.sum(window * kernel)

    return filtered.astype(np.uint8)


# ----------------------------
# Funkcja zapisująca obraz do pliku
# array - tablica obrazu
# output_path - ścieżka do pliku wyjściowego
# ----------------------------
def save_image(array, output_path):
    Image.fromarray(array).save(output_path)


# ----------------------------
# Główna funkcja:
# - wczytuje obraz
# - dla każdej maski aplikuje filtr uśredniający i filtry Gaussa dla różnych sigma
# - zapisuje wyniki do plików
# - wyświetla oryginał i wyniki filtracji
# ----------------------------
def show_and_save_filters(image_path, kernel_sizes, sigmas, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = np.array(Image.open(image_path).convert('L'))  # Wczytanie obrazu w skali szarości

    for ksize in kernel_sizes:
        # Filtr uśredniający
        avg_filtered = apply_average_filter(image, ksize)
        avg_filename = f"{image_name}_avg_{ksize}x{ksize}.tif"
        save_image(avg_filtered, os.path.join(output_dir, avg_filename))

        for sigma in sigmas:
            # Filtr Gaussa
            gauss_filtered = apply_gaussian_filter(image, ksize, sigma)
            gauss_filename = f"{image_name}_gauss_{ksize}x{ksize}_sigma{sigma}.tif"
            save_image(gauss_filtered, os.path.join(output_dir, gauss_filename))

            # Wyświetlanie wyników: oryginał, filtr uśredniający, filtr Gaussa
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(image, cmap='gray')
            plt.title('Oryginał')
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(avg_filtered, cmap='gray')
            plt.title(f'Uśredniający {ksize}x{ksize}')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(gauss_filtered, cmap='gray')
            plt.title(f'Gauss {ksize}x{ksize}, σ={sigma}')
            plt.axis('off')

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    input_dir = 'Images'           # Katalog z obrazami wejściowymi
    output_dir = 'Images-converted-Z10'  # Katalog na wyniki
    os.makedirs(output_dir, exist_ok=True)

    kernel_sizes = [3, 5, 7]       # Rozmiary filtrów
    sigmas = [0.5, 1.0, 2.0]       # Odchylenia standardowe Gaussa
    images = ['characters_test_pattern.tif', 'zoneplate.tif']  # Lista plików do przetworzenia

    for img_name in images:
        print(f'Przetwarzanie obrazu: {img_name}')
        show_and_save_filters(os.path.join(input_dir, img_name), kernel_sizes, sigmas, output_dir)
