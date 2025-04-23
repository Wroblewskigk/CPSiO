import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def local_histogram_equalization(image, mask_size):
    height, width = image.shape
    pad = mask_size // 2
    padded = np.pad(image, pad, mode='reflect')
    equalized = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            window = padded[i:i + mask_size, j:j + mask_size]
            hist, _ = np.histogram(window, bins=256, range=(0, 255))
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-7)
            equalized[i, j] = cdf[image[i, j]]

    return equalized.astype(np.uint8)

def local_statistics_enhancement(image, mask_size, a=1.5, gamma=1.0):
    pad = mask_size // 2
    padded = np.pad(image.astype(float), pad, mode='reflect')
    enhanced = np.zeros_like(image, dtype=float)

    # Manual mean filtering
    kernel = np.ones((mask_size, mask_size)) / (mask_size ** 2)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i + mask_size, j:j + mask_size]
            enhanced[i, j] = np.sum(window * kernel)

    sharpened = image + a * (image - enhanced)
    sharpened = np.clip(sharpened, 0, 255)

    if gamma != 1.0:
        sharpened = 255 * (sharpened / 255) ** gamma

    return sharpened.astype(np.uint8)


def main():
    # Wczytaj obraz
    try:
        image = np.array(Image.open('hidden-symbols.tif').convert('L'))
    except:
        # Generuj przykładowy obraz
        image = np.zeros((200, 200), dtype=np.uint8)
        for i in range(200):
            for j in range(200):
                if (i - 100) ** 2 + (j - 100) ** 2 < 900:
                    image[i, j] = 150
        image = image + np.random.normal(0, 30, image.shape).astype(np.uint8)

    mask_sizes = [3, 5, 7, 9, 11]

    for size in mask_sizes:
        eq = local_histogram_equalization(image, size)
        enhanced = local_statistics_enhancement(image, size)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Oryginał')
        plt.subplot(1, 3, 2), plt.imshow(eq, cmap='gray'), plt.title(f'Lokalne wyrównanie ({size}x{size})')
        plt.subplot(1, 3, 3), plt.imshow(enhanced, cmap='gray'), plt.title(f'Poprawa statystyczna ({size}x{size})')
        plt.show()


if __name__ == "__main__":
    main()
