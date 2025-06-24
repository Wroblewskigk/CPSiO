import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

def apply_average_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.mean(window)
    return filtered.astype(np.uint8)

def gaussian_kernel(kernel_size, sigma=1.0):
    ax = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_filter(image, kernel_size, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.sum(window * kernel)
    return filtered.astype(np.uint8)

def plot_and_save_grid(image, kernel_sizes, sigmas, image_name, output_dir):
    rows = len(kernel_sizes)
    cols = 2 + len(sigmas)  # Oryginał + filtr średni + filtry Gaussa dla każdej sigma

    plt.figure(figsize=(4*cols, 4*rows))

    for row_idx, ksize in enumerate(kernel_sizes):
        avg_filtered = apply_average_filter(image, ksize)
        # Kolumna 0: oryginał (tylko w pierwszym wierszu, żeby nie dublować)
        plt.subplot(rows, cols, row_idx*cols + 1)
        plt.imshow(image, cmap='gray')
        if row_idx == 0:
            plt.title('Oryginał')
        plt.axis('off')

        # Kolumna 1: filtr uśredniający
        plt.subplot(rows, cols, row_idx*cols + 2)
        plt.imshow(avg_filtered, cmap='gray')
        plt.title(f'Uśredniający {ksize}x{ksize}')
        plt.axis('off')

        # Kolumny 2+: filtry Gaussa dla każdego sigma
        for col_idx, sigma in enumerate(sigmas):
            gauss_filtered = apply_gaussian_filter(image, ksize, sigma)
            plt.subplot(rows, cols, row_idx*cols + 3 + col_idx)
            plt.imshow(gauss_filtered, cmap='gray')
            if row_idx == 0:
                plt.title(f'Gauss σ={sigma}')
            plt.axis('off')

    plt.tight_layout()
    filename = f"{image_name}_comparison_grid.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def process_images_grid(input_dir, output_dir, kernel_sizes, sigmas, image_files):
    os.makedirs(output_dir, exist_ok=True)

    for image_file in image_files:
        print(f'Przetwarzanie obrazu: {image_file}')
        image_path = os.path.join(input_dir, image_file)
        image_name = os.path.splitext(image_file)[0]
        image = np.array(Image.open(image_path).convert('L'))

        plot_and_save_grid(image, kernel_sizes, sigmas, image_name, output_dir)

if __name__ == "__main__":
    input_dir = 'Images'
    output_dir = 'Images-converted-Z10'
    kernel_sizes = [3, 7, 11, 19]
    sigmas = [0.5, 1.0, 2.0]
    images = ['characters_test_pattern.tif', 'zoneplate.tif']

    process_images_grid(input_dir, output_dir, kernel_sizes, sigmas, images)
